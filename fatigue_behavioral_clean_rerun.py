"""
Clean rerun for behavioral driver drowsiness detection.
Author: generated for Ajayi et al. project audit
Purpose: one defensible experimental pipeline using real driver IDs from NTHU-style filenames.

Expected input CSV columns:
    image_name, ear, label
Example image_name:
    001_glasses_sleepyCombination_1000_drowsy.jpg

What this script does:
1. Extracts real driver ID from the first three digits of image_name.
2. Extracts scenario and frame number from image_name.
3. Computes engineered EAR features within each driver/scenario stream.
4. Generates temporal sequences without crossing scenario boundaries.
5. Evaluates models using Leave-One-Driver-Out validation.
6. Saves all results to CSV files.

Usage:
    python fatigue_behavioral_clean_rerun.py --data eye_features.csv --out results_clean --seq_lens 10 20 30

Notes:
- TensorFlow is optional. If unavailable, only Logistic Regression and Random Forest run.
- LSTM/BiLSTM run only if TensorFlow is installed.
"""

from __future__ import annotations

import argparse
import os
import re
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


FEATURE_COLUMNS = ["ear", "ear_rolling_mean", "ear_delta", "eye_closed"]
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


def parse_filename(name: str) -> Tuple[str, str, int]:
    """Return (driver_id, scenario_id, frame_index)."""
    base = os.path.basename(str(name))
    m_driver = re.match(r"^(\d{3})_", base)
    driver_id = m_driver.group(1) if m_driver else "unknown"

    stem = os.path.splitext(base)[0]
    parts = stem.split("_")

    # Expected: driver + scenario tokens + frame number + label token
    frame_idx = -1
    frame_pos = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            frame_idx = int(parts[i])
            frame_pos = i
            break

    if frame_pos is None:
        scenario = "_".join(parts[1:-1]) if len(parts) > 2 else "unknown"
    else:
        scenario = "_".join(parts[1:frame_pos]) if frame_pos > 1 else "unknown"

    return driver_id, scenario, frame_idx


def load_and_prepare(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required = {"image_name", "ear", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    parsed = df["image_name"].apply(parse_filename)
    df["driver_id"] = parsed.apply(lambda x: x[0])
    df["scenario"] = parsed.apply(lambda x: x[1])
    df["frame_idx"] = parsed.apply(lambda x: x[2])
    df["stream_id"] = df["driver_id"].astype(str) + "__" + df["scenario"].astype(str)

    df = df.sort_values(["driver_id", "scenario", "frame_idx", "image_name"]).reset_index(drop=True)

    # Feature engineering must not cross driver/scenario streams.
    df["ear_rolling_mean"] = df.groupby("stream_id")["ear"].transform(
        lambda s: s.rolling(window=5, min_periods=1).mean()
    )
    df["ear_delta"] = df.groupby("stream_id")["ear"].diff().fillna(0)
    df["eye_closed"] = (df["ear"] < 0.21).astype(int)

    # Keep valid binary labels only.
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    return df


def make_sequences(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences within each stream_id. Returns X, y, groups(driver_id), stream_ids."""
    X_list, y_list, g_list, s_list = [], [], [], []

    for stream_id, gdf in df.groupby("stream_id", sort=False):
        gdf = gdf.sort_values("frame_idx")
        feats = gdf[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = gdf["label"].to_numpy(dtype=np.int32)
        drivers = gdf["driver_id"].to_numpy()

        if len(gdf) < seq_len:
            continue

        for start in range(0, len(gdf) - seq_len + 1):
            end = start + seq_len
            X_list.append(feats[start:end])
            y_list.append(labels[end - 1])  # final-frame label
            g_list.append(drivers[end - 1])
            s_list.append(stream_id)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    groups = np.asarray(g_list)
    streams = np.asarray(s_list)
    return X, y, groups, streams


def build_lstm(input_shape: Tuple[int, int], bidirectional: bool = False) -> "Sequential":
    model = Sequential()
    model.add(Input(shape=input_shape))
    if bidirectional:
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
    else:
        model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_drowsy": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_drowsy": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_drowsy": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        try:
            out["auroc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            out["auroc"] = np.nan
        try:
            out["auprc"] = average_precision_score(y_true, y_prob)
        except Exception:
            out["auprc"] = np.nan
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
    return out


def scale_sequence_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    n_train, t, f = X_train.shape
    n_test = X_test.shape[0]
    X_train_2d = X_train.reshape(-1, f)
    X_test_2d = X_test.reshape(-1, f)
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, t, f)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, t, f)
    return X_train_scaled, X_test_scaled


def run_logo_experiment(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seq_len: int, out_dir: str) -> pd.DataFrame:
    logo = LeaveOneGroupOut()
    rows = []
    cm_rows = []

    classical_models = {
        "LogisticRegression": Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    fold_id = 0
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        fold_id += 1
        test_driver = np.unique(groups[test_idx]).tolist()[0]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Flatten sequence for classical baselines.
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        for model_name, estimator in classical_models.items():
            set_seed()
            est = clone(estimator)
            est.fit(X_train_flat, y_train)
            if hasattr(est, "predict_proba"):
                y_prob = est.predict_proba(X_test_flat)[:, 1]
            else:
                y_prob = est.predict(X_test_flat).astype(float)
            metrics = compute_metrics(y_test, y_prob)
            row = {"seq_len": seq_len, "fold": fold_id, "test_driver": test_driver, "model": model_name, **metrics}
            rows.append(row)
            tn, fp, fn, tp = confusion_matrix(y_test, (y_prob >= 0.5).astype(int), labels=[0, 1]).ravel()
            cm_rows.append({"seq_len": seq_len, "fold": fold_id, "test_driver": test_driver, "model": model_name,
                            "tn": tn, "fp": fp, "fn": fn, "tp": tp})

        if TF_AVAILABLE:
            X_train_s, X_test_s = scale_sequence_train_test(X_train, X_test)
            for model_name, bidir in [("LSTM", False), ("BiLSTM", True)]:
                set_seed()
                model = build_lstm(input_shape=(X.shape[1], X.shape[2]), bidirectional=bidir)
                callbacks = [EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)]
                model.fit(
                    X_train_s, y_train,
                    epochs=50,
                    batch_size=64,
                    validation_split=0.15,
                    callbacks=callbacks,
                    verbose=0,
                )
                y_prob = model.predict(X_test_s, verbose=0).ravel()
                metrics = compute_metrics(y_test, y_prob)
                row = {"seq_len": seq_len, "fold": fold_id, "test_driver": test_driver, "model": model_name, **metrics}
                rows.append(row)
                tn, fp, fn, tp = confusion_matrix(y_test, (y_prob >= 0.5).astype(int), labels=[0, 1]).ravel()
                cm_rows.append({"seq_len": seq_len, "fold": fold_id, "test_driver": test_driver, "model": model_name,
                                "tn": tn, "fp": fp, "fn": fn, "tp": tp})

    results = pd.DataFrame(rows)
    cm = pd.DataFrame(cm_rows)
    results.to_csv(os.path.join(out_dir, f"fold_results_seq{seq_len}.csv"), index=False)
    cm.to_csv(os.path.join(out_dir, f"confusion_counts_seq{seq_len}.csv"), index=False)
    return results


def summarize(results: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    metric_cols = ["accuracy", "precision_drowsy", "recall_drowsy", "f1_drowsy", "precision_macro", "recall_macro", "f1_macro", "auroc", "auprc"]
    summary = results.groupby(["seq_len", "model"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(os.path.join(out_dir, "summary_results.csv"), index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to eye_features.csv")
    parser.add_argument("--out", default="results_clean", help="Output directory")
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[10, 20, 30])
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed()

    df = load_and_prepare(args.data)
    print("Loaded rows:", len(df))
    print("Driver counts:\n", df["driver_id"].value_counts().sort_index())
    print("Label counts:\n", df["label"].value_counts().sort_index())
    print("TensorFlow available:", TF_AVAILABLE)

    inventory = {
        "n_rows": [len(df)],
        "drivers": [",".join(sorted(df["driver_id"].unique()))],
        "label_0": [int((df["label"] == 0).sum())],
        "label_1": [int((df["label"] == 1).sum())],
        "tensorflow_available": [TF_AVAILABLE],
    }
    pd.DataFrame(inventory).to_csv(os.path.join(args.out, "dataset_inventory.csv"), index=False)

    all_results = []
    for seq_len in args.seq_lens:
        X, y, groups, streams = make_sequences(df, seq_len=seq_len)
        print(f"\nSEQ_LEN={seq_len}: X={X.shape}, y={y.shape}, groups={dict(pd.Series(groups).value_counts().sort_index())}")
        seq_inv = pd.DataFrame({
            "seq_len": [seq_len],
            "n_sequences": [len(y)],
            "label_0": [int((y == 0).sum())],
            "label_1": [int((y == 1).sum())],
            "n_groups": [len(np.unique(groups))],
            "groups": [",".join(sorted(np.unique(groups)))],
        })
        seq_inv.to_csv(os.path.join(args.out, f"sequence_inventory_seq{seq_len}.csv"), index=False)
        results = run_logo_experiment(X, y, groups, seq_len=seq_len, out_dir=args.out)
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(args.out, "all_fold_results.csv"), index=False)
    summary = summarize(combined, args.out)
    print("\nSummary results:")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.out}")


if __name__ == "__main__":
    main()
