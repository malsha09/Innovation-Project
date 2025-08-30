#!/usr/bin/env python3
"""
Phase 2 — Feature Engineering & Baseline Modelling

Run:
  python src/phase2_pipeline.py --data data/cleaned_electricity_demand.csv --outdir outputs --use-xgb 0

Outputs:
  - <outdir>/features_ready.csv
  - <outdir>/phase2_metrics.csv
  - <outdir>/lr_model.joblib
  - <outdir>/tree_model.joblib
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump

# Optional XGBoost; fallback is HistGradientBoosting
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_XGB = False


TARGET = "nat_demand"
RANDOM_STATE = 42


# ---------- Metrics ----------
def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100.0

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))  # version-safe

def metrics_dict(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE%": mape(y_true, y_pred),
    }

def aligned_metrics(y_true_series, y_pred_series):
    aligned = pd.concat({"y": y_true_series, "pred": y_pred_series}, axis=1).dropna()
    return metrics_dict(aligned["y"].values, aligned["pred"].values)


# ---------- Feature Engineering ----------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Lags of target (1h, 24h, 168h = 1 week)
    for L in [1, 24, 168]:
        df[f"lag_{L}"] = df[TARGET].shift(L)

    # Rolling mean/std (past only)
    for W in [3, 6, 12, 24, 168]:
        df[f"roll_mean_{W}"] = df[TARGET].shift(1).rolling(window=W).mean()
        df[f"roll_std_{W}"]  = df[TARGET].shift(1).rolling(window=W).std()

    # Weather summaries
    for pref in ["T2M_", "QV2M_", "TQL_", "W2M_"]:
        cols = [c for c in df.columns if c.startswith(pref)]
        if len(cols) >= 2:
            df[f"{pref}mean"] = df[cols].mean(axis=1)
            df[f"{pref}min"]  = df[cols].min(axis=1)
            df[f"{pref}max"]  = df[cols].max(axis=1)

    df = df.dropna().reset_index(drop=True)
    return df


# ---------- Time split ----------
def time_split(df_feat: pd.DataFrame, train_frac=0.80, val_frac=0.10):
    n = len(df_feat)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train = df_feat.iloc[:train_end]
    val   = df_feat.iloc[train_end:val_end]
    test  = df_feat.iloc[val_end:]

    feature_cols = [c for c in df_feat.columns if c not in ["datetime", TARGET]]

    X_train, y_train = train[feature_cols], train[TARGET]
    X_val,   y_val   = val[feature_cols],   val[TARGET]
    X_test,  y_test  = test[feature_cols],  test[TARGET]

    return (train, val, test), (X_train, y_train, X_val, y_val, X_test, y_test), feature_cols


# ---------- Models ----------
def build_lr_pipeline():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

def build_tree_pipeline(use_xgb=False):
    if use_xgb and HAS_XGB:
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
        name = "XGBoost"
    else:
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        )
        name = "HistGB"
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])
    return pipe, name


# ---------- Main ----------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.data)
    print(f"Rows: {len(df):,}  Columns: {len(df.columns)}")

    print("Engineering features (lags, rolling, weather summaries)...")
    df_feat = engineer_features(df)

    print("Time-based splitting (80/10/10)...")
    (train, val, test), (X_train, y_train, X_val, y_val, X_test, y_test), feature_cols = time_split(df_feat)

    # ----- Baselines -----
    print("Evaluating naive baselines...")
    val_pred_naive_1h  = val[TARGET].shift(1)
    test_pred_naive_1h = test[TARGET].shift(1)
    base1_val  = aligned_metrics(val[TARGET],  val_pred_naive_1h)
    base1_test = aligned_metrics(test[TARGET], test_pred_naive_1h)

    val_pred_naive_24h  = val[TARGET].shift(24)
    test_pred_naive_24h = test[TARGET].shift(24)
    base24_val  = aligned_metrics(val[TARGET],  val_pred_naive_24h)
    base24_test = aligned_metrics(test[TARGET], test_pred_naive_24h)

    # ----- Linear Regression -----
    print("Training Linear Regression...")
    lr = build_lr_pipeline()
    lr.fit(X_train, y_train)
    lr_val_pred  = lr.predict(X_val)
    lr_test_pred = lr.predict(X_test)
    lr_val  = metrics_dict(y_val,  lr_val_pred)
    lr_test = metrics_dict(y_test, lr_test_pred)

    # ----- Tree model -----
    print(f"Training tree model (use_xgb={args.use_xgb and HAS_XGB})...")
    tree, tree_name = build_tree_pipeline(use_xgb=bool(args.use_xgb))
    tree.fit(X_train, y_train)
    tree_val_pred  = tree.predict(X_val)
    tree_test_pred = tree.predict(X_test)
    tree_val  = metrics_dict(y_val,  tree_val_pred)
    tree_test = metrics_dict(y_test, tree_test_pred)

    # ----- Summarize -----
    print("Saving outputs...")
    summary = pd.DataFrame([
        ["Naive-1h", "val",  base1_val["MAE"], base1_val["RMSE"], base1_val["MAPE%"]],
        ["Naive-1h", "test", base1_test["MAE"], base1_test["RMSE"], base1_test["MAPE%"]],
        ["Naive-24h", "val",  base24_val["MAE"], base24_val["RMSE"], base24_val["MAPE%"]],
        ["Naive-24h", "test", base24_test["MAE"], base24_test["RMSE"], base24_test["MAPE%"]],
        ["LinearRegression", "val",  lr_val["MAE"],  lr_val["RMSE"],  lr_val["MAPE%"]],
        ["LinearRegression", "test", lr_test["MAE"], lr_test["RMSE"], lr_test["MAPE%"]],
        [tree_name, "val",  tree_val["MAE"],  tree_val["RMSE"],  tree_val["MAPE%"]],
        [tree_name, "test", tree_test["MAE"], tree_test["RMSE"], tree_test["MAPE%"]],
    ], columns=["Model", "Split", "MAE", "RMSE", "MAPE%"]).round(3)

    df_feat.to_csv(outdir / "features_ready.csv", index=False)
    summary.to_csv(outdir / "phase2_metrics.csv", index=False)
    dump(lr, outdir / "lr_model.joblib")
    dump(tree, outdir / f"{tree_name.lower()}_model.joblib")

    print("\n=== Phase 2 Metrics ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cleaned_electricity_demand.csv")
    parser.add_argument("--outdir", default="outputs", help="Directory to write outputs")
    parser.add_argument("--use-xgb", type=int, default=0, help="1 = try XGBoost, 0 = fast sklearn tree")
    args = parser.parse_args()
    main(args)
