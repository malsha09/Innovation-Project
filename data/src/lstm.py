import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import zipfile

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


TARGET = "nat_demand"
RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE):
    tf.keras.utils.set_random_seed(seed)


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def detect_datetime_col(df: pd.DataFrame):
    candidates = [c for c in df.columns if c.lower() in {"datetime", "timestamp", "date", "time"}]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None


def add_calendar_features(df: pd.DataFrame, dt_col: str):
    dt = pd.to_datetime(df[dt_col])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def make_windows(feature_frame: pd.DataFrame, target_series: pd.Series, lookback: int, horizon: int):
    X, y = [], []
    for i in range(lookback, len(feature_frame) - horizon + 1):
        X.append(feature_frame.iloc[i - lookback:i].values)
        y.append(target_series.iloc[i:i + horizon].values)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return X, y


def build_model(kind: str, lookback: int, n_features: int, horizon: int,
                width: int = 96, depth: int = 1, dropout: float = 0.2,
                lr: float = 1e-3, l2: float = 1e-4):
    Cell = tf.keras.layers.LSTM if kind.lower() == "lstm" else tf.keras.layers.GRU
    reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None

    inputs = tf.keras.Input(shape=(lookback, n_features))
    x = inputs
    for _ in range(max(0, depth - 1)):
        x = Cell(width, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = Cell(width, return_sequences=False, kernel_regularizer=reg, recurrent_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(horizon)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV, NPZ, or NPZ.ZIP."""
    path = Path(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".npz":
        npz = np.load(path)
        return pd.DataFrame({k: npz[k] for k in npz.files})

    if path.suffix == ".zip":
        # Assume it contains a single .npz file
        with zipfile.ZipFile(path, "r") as z:
            npz_files = [f for f in z.namelist() if f.endswith(".npz")]
            if not npz_files:
                raise ValueError("No .npz file found inside zip archive")
            extract_path = path.parent / npz_files[0]
            z.extract(npz_files[0], path.parent)
            npz = np.load(extract_path)
            return pd.DataFrame({k: npz[k] for k in npz.files})

    raise ValueError(f"Unsupported file format: {path}")


def main(args):
    set_seed(RANDOM_STATE)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load & sort data
    df = load_dataset(args.data)
    dt_col = detect_datetime_col(df)
    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df.sort_values(dt_col, inplace=True)
        df.set_index(dt_col, inplace=True)
    else:
        print("[WARN] No datetime/timestamp column detected. Using current row order.")

    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' not found. Columns: {list(df.columns)}")

    # ---- Optional calendar features
    if args.add_calendar:
        if dt_col is None:
            print("[WARN] --add-calendar requested but no datetime column found; skipping calendar features.")
        else:
            df = add_calendar_features(df.reset_index(), dt_col="index").set_index("index")

    # ---- Features
    base_feature_cols = [c for c in df.columns if c != TARGET]
    df["nat_demand_feat"] = df[TARGET].astype(float)
    feature_cols = base_feature_cols + ["nat_demand_feat"]

    # ---- Split train/val/test
    N = len(df)
    i_train = int(0.8 * N)
    i_val = int(0.9 * N)

    df_train = df.iloc[:i_train].copy()
    df_val = df.iloc[i_train:i_val].copy()
    df_test = df.iloc[i_val:].copy()

    scaler = StandardScaler().fit(df_train[feature_cols])
    X_train_frame = pd.DataFrame(scaler.transform(df_train[feature_cols]),
                                 index=df_train.index, columns=feature_cols)
    X_val_frame = pd.DataFrame(scaler.transform(df_val[feature_cols]),
                               index=df_val.index, columns=feature_cols)
    X_test_frame = pd.DataFrame(scaler.transform(df_test[feature_cols]),
                                index=df_test.index, columns=feature_cols)

    y_train_series = df_train[TARGET].astype(float)
    y_val_series = df_val[TARGET].astype(float)
    y_test_series = df_test[TARGET].astype(float)

    # ---- Windows
    T = int(args.lookback)
    H = int(args.horizon)

    Xtr, ytr = make_windows(X_train_frame, y_train_series, T, H)
    Xva, yva = make_windows(X_val_frame, y_val_series, T, H)
    Xte, yte = make_windows(X_test_frame, y_test_series, T, H)

    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        raise RuntimeError("Not enough rows after windowing. Try reducing --lookback or check your data length.")

    # ---- Model
    model = build_model(kind=args.model, lookback=T, n_features=Xtr.shape[-1], horizon=H,
                        width=args.width, depth=args.depth, dropout=args.dropout,
                        lr=args.lr, l2=args.l2)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=max(1, args.patience // 2), verbose=1)
    ]

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
        callbacks=callbacks
    )

    # ---- Evaluation
    def evaluate(X, y):
        preds = model.predict(X, verbose=0)
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.ravel()
            y = y.ravel()
        return {"MAE": float(mean_absolute_error(y, preds)),
                "RMSE": rmse(y, preds),
                "MAPE": mape(y, preds)}

    metrics_val = evaluate(Xva, yva)
    metrics_test = evaluate(Xte, yte)

    metrics_df = pd.DataFrame([
        {"Model": args.model.upper(), "Horizon": H, "Split": "val", **metrics_val},
        {"Model": args.model.upper(), "Horizon": H, "Split": "test", **metrics_test},
    ])
    metrics_df.to_csv(outdir / "phase3_metrics.csv", index=False)

    # ---- Save model & config
    model.save(outdir / f"{args.model}_h{H}.keras")
    joblib.dump(scaler, outdir / "scaler.joblib")

    config = dict(model=args.model, horizon=H, lookback=T, width=args.width, depth=args.depth,
                  dropout=args.dropout, lr=args.lr, l2=args.l2, batch=args.batch,
                  epochs=args.epochs, patience=args.patience, features=feature_cols)
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    print("=== Phase 3 complete ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .csv, .npz, or .npz.zip dataset")
    parser.add_argument("--outdir", default="outputs/seq", help="Directory to save models & metrics")
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in hours (1 or 24 typical)")
    parser.add_argument("--lookback", type=int, default=168, help="Lookback window size (hours)")
    parser.add_argument("--width", type=int, default=96, help="Hidden width per recurrent layer")
    parser.add_argument("--depth", type=int, default=1, help="Number of recurrent layers (1-2 recommended)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=8, help="EarlyStopping patience (epochs)")
    parser.add_argument("--add-calendar", dest="add_calendar", action="store_true",
                        help="Add calendar features (hour/dow/month/weekend) if a datetime column exists")
    parser.add_argument("--no-add-calendar", dest="add_calendar", action="store_false",
                        help="Disable calendar features")
    parser.set_defaults(add_calendar=True)
    args = parser.parse_args()
    main(args)

