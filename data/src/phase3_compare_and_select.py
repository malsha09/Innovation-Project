#!/usr/bin/env python3
"""
Phase 3 — Advanced Models Comparison and Selection (No SHAP)

Run:
  python src/phase3_compare_and_select.py --npz data/all_datasets_compressed.npz --outdir outputs --horizons 1 24
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- metrics ----------
def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float).reshape(-1)
    y_pred = np.array(y_pred, dtype=float).reshape(-1)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100.0

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_dict(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAPE%": mape(y_true, y_pred),
    }

# ---------- model builders ----------
def build_tcn(n_features, lookback, hidden=64, dropout=0.2, horizon=1):
    inp = layers.Input(shape=(lookback, n_features))
    x = inp
    for channels, dilation in [(hidden,1),(hidden,2),(hidden,4),(hidden,8)]:
        res = x
        x = layers.Conv1D(channels, 3, padding='causal', dilation_rate=dilation, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(channels, 3, padding='causal', dilation_rate=dilation, activation='relu')(x)
        if res.shape[-1] != x.shape[-1]:
            res = layers.Conv1D(channels, 1, padding='same')(res)
        x = layers.Add()([x, res])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hidden, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1 if horizon==1 else horizon)(x)
    return keras.Model(inp, out)

def build_transformer(n_features, lookback, dim=64, heads=4, ff_dim=128, dropout=0.1, horizon=1):
    inp = layers.Input(shape=(lookback, n_features))
    x = layers.Dense(dim)(inp)
    positions = tf.range(0, lookback)
    pos_emb = layers.Embedding(input_dim=lookback, output_dim=dim)(positions)
    pos_emb = tf.expand_dims(pos_emb, 0)
    x = x + pos_emb
    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim, dropout=dropout)
        x1 = attn(x, x)
        x = layers.LayerNormalization()(x + x1)
        x1 = layers.Dense(ff_dim, activation='relu')(x)
        x1 = layers.Dropout(dropout)(x1)
        x1 = layers.Dense(dim)(x1)
        x = layers.LayerNormalization()(x + x1)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1 if horizon==1 else horizon)(x)
    return keras.Model(inp, out)

def fit_and_eval(model, name, Xtr, ytr, Xv, yv, Xt, yt, epochs=5, batch=64, outdir=Path("."), horizon=1):
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
    print(f"\nTraining {name} (H={horizon})...")
    model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=epochs, batch_size=batch, verbose=2)
    yv_pred = model.predict(Xv, verbose=0)
    yt_pred = model.predict(Xt, verbose=0)

    val_metrics  = metrics_dict(yv, yv_pred)
    test_metrics = metrics_dict(yt, yt_pred)

    model_path = outdir / f"{name}_h{horizon}.keras"
    model.save(model_path)
    np.savez_compressed(outdir / f"{name}_preds_h{horizon}.npz", yv_pred=yv_pred, yt_pred=yt_pred)
    return val_metrics, test_metrics, str(model_path)

def load_npz(npz_path, horizon):
    with np.load(npz_path) as data:
        Xtr = data[f"X_train_{horizon}"]
        ytr = data[f"y_train_{horizon}"]
        Xv  = data[f"X_val_{horizon}"]
        yv  = data[f"y_val_{horizon}"]
        Xt  = data[f"X_test_{horizon}"]
        yt  = data[f"y_test_{horizon}"]
    return Xtr, ytr, Xv, yv, Xt, yt

def maybe_load_phase2_metrics(outdir):
    p = Path(outdir) / "phase2_metrics.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df[df["Split"].str.lower()=="test"].copy()
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    phase2 = maybe_load_phase2_metrics(outdir)
    all_rows, best_by_h = [], {}

    for H in args.horizons:
        print(f"\n=== Evaluating Horizon {H} ===")
        Xtr, ytr, Xv, yv, Xt, yt = load_npz(args.npz, H)
        lookback, n_features = Xtr.shape[1], Xtr.shape[2]

        tcn = build_tcn(n_features, lookback, horizon=H)
        trf = build_transformer(n_features, lookback, horizon=H)

        tcn_val, tcn_test, tcn_path = fit_and_eval(tcn,"TCN",Xtr,ytr,Xv,yv,Xt,yt,args.epochs,args.batch,outdir,H)
        trf_val, trf_test, trf_path = fit_and_eval(trf,"Transformer",Xtr,ytr,Xv,yv,Xt,yt,args.epochs,args.batch,outdir,H)

        all_rows += [
            [H,"TCN","val",tcn_val["MAE"],tcn_val["RMSE"],tcn_val["MAPE%"],tcn_path],
            [H,"TCN","test",tcn_test["MAE"],tcn_test["RMSE"],tcn_test["MAPE%"],tcn_path],
            [H,"Transformer","val",trf_val["MAE"],trf_val["RMSE"],trf_val["MAPE%"],trf_path],
            [H,"Transformer","test",trf_test["MAE"],trf_test["RMSE"],trf_test["MAPE%"],trf_path]
        ]

        # select best on test RMSE
        best = min([("TCN",tcn_test,tcn_path),("Transformer",trf_test,trf_path)],
                   key=lambda x:(x[1]["RMSE"],x[1]["MAE"]))
        best_by_h[H] = {"model":best[0],"metrics_test":best[1],"model_path":best[2]}

    adv_df = pd.DataFrame(all_rows,columns=["Horizon","Model","Split","MAE","RMSE","MAPE%","ModelPath"]).round(4)
    adv_df.to_csv(outdir/"phase3_advanced_metrics.csv",index=False)

    if phase2 is not None:
        phase2["Horizon"] = np.nan; phase2["ModelPath"] = ""
        combined = pd.concat([phase2[["Horizon","Model","Split","MAE","RMSE","MAPE%","ModelPath"]],
                              adv_df[adv_df["Split"]=="test"]],ignore_index=True)
        combined.to_csv(outdir/"phase3_compare_with_phase2.csv",index=False)

    with open(outdir/"phase3_best_models.json","w") as f: json.dump(best_by_h,f,indent=2)
    print("\n=== Best Models by Horizon ===")
    for H,info in best_by_h.items():
        print(f"H={H}: {info['model']} | Test {info['metrics_test']} | Saved at {info['model_path']}")
    print("\nCheckpoint complete (Phase 3).")

if __name__=="__main__":
    main()