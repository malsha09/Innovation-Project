# Phase 3 — Sequence Models (TCN + Transformer)
# Notebook-style Python script (use in Colab / Jupyter)
# Save as: notebooks/phase3_sequence_models.ipynb

"""
Overview
--------
This notebook trains and tunes a TCN and a Transformer to predict the next 24 hours of
`nat_demand` using past sequence data (default lookback 168 hours).

How to run
----------
1. Open in Colab or Jupyter.
2. Upload `cleaned_electricity_demand.csv` to the notebook working directory (Colab: left Files panel -> upload).
3. Run cells in order.

Notes
-----
- Framework: TensorFlow / Keras
- Outputs saved to `phase3_outputs/`
- If running in Colab you may need to `!pip install -q tensorflow` if not present.

"""

# %%
# Install (uncomment if needed in Colab)
# !pip install -q tensorflow

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(1337)
tf.random.set_seed(1337)

# %%
# ---------- CONFIG ----------
CSV_PATH = 'cleaned_electricity_demand.csv'  # upload this file in Colab or put in repo
TARGET = 'nat_demand'
TIME = 'datetime'

TRAIN_END = pd.Timestamp('2018-12-31 23:00:00')
VAL_END   = pd.Timestamp('2019-12-31 23:00:00')

LOOKBACKS = [168]            # you can add 72, 336 if you want more experiments
HIDDEN_SIZES = [64, 128]
DROPOUTS = [0.1, 0.3]
LRATES = [1e-3, 5e-4]
BATCHES = [128, 256]
EPOCHS = 30
PATIENCE = 5
HORIZON = 24
OUTDIR = 'phase3_outputs'
os.makedirs(OUTDIR, exist_ok=True)

# %%
# ---------- HELPERS ----------

def add_cyclical_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'hour' in out.columns:
        out['hour_sin'] = np.sin(2*np.pi*out['hour']/24)
        out['hour_cos'] = np.cos(2*np.pi*out['hour']/24)
    if 'day_of_week' in out.columns:
        out['dow_sin'] = np.sin(2*np.pi*out['day_of_week']/7)
        out['dow_cos'] = np.cos(2*np.pi*out['day_of_week']/7)
    if 'month' in out.columns:
        out['month_sin'] = np.sin(2*np.pi*out['month']/12)
        out['month_cos'] = np.cos(2*np.pi*out['month']/12)
    return out


def train_val_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.sort_values(TIME).reset_index(drop=True)
    train_df = df[df[TIME] <= TRAIN_END]
    val_df   = df[(df[TIME] > TRAIN_END) & (df[TIME] <= VAL_END)]
    test_df  = df[df[TIME] > VAL_END]
    return train_df, val_df, test_df


def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    Xtr = train_df[feature_cols].astype(float).values
    Xv  = val_df[feature_cols].astype(float).values
    Xt  = test_df[feature_cols].astype(float).values
    mean_ = Xtr.mean(axis=0)
    std_ = Xtr.std(axis=0)
    std_[std_ == 0] = 1.0
    Xtr = (Xtr - mean_) / std_
    Xv  = (Xv - mean_) / std_
    Xt  = (Xt - mean_) / std_
    return Xtr, Xv, Xt, mean_, std_


def make_windows(X: np.ndarray, y: np.ndarray, lookback: int, horizon: int):
    Xs, Ys = [], []
    n = len(y)
    last_start = n - lookback - horizon + 1
    for s in range(last_start):
        e = s + lookback
        Xs.append(X[s:e])
        if horizon == 1:
            Ys.append(y[e])
        else:
            Ys.append(y[e:e+horizon])
    return np.stack(Xs), np.array(Ys)


def mae(a,b):
    return float(np.mean(np.abs(a-b)))

def rmse(a,b):
    return float(np.sqrt(np.mean((a-b)**2)))

def mape(a,b):
    eps = 1e-8
    return float(np.mean(np.abs((a-b)/(np.abs(a)+eps)))*100)


def eval_metrics(y_true, y_pred, horizon):
    if horizon == 1:
        y_pred = y_pred.flatten()
        return {'MAE': mae(y_true, y_pred), 'RMSE': rmse(y_true, y_pred), 'MAPE': mape(y_true, y_pred)}
    maes = []
    rmses = []
    mapes = []
    for i in range(horizon):
        maes.append(mae(y_true[:,i], y_pred[:,i]))
        rmses.append(rmse(y_true[:,i], y_pred[:,i]))
        mapes.append(mape(y_true[:,i], y_pred[:,i]))
    return {'MAE_mean': float(np.mean(maes)), 'RMSE_mean': float(np.mean(rmses)), 'MAPE_mean': float(np.mean(mapes))}

# %%
# ---------- Model Builders ----------

def build_tcn(n_features: int, lookback: int, hidden: int = 64, dropout: float = 0.2, horizon: int = 24):
    inp = layers.Input(shape=(lookback, n_features))
    x = inp
    # stack of residual dilated conv blocks
    for channels, dilation in [(hidden,1),(hidden,2),(hidden,4),(hidden,8)]:
        res = x
        x = layers.Conv1D(channels, kernel_size=3, padding='causal', dilation_rate=dilation, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(channels, kernel_size=3, padding='causal', dilation_rate=dilation, activation='relu')(x)
        if res.shape[-1] != x.shape[-1]:
            res = layers.Conv1D(channels, kernel_size=1, padding='same')(res)
        x = layers.Add()([x, res])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hidden, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1 if horizon==1 else horizon)(x)
    model = keras.Model(inp, out)
    return model


def build_transformer_encoder(n_features: int, lookback: int, dim: int = 64, heads: int = 4, ff_dim: int = 128, dropout: float = 0.1, horizon: int = 24):
    inp = layers.Input(shape=(lookback, n_features))
    # linear projection
    x = layers.Dense(dim)(inp)
    # simple positional encoding (trainable)
    positions = tf.range(start=0, limit=lookback, delta=1)
    pos_emb = layers.Embedding(input_dim=lookback, output_dim=dim)(positions)
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x = x + pos_emb

    # two encoder blocks
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
    model = keras.Model(inp, out)
    return model

# %%
# ---------- Load data and prepare features ----------
print('Loading CSV...')
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Place {CSV_PATH} in the notebook working directory and re-run")

df = pd.read_csv(CSV_PATH, parse_dates=[TIME])
print('Rows, cols:', df.shape)

# add cyclical features
df = add_cyclical_feats(df)

# identify feature columns (drop target and time)
drop_cols = {TARGET, TIME}
feature_cols = [c for c in df.columns if c not in drop_cols]
print('Feature columns:', feature_cols)

train_df, val_df, test_df = train_val_test_split(df)
print('Train/Val/Test sizes:', len(train_df), len(val_df), len(test_df))

# scale features
Xtr, Xv, Xt, mean_, std_ = scale_features(train_df, val_df, test_df, feature_cols)
ytr = train_df[TARGET].values.astype(float)
yv = val_df[TARGET].values.astype(float)
yt  = test_df[TARGET].values.astype(float)

# %%
# ---------- Run experiments (grid search) ----------
results = []

for lookback in LOOKBACKS:
    print('\n--- lookback', lookback)
    # make windows
    Xtr_win, ytr_win = make_windows(Xtr, ytr, lookback, HORIZON)
    Xv_win,  yv_win  = make_windows(Xv,  yv,  lookback, HORIZON)
    Xt_win,  yt_win  = make_windows(Xt,  yt,  lookback, HORIZON)
    print('windows shapes: ', Xtr_win.shape, ytr_win.shape)

    for hidden in HIDDEN_SIZES:
        for dropout in DROPOUTS:
            for lr in LRATES:
                for batch in BATCHES:

                    # ---- TCN ----
                    tcn = build_tcn(n_features=Xtr_win.shape[2], lookback=lookback, hidden=hidden, dropout=dropout, horizon=HORIZON)
                    tcn.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
                    cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
                    print(f"Training TCN lb={lookback} hid={hidden} do={dropout} lr={lr} b={batch}")
                    tcn.fit(Xtr_win, ytr_win, validation_data=(Xv_win, yv_win), epochs=EPOCHS, batch_size=batch, callbacks=cb, verbose=2)
                    yv_pred = tcn.predict(Xv_win, verbose=0)
                    yt_pred = tcn.predict(Xt_win, verbose=0)
                    metrics_val = eval_metrics(yv_win, yv_pred, HORIZON)
                    metrics_test = eval_metrics(yt_win, yt_pred, HORIZON)
                    print('TCN metrics val:', metrics_val, 'test:', metrics_test)
                    results.append({'model':'TCN','lookback':lookback,'hidden':hidden,'dropout':dropout,'lr':lr,'batch':batch,'val':metrics_val,'test':metrics_test})
                    # save model snapshot
                    mname = f"tcn_lb{lookback}_h{hidden}_d{int(dropout*100)}_lr{lr}_b{batch}.keras"
                    tcn.save(os.path.join(OUTDIR, mname))

                    # ---- Transformer ----
                    trf = build_transformer_encoder(n_features=Xtr_win.shape[2], lookback=lookback, dim=hidden, heads=4, ff_dim=hidden*2, dropout=dropout, horizon=HORIZON)
                    trf.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
                    print(f"Training Transformer lb={lookback} dim={hidden} do={dropout} lr={lr} b={batch}")
                    trf.fit(Xtr_win, ytr_win, validation_data=(Xv_win, yv_win), epochs=EPOCHS, batch_size=batch, callbacks=cb, verbose=2)
                    yv_pred = trf.predict(Xv_win, verbose=0)
                    yt_pred = trf.predict(Xt_win, verbose=0)
                    metrics_val = eval_metrics(yv_win, yv_pred, HORIZON)
                    metrics_test = eval_metrics(yt_win, yt_pred, HORIZON)
                    print('Transformer metrics val:', metrics_val, 'test:', metrics_test)
                    results.append({'model':'Transformer','lookback':lookback,'dim':hidden,'dropout':dropout,'lr':lr,'batch':batch,'val':metrics_val,'test':metrics_test})
                    mname = f"trf_lb{lookback}_d{hidden}_do{int(dropout*100)}_lr{lr}_b{batch}.keras"
                    trf.save(os.path.join(OUTDIR, mname))

# %%
# ---------- Save results ----------
with open(os.path.join(OUTDIR,'phase3_sequence_results.json'),'w') as f:
    json.dump(results, f, indent=2)
print('Saved results to', os.path.join(OUTDIR,'phase3_sequence_results.json'))

# %%
# ---------- Quick plotting helper: compare best models ----------
# load results and pick best by test RMSE_mean for each model type
with open(os.path.join(OUTDIR,'phase3_sequence_results.json')) as f:
    all_res = json.load(f)

# simplify selection: compute a score key
for r in all_res:
    if 'test' in r and 'RMSE_mean' in r['test']:
        r['score'] = r['test']['RMSE_mean']
    else:
        r['score'] = float('inf')

best_tcn = min([x for x in all_res if x['model']=='TCN'], key=lambda z: z['score']) if any(x['model']=='TCN' for x in all_res) else None
best_trf = min([x for x in all_res if x['model']=='Transformer'], key=lambda z: z['score']) if any(x['model']=='Transformer' for x in all_res) else None

print('Best TCN config:', best_tcn)
print('Best Transformer config:', best_trf)

# If best models saved, load and plot a prediction slice
def plot_preds(model_path, X_test_win, y_test_win, title=''):
    mdl = keras.models.load_model(model_path)
    yp = mdl.predict(X_test_win[:200])
    # for H=24 we plot the first horizon step vs actual first step (or mean)
    if yp.ndim==2:
        pred = yp[:,0]
        true = y_test_win[:200,0]
    else:
        pred = yp.flatten(); true = y_test_win[:len(pred)]
    plt.figure(figsize=(12,4)); plt.plot(true, label='true'); plt.plot(pred, label='pred'); plt.title(title); plt.legend(); plt.show()

# Try plotting if the best models were saved
if best_tcn:
    tcn_name = [f for f in os.listdir(OUTDIR) if f.startswith('tcn') and str(best_tcn['lookback']) in f]
    if tcn_name:
        print('Plotting best TCN sample...')
        plot_preds(os.path.join(OUTDIR,tcn_name[0]), Xt_win, yt_win, 'Best TCN sample')

if best_trf:
    trf_name = [f for f in os.listdir(OUTDIR) if f.startswith('trf') and str(best_trf['lookback']) in f]
    if trf_name:
        print('Plotting best Transformer sample...')
        plot_preds(os.path.join(OUTDIR,trf_name[0]), Xt_win, yt_win, 'Best Transformer sample')

# %%
# ---------- Wrap-up ----------
print('\nPhase 3 sequence experiments finished.\n')
print('Check folder:', OUTDIR, 'for saved models and phase3_sequence_results.json')

"""
# End of notebook content
