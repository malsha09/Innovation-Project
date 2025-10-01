import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ---------- CONFIG ----------
NPZ_PATH = 'all_datasets_compressed.npz'
TARGET_HORIZON = 1  # choose 1 or 24 depending on arrays
EPOCHS = 3           # small for testing
BATCH = 64

OUTDIR = 'phase3_outputs'
os.makedirs(OUTDIR, exist_ok=True)

# ---------- HELPERS ----------
def build_tcn(n_features, lookback, hidden=64, dropout=0.2, horizon=1):
    inp = layers.Input(shape=(lookback, n_features))
    x = inp
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
    return keras.Model(inp, out)

def build_transformer(n_features, lookback, dim=64, heads=4, ff_dim=128, dropout=0.1, horizon=1):
    inp = layers.Input(shape=(lookback, n_features))
    x = layers.Dense(dim)(inp)
    pos_emb = layers.Embedding(input_dim=lookback, output_dim=dim)(keras.backend.arange(lookback))
    pos_emb = keras.backend.expand_dims(pos_emb, axis=0)
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

# ---------- LOAD DATA ----------
print("Loading NPZ...")
with np.load(NPZ_PATH) as data:
    Xtr = data[f'X_train_{TARGET_HORIZON}']
    ytr = data[f'y_train_{TARGET_HORIZON}']
    Xv  = data[f'X_val_{TARGET_HORIZON}']
    yv  = data[f'y_val_{TARGET_HORIZON}']
    Xt  = data[f'X_test_{TARGET_HORIZON}']
    yt  = data[f'y_test_{TARGET_HORIZON}']

LOOKBACK = Xtr.shape[1]
N_FEATURES = Xtr.shape[2]
print("Shapes:", Xtr.shape, ytr.shape, Xv.shape, yv.shape, Xt.shape, yt.shape)

# ---------- TRAIN & EVALUATE ----------
def train_and_eval(model, name):
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    print(f"\nTraining {name}...")
    model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=EPOCHS, batch_size=BATCH, verbose=2)
    yv_pred = model.predict(Xv)
    yt_pred = model.predict(Xt)
    mae_val = np.mean(np.abs(yv - yv_pred))
    mae_test = np.mean(np.abs(yt - yt_pred))
    print(f"{name} Validation MAE: {mae_val:.4f}, Test MAE: {mae_test:.4f}")
    model.save(os.path.join(OUTDIR, f'{name}_model.keras'))
    return yv_pred, yt_pred

# Build models
tcn = build_tcn(N_FEATURES, LOOKBACK)
transformer = build_transformer(N_FEATURES, LOOKBACK)

# Train & evaluate
yv_pred_tcn, yt_pred_tcn = train_and_eval(tcn, 'TCN')
yv_pred_trf, yt_pred_trf = train_and_eval(transformer, 'Transformer')

# ---------- PLOT SAMPLE ----------
plt.figure(figsize=(12,4))
plt.plot(yt[:200], label='true')
plt.plot(yt_pred_tcn[:200], label='TCN pred')
plt.plot(yt_pred_trf[:200], label='Transformer pred')
plt.title('Sample Test Prediction')
plt.legend()
plt.show()
