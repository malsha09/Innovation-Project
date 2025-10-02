import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import shap

# ---------- CONFIG ----------
NPZ_PATH = 'all_datasets_compressed.npz'
TARGET_HORIZON = 1  # choose 1 or 24 depending on arrays
EPOCHS = 1          # small for testing
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

# ---------- BUILD & TRAIN MODELS ----------
tcn = build_tcn(N_FEATURES, LOOKBACK)
transformer = build_transformer(N_FEATURES, LOOKBACK)

yv_pred_tcn, yt_pred_tcn = train_and_eval(tcn, 'TCN')
yv_pred_trf, yt_pred_trf = train_and_eval(transformer, 'Transformer')

import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# -------------------------
# Config
# -------------------------
LOOKBACK = Xtr.shape[1]        # timesteps
N_FEATURES = Xtr.shape[2]      # features
SAMPLE_SIZE = 2               # number of test samples to explain
BACKGROUND_SIZE = 50            # number of background samples

# -------------------------
# Prepare background and sample
# -------------------------
X_sample = Xt[:SAMPLE_SIZE]    # (samples, timesteps, features)
X_sample_2d = X_sample.reshape(SAMPLE_SIZE, -1)

background_idx = np.random.choice(Xtr.shape[0], BACKGROUND_SIZE, replace=False)
background_2d = Xtr[background_idx].reshape(BACKGROUND_SIZE, -1)

# -------------------------
# Define prediction function for KernelExplainer
# -------------------------
def model_predict(x_flat):
    """
    x_flat: (samples, timesteps*features)
    Returns: (samples, output_dim)
    """
    x_reshaped = x_flat.reshape(x_flat.shape[0], LOOKBACK, N_FEATURES)
    y_pred = tcn.predict(x_reshaped)
    
    # If multi-output, KernelExplainer returns list per output
    if isinstance(y_pred, list):
        return [yp for yp in y_pred]
    else:
        return y_pred

# -------------------------
# Initialize KernelExplainer
# -------------------------
explainer = shap.KernelExplainer(model_predict, background_2d)
shap_values = explainer.shap_values(X_sample_2d)

# Handle single vs multi-output
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# -------------------------
# Reshape SHAP values back to (samples, timesteps, features)
# -------------------------
shap_values_3d = np.array(shap_values).reshape(SAMPLE_SIZE, LOOKBACK, N_FEATURES)

# -------------------------
# Aggregate over timesteps
# -------------------------
mean_shap_per_sample = np.mean(np.abs(shap_values_3d), axis=1)  # (samples, features)
feature_importance = np.mean(mean_shap_per_sample, axis=0)       # (features,)

# -------------------------
# Bar plot: feature importance
# -------------------------
plt.figure(figsize=(10,5))
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel('Feature Index')
plt.ylabel('Mean |SHAP value|')
plt.title('TCN Feature Importance (aggregated over timesteps and samples)')
plt.show()

# -------------------------
# Heatmap: SHAP values per sample × feature
# -------------------------
plt.figure(figsize=(12,6))
sns.heatmap(mean_shap_per_sample, cmap='coolwarm', annot=False)
plt.xlabel('Feature Index')
plt.ylabel('Sample Index')
plt.title('TCN SHAP Heatmap (samples × features, averaged over timesteps)')
plt.show()








# ---------- PLOT SAMPLE ----------
plt.figure(figsize=(12,4))
plt.plot(yt[:200], label='true')
plt.plot(yt_pred_tcn[:200], label='TCN pred')
plt.plot(yt_pred_trf[:200], label='Transformer pred')
plt.title('Sample Test Prediction')
plt.legend()
plt.show()
