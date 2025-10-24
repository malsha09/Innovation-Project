from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

app = Flask(__name__)

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../outputs/TCN_h1.keras")
DATA_PATH = os.path.join(BASE_DIR, "../data/cleaned_electricity_demand.csv")

# -----------------------------
# LOAD MODEL AND DATA
# -----------------------------
model = load_model(MODEL_PATH)
data = pd.read_csv(DATA_PATH)
data["datetime"] = pd.to_datetime(data["datetime"])

# -----------------------------
# HELPER METRICS
# -----------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100


# -----------------------------
# FORECAST GENERATION LOGIC
# -----------------------------
def generate_forecast(selected_date):
    try:
        selected_date = pd.to_datetime(selected_date)

        # Ensure valid date range
        last_known_date = data["datetime"].max()
        if selected_date <= last_known_date:
            selected_date = last_known_date + pd.Timedelta(days=1)

        # Use last 168 hours for prediction context
        recent_data = data.tail(168)
        numeric_data = recent_data.select_dtypes(include=[np.number]).astype(np.float32)

        expected_features = model.input_shape[2]
        actual_features = numeric_data.shape[1]

        # Match model input shape
        if actual_features > expected_features:
            numeric_data = numeric_data.iloc[:, :expected_features]
        elif actual_features < expected_features:
            pad = np.zeros((numeric_data.shape[0], expected_features - actual_features), dtype=np.float32)
            numeric_data = np.hstack([numeric_data.values, pad])
            numeric_data = pd.DataFrame(numeric_data)

        # Initial model input (window of 168 hours)
        X_input = np.expand_dims(numeric_data.values, axis=0)

        # Recursive multi-step forecasting (e.g., 24h)
        forecast_values = []
        forecast_horizon = 24  # hours

        for i in range(forecast_horizon):
            pred = model.predict(X_input, verbose=0)[0, 0]
            forecast_values.append(pred)

            # Slide window forward (replace oldest row with latest prediction)
            new_row = X_input[:, -1:, :].copy()
            new_row[0, 0, 0] = pred  # assumes target variable is first column
            X_input = np.concatenate([X_input[:, 1:, :], new_row], axis=1)

        # Build forecast DataFrame
        future_dates = pd.date_range(start=selected_date, periods=forecast_horizon, freq="H")
        forecast_df = pd.DataFrame({"datetime": future_dates, "forecast_MW": forecast_values})

        # Optional: calculate metrics if true data exists
        actual = data[data["datetime"].isin(future_dates)]["nat_demand"]
        if len(actual) == len(forecast_values):
            mae = mean_absolute_error(actual, forecast_values)
            rmse = np.sqrt(mean_squared_error(actual, forecast_values))
            mape_val = mape(actual, forecast_values)
        else:
            mae = rmse = mape_val = None

        return forecast_df, mae, rmse, mape_val

    except Exception as e:
        print("Error generating forecast:", e)
        return None, None, None, None


# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    forecast_data = None
    mae = rmse = mape_val = None

    if request.method == "POST":
        selected_date = request.form.get("date")
        if selected_date:
            forecast_data, mae, rmse, mape_val = generate_forecast(selected_date)

    # Prevent template errors
    if forecast_data is None:
        forecast_data = pd.DataFrame(columns=["datetime", "forecast_MW"])

    return render_template(
        "index.html",
        forecast=forecast_data,
        mae=mae,
        rmse=rmse,
        mape=mape_val
    )


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
