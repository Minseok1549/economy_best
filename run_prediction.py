from __future__ import annotations

import datetime as dt

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- FIX 1: lightgbm 대신 xgboost를 임포트합니다. ---
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm


def predict_ets(close_prices: pd.Series, n_periods: int) -> pd.Series:
    """Exponential Smoothing 모델을 사용하여 미래 가격을 예측합니다."""
    model = ExponentialSmoothing(close_prices, trend="add", seasonal=None)
    model_fit = model.fit()
    return model_fit.forecast(steps=n_periods)


# --- FIX 2: predict_lightgbm 함수를 predict_xgboost 함수로 변경합니다. ---
def predict_xgboost(df: pd.DataFrame, n_periods: int) -> pd.Series:
    """XGBoost 모델을 사용하여 미래 가격을 예측합니다."""
    df_xgb = df.copy()
    df_xgb["dayofweek"] = df_xgb.index.dayofweek
    df_xgb["month"] = df_xgb.index.month
    df_xgb["year"] = df_xgb.index.year
    df_xgb["lag_1"] = df_xgb["Close"].shift(1)
    df_xgb = df_xgb.dropna()
    features = ["dayofweek", "month", "year", "lag_1"]
    target = "Close"
    X_train, y_train = df_xgb[features], df_xgb[target]

    # XGBRegressor 모델을 사용하고, 과적합 방지를 위해 max_depth를 설정합니다.
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, seed=42, random_state=42)
    model.fit(X_train, y_train)

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1), periods=n_periods
    )
    predictions = []
    last_known_price = df_xgb["Close"].iloc[-1]
    for date in future_dates:
        features_to_predict = pd.DataFrame(
            [
                {
                    "dayofweek": date.dayofweek,
                    "month": date.month,
                    "year": date.year,
                    "lag_1": last_known_price,
                }
            ]
        )
        prediction = model.predict(features_to_predict)[0]
        predictions.append(prediction)
        last_known_price = prediction
    return pd.Series(predictions, index=future_dates)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def predict_lstm(
    close_prices: pd.Series, n_periods: int, epochs: int = 100, look_back: int = 15
) -> pd.Series:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price_scaled = scaler.fit_transform(close_prices.values.reshape(-1, 1))
    price_scaled = torch.FloatTensor(price_scaled).view(-1)

    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        iterator = tqdm(
            range(L - tw),
            desc=f"Preparing LSTM data for {close_prices.name}",
            leave=False,
        )
        for i in iterator:
            train_seq = input_data[i : i + tw]
            train_label = input_data[i + tw : i + tw + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    train_inout_seq = create_inout_sequences(price_scaled, look_back)
    model = LSTMModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    iterator = tqdm(
        range(epochs), desc=f"LSTM Training for {close_prices.name}", leave=False
    )
    for i in iterator:
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    test_inputs = price_scaled[-look_back:].tolist()
    for _ in range(n_periods):
        seq = torch.FloatTensor(test_inputs[-look_back:])
        with torch.no_grad():
            model.hidden = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )
            test_inputs.append(model(seq).item())
    actual_predictions = scaler.inverse_transform(
        np.array(test_inputs[look_back:]).reshape(-1, 1)
    )
    future_dates = pd.date_range(
        start=close_prices.index[-1] + pd.Timedelta(days=1), periods=n_periods
    )
    return pd.Series(actual_predictions.flatten(), index=future_dates)


def run_and_save_predictions(ticker: str, forecast_days: int = 1):
    """특정 종목의 데이터를 불러와 예측하고 CSV 파일로 저장합니다."""
    today = dt.date.today()
    start_date = today - dt.timedelta(days=730)

    print(f"[{ticker}] 데이터 로드 중...")
    df = fdr.DataReader(ticker, start_date, today)
    if df.empty:
        print(f"[{ticker}] 데이터 로드 실패.")
        return

    df = df.asfreq("B").ffill()
    df["Close"].name = ticker

    print(f"[{ticker}] 예측 실행 중...")
    ets_preds = predict_ets(df["Close"], forecast_days)
    # --- FIX 3: predict_lightgbm 호출을 predict_xgboost로 변경합니다. ---
    xgb_preds = predict_xgboost(df, forecast_days)
    lstm_preds = predict_lstm(df["Close"], forecast_days)

    df.to_csv(f"{ticker}_history.csv")

    # --- FIX 4: 결과 저장 시 컬럼명을 'LightGBM'에서 'XGBoost'로 변경합니다. ---
    forecast_df = pd.DataFrame(
        {"ETS": ets_preds, "XGBoost": xgb_preds, "LSTM": lstm_preds}
    )
    forecast_df.to_csv(f"{ticker}_forecast.csv")

    print(f"[{ticker}] 예측 결과 저장 완료.")


if __name__ == "__main__":
    tickers_to_predict = ["005930", "000660", "035720"]

    for ticker in tqdm(tickers_to_predict, desc="Overall Progress"):
        run_and_save_predictions(ticker)
