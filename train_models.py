from __future__ import annotations

import datetime as dt
import json
import os
import warnings

import FinanceDataReader as fdr
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import ExponentialSmoothing

warnings.simplefilter(action="ignore", category=FutureWarning)

MODEL_SAVE_PATH = "models"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)


def train_and_save_models(ticker: str):
    """지정된 티커의 데이터를 불러와 모든 모델을 학습하고 파일로 저장합니다."""
    print(f"[{ticker}] 모델 학습 및 저장을 시작합니다...")

    # 1. 데이터 로드 (3년치 데이터로 변경)
    df = fdr.DataReader(ticker, start=dt.date.today() - dt.timedelta(days=1095))
    if df.empty:
        print(f"[{ticker}] 데이터가 없습니다. 모델 학습을 건너뜁니다.")
        return

    # 2. 저장 경로 확인
    ticker_path = os.path.join(MODEL_SAVE_PATH, ticker)
    if not os.path.exists(ticker_path):
        os.makedirs(ticker_path)

    # 3. 모델 학습 및 저장
    try:
        # FIX 1: 'Cloase' -> 'Close' 오타 수정
        model = ExponentialSmoothing(
            df["Close"],
            trend="add",
            seasonal="add",
            seasonal_periods=5,
        ).fit()
        joblib.dump(model, os.path.join(ticker_path, "exp_smoothing.joblib"))
        print(f"  - 지수 평활 모델 저장 완료.")
    except Exception as e:
        print(f"[{ticker}] Exponential Smoothing 모델 학습 실패: {e}")

    try:
        # FIX 2: 국내/해외 주식 모두 처리하도록 Prophet 데이터 준비 방식 변경
        prophet_df = df.reset_index()
        # 첫 번째 열(날짜 열)의 이름을 'ds'로, 'Close' 열을 'y'로 변경
        prophet_df = prophet_df.rename(
            columns={prophet_df.columns[0]: "ds", "Close": "y"}
        )

        model = Prophet(daily_seasonality=True).fit(prophet_df)
        with open(os.path.join(ticker_path, "prophet_model.json"), "w") as fout:
            json.dump(model_to_json(model), fout)
        print(f"  - Prophet 모델 저장 완료.")
    except Exception as e:
        print(f"[{ticker}] Prophet 모델 학습 실패: {e}")

    df_feat = df.copy()
    df_feat["time"] = np.arange(len(df.index))
    # FIX 3: fillna(method="bfill") -> .bfill() 로 변경
    df_feat["MA5"] = df_feat["Close"].rolling(window=5).mean().bfill()
    X = df_feat[["time", "MA5"]]
    y = df_feat["Close"]

    try:
        model = LinearRegression().fit(X, y)
        joblib.dump(model, os.path.join(ticker_path, "linear_regression.joblib"))
        print(f"  - 선형 회귀 모델 저장 완료.")
    except Exception as e:
        print(f"  - 선형 회귀 모델 학습 오류: {e}")

    try:
        model = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=100, random_state=42
        ).fit(X, y)
        joblib.dump(model, os.path.join(ticker_path, "xgboost.joblib"))
        print(f"  - XGBoost 모델 저장 완료.")
    except Exception as e:
        print(f"  - XGBoost 모델 학습 오류: {e}")

    try:
        scaler_X = MinMaxScaler().fit(X)
        scaler_y = MinMaxScaler().fit(y.values.reshape(-1, 1))
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.values.reshape(-1, 1))

        model = MLPRegressor(
            hidden_layer_sizes=(50, 25), max_iter=500, random_state=42
        ).fit(X_scaled, y_scaled.ravel())

        joblib.dump(model, os.path.join(ticker_path, "mlp.joblib"))
        joblib.dump(scaler_X, os.path.join(ticker_path, "mlp_scaler_X.joblib"))
        joblib.dump(scaler_y, os.path.join(ticker_path, "mlp_scaler_y.joblib"))
        print(f"  - MLP 모델 및 스케일러 저장 완료.")
    except Exception as e:
        print(f"  - MLP 모델 학습 오류: {e}")

    print(f"[{ticker}] 모든 모델 학습 및 저장이 완료되었습니다.\n")


if __name__ == "__main__":
    TICKERS_TO_TRAIN = ["005930", "035720", "000660", "TSLA", "AAPL"]
    for ticker in TICKERS_TO_TRAIN:
        train_and_save_models(ticker)
