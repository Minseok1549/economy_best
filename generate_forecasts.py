import datetime as dt
import json
import os
import warnings

import FinanceDataReader as fdr
import joblib
import numpy as np
import pandas as pd
from prophet.serialize import model_from_json

warnings.filterwarnings("ignore")

# --- 모델 저장 경로 ---
MODEL_SAVE_PATH = "models"
FORECAST_DATA_PATH = "data/forecasts"
if not os.path.exists(FORECAST_DATA_PATH):
    os.makedirs(FORECAST_DATA_PATH)


# --- 메인 예측 함수 ---
def generate_forecasts_from_saved_models(ticker: str):
    """저장된 모델들을 불러와 다음 날 주가를 예측합니다."""
    print(f"[{ticker}] 저장된 모델을 불러와 예측을 생성합니다...")

    ticker_path = os.path.join(MODEL_SAVE_PATH, ticker)
    if not os.path.exists(ticker_path):
        print(
            f" ! 경고: '{ticker}'에 대한 학습된 모델이 없습니다. 'train_models.py'를 먼저 실행해주세요."
        )
        return

    # 1. 예측에 필요한 최신 데이터 로드 (짧은 기간)
    df = fdr.DataReader(ticker, start=dt.date.today() - dt.timedelta(days=30))
    if df.empty:
        print(f"[{ticker}] 최신 데이터 로드 실패.")
        return

    predictions = {}

    # 2. 모델별 예측 수행

    # 2-1. 지수 평활 모델
    try:
        model = joblib.load(os.path.join(ticker_path, "exp_smoothing.joblib"))
        predictions["ExpSmooth"] = model.forecast(1).values[0]
    except Exception as e:
        print(f"  - 지수 평활 모델 예측 오류: {e}")

    # 2-2. Prophet 모델
    try:
        with open(os.path.join(ticker_path, "prophet_model.json"), "r") as fin:
            model = model_from_json(json.load(fin))
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        predictions["Prophet"] = forecast["yhat"].iloc[-1]
    except Exception as e:
        print(f"  - Prophet 모델 예측 오류: {e}")

    # --- 특성(Feature) 기반 모델들을 위한 데이터 준비 ---
    last_row = df.iloc[-1]
    time_step = (
        len(df)
        - 1
        + (df.index[-1].date() - dt.date.today() + dt.timedelta(days=730)).days
    )  # Approximation
    ma5 = df["Close"].rolling(window=5).mean().iloc[-1]
    next_features = np.array([[time_step + 1, ma5]])

    # 2-3. 선형 회귀 모델
    try:
        model = joblib.load(os.path.join(ticker_path, "linear_regression.joblib"))
        predictions["LinearReg"] = model.predict(next_features)[0]
    except Exception as e:
        print(f"  - 선형 회귀 모델 예측 오류: {e}")

    # 2-4. XGBoost 모델
    try:
        model = joblib.load(os.path.join(ticker_path, "xgboost.joblib"))
        predictions["XGBoost"] = model.predict(next_features)[0]
    except Exception as e:
        print(f"  - XGBoost 모델 예측 오류: {e}")

    # 2-5. MLP 모델 (스케일러도 함께 로드)
    try:
        model = joblib.load(os.path.join(ticker_path, "mlp.joblib"))
        scaler_X = joblib.load(os.path.join(ticker_path, "mlp_scaler_X.joblib"))
        scaler_y = joblib.load(os.path.join(ticker_path, "mlp_scaler_y.joblib"))

        next_features_scaled = scaler_X.transform(next_features)
        prediction_scaled = model.predict(next_features_scaled)
        predictions["MLP"] = scaler_y.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        )[0][0]
    except Exception as e:
        print(f"  - MLP 모델 예측 오류: {e}")

    # 3. 예측 결과 저장
    if not predictions:
        print(f"[{ticker}] 예측을 생성하지 못했습니다.")
        return

    next_day = df.index[-1] + dt.timedelta(days=1)
    forecast_df = pd.DataFrame(predictions, index=[next_day])
    output_path = os.path.join(FORECAST_DATA_PATH, f"{ticker}_forecast.csv")
    forecast_df.to_csv(output_path)
    print(f" > [{ticker}] 예측 완료. '{output_path}'에 저장되었습니다.\n")


if __name__ == "__main__":
    TICKERS_TO_FORECAST = ["005930", "035720", "000660", "TSLA", "AAPL"]
    for ticker in TICKERS_TO_FORECAST:
        generate_forecasts_from_saved_models(ticker)
