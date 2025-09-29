from __future__ import annotations

import datetime as dt

import FinanceDataReader as fdr

# 예측 모델을 위한 라이브러리
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA


# 1. Moving average
def calculate_ma(close_prices: pd.Series, window: int) -> pd.Series:
    return close_prices.rolling(window=window).mean()


def interpret_ma_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    signals = pd.Series(index=short_ma.index, data="Hold")
    signals[(short_ma.shift(1) < long_ma.shift(1)) & (short_ma > long_ma)] = "Buy"
    signals[(short_ma.shift(1) > long_ma.shift(1)) & (short_ma < long_ma)] = "Sell"
    return signals


# 2. 상대강도지수 (Relative Strength Index, RSI)
def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    delta = pd.to_numeric(close_prices.diff(1), errors="coerce")
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def interpret_rsi(
    rsi: pd.Series, overbought_threshold: int = 70, oversold_threshold: int = 30
) -> pd.Series:
    signals = pd.Series(index=rsi.index, data="Hold")
    signals[rsi > overbought_threshold] = "Sell"
    signals[rsi < oversold_threshold] = "Buy"
    return signals


# 3. MACD (Moving Average Convergence Divergence)
def calculate_macd(
    close_prices: pd.Series,
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    short_ema = close_prices.ewm(span=short_window, adjust=False).mean()
    long_ema = close_prices.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram}
    )


def interpret_macd(macd_df: pd.DataFrame) -> pd.Series:
    signals = pd.Series(index=macd_df.index, data="Hold")
    signals[
        (macd_df["MACD"].shift(1) < macd_df["Signal"].shift(1))
        & (macd_df["MACD"] > macd_df["Signal"])
    ] = "Buy"
    signals[
        (macd_df["MACD"].shift(1) > macd_df["Signal"].shift(1))
        & (macd_df["MACD"] < macd_df["Signal"])
    ] = "Sell"
    return signals


# ---  예측 모델 함수 (수정 없음) ---
# ... (이전 코드와 동일한 예측 모델 함수들) ...
# 1. ARIMA 모델
@st.cache_data
def predict_arima(close_prices: pd.Series, n_periods: int) -> pd.Series:
    model = auto_arima(
        close_prices,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        m=1,
        d=1,
        seasonal=False,
        trace=False,
        suppress_warnings=True,
    )
    model.fit(close_prices)
    return model.predict(n_periods=n_periods)


# 2. LightGBM 모델
@st.cache_data
def predict_lightgbm(df: pd.DataFrame, n_periods: int) -> pd.Series:
    df_lgbm = df.copy()
    df_lgbm["dayofweek"] = df_lgbm.index.dayofweek
    df_lgbm["month"] = df_lgbm.index.month
    df_lgbm["year"] = df_lgbm.index.year
    df_lgbm["lag_1"] = df_lgbm["Close"].shift(1)
    df_lgbm = df_lgbm.dropna()
    features = ["dayofweek", "month", "year", "lag_1"]
    target = "Close"
    X_train, y_train = df_lgbm[features], df_lgbm[target]
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1), periods=n_periods
    )
    predictions = []
    last_known_price = df_lgbm["Close"].iloc[-1]
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


# 3. LSTM (PyTorch) 모델
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


@st.cache_data
def predict_lstm(
    close_prices: pd.Series, n_periods: int, epochs: int = 100, look_back: int = 15
) -> pd.Series:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price_scaled = scaler.fit_transform(close_prices.values.reshape(-1, 1))
    price_scaled = torch.FloatTensor(price_scaled).view(-1)

    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i : i + tw]
            train_label = input_data[i + tw : i + tw + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    train_inout_seq = create_inout_sequences(price_scaled, look_back)
    model = LSTMModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(epochs):
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


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="기술적 지표 및 주가 예측 툴")
    st.title("📈 주식 기술적 지표 분석 및 주가 예측 대시보드")
    st.markdown(
        "관심 종목의 주가와 기술적 지표를 시각화하고, 다양한 모델을 통해 미래 주가를 예측합니다."
    )

    st.sidebar.header("⚙️ 설정")
    ticker = st.sidebar.text_input("종목 코드 (Ticker)", "005930")
    today = dt.date.today()
    start_date = st.sidebar.date_input("시작일", today - dt.timedelta(days=730))
    end_date = st.sidebar.date_input("종료일", today)
    st.sidebar.subheader("지표 파라미터")
    ma_short_window = st.sidebar.number_input("단기 이동평균 기간", 5, 50, 20, 1)
    ma_long_window = st.sidebar.number_input("장기 이동평균 기간", 20, 200, 60, 5)
    rsi_window = st.sidebar.number_input("RSI 기간", 5, 30, 14, 1)
    st.sidebar.subheader("📈 예측 설정")
    forecast_days = st.sidebar.number_input("예측 기간 (일)", 1, 90, 30)

    if st.sidebar.button("📊 분석 및 예측 시작"):
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            if df.empty:
                st.error(
                    "해당 기간에 대한 데이터가 없습니다. 종목 코드나 기간을 확인해주세요."
                )
            else:
                # --- 지표 계산 및 신호 해석 (기존과 동일) ---
                df["MA_Short"] = calculate_ma(df["Close"], ma_short_window)
                df["MA_Long"] = calculate_ma(df["Close"], ma_long_window)
                df["RSI"] = calculate_rsi(df["Close"], rsi_window)
                macd_df = calculate_macd(df["Close"])
                df = df.join(macd_df)
                macd_signals = interpret_macd(df)

                st.header(f"'{ticker}' 기술적 분석 결과")
                latest_data = df.iloc[-1]
                price_change = latest_data["Close"] - df.iloc[-2]["Close"]
                change_percent = (price_change / df.iloc[-2]["Close"]) * 100
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "최신 종가",
                    f"{latest_data['Close']:,.0f} 원",
                    f"{price_change:,.0f} 원 ({change_percent:.2f}%)",
                )
                col2.metric("최신 RSI", f"{latest_data['RSI']:.2f}")
                col3.metric("최신 MACD 신호", macd_signals.iloc[-1])

                # --- 예측 모델 실행 ---
                with st.spinner("ARIMA 모델로 예측 중..."):
                    arima_preds = predict_arima(df["Close"], forecast_days)
                with st.spinner("LightGBM 모델로 예측 중..."):
                    lgbm_preds = predict_lightgbm(df, forecast_days)
                with st.spinner("LSTM (Pytorch) 모델로 예측 중..."):
                    lstm_preds = predict_lstm(df["Close"], forecast_days)

                # --- 예측 결과 통합 및 범위 계산 ---
                forecast_df = pd.DataFrame(
                    {"ARIMA": arima_preds, "LightGBM": lgbm_preds, "LSTM": lstm_preds}
                )
                min_preds = forecast_df.min(axis=1)
                max_preds = forecast_df.max(axis=1)
                mean_preds = forecast_df.mean(axis=1)

                # --- 시각화 ---
                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.5, 0.1, 0.2, 0.2],
                )

                # 1. 캔들차트, 이평선, 매매신호, 예측 결과 (밴드)
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="캔들차트",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["MA_Short"],
                        mode="lines",
                        name=f"MA {ma_short_window}",
                        line=dict(color="orange"),
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["MA_Long"],
                        mode="lines",
                        name=f"MA {ma_long_window}",
                        line=dict(color="purple"),
                    ),
                    row=1,
                    col=1,
                )

                # ✨ 예측 범위를 밴드로 추가 ✨
                # 상한선과 하한선을 투명한 선으로 먼저 그리고, 그 사이를 채웁니다.
                fig.add_trace(
                    go.Scatter(
                        x=max_preds.index,
                        y=max_preds,
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=min_preds.index,
                        y=min_preds,
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        fillcolor="rgba(255, 165, 0, 0.2)",
                        fill="tonexty",
                        name="예측 범위",
                    ),
                    row=1,
                    col=1,
                )
                # 예측 평균선을 점선으로 추가
                fig.add_trace(
                    go.Scatter(
                        x=mean_preds.index,
                        y=mean_preds,
                        mode="lines",
                        name="예측 평균",
                        line=dict(color="orange", dash="dot"),
                    ),
                    row=1,
                    col=1,
                )

                # 나머지 차트 (거래량, RSI, MACD)는 기존과 동일
                colors = [
                    "green" if row["Open"] - row["Close"] >= 0 else "red"
                    for index, row in df.iterrows()
                ]
                fig.add_trace(
                    go.Bar(
                        x=df.index, y=df["Volume"], name="거래량", marker_color=colors
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["RSI"],
                        mode="lines",
                        name="RSI",
                        line=dict(color="blue"),
                    ),
                    row=3,
                    col=1,
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["MACD"],
                        mode="lines",
                        name="MACD",
                        line=dict(color="brown"),
                    ),
                    row=4,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["Signal"],
                        mode="lines",
                        name="Signal Line",
                        line=dict(color="goldenrod"),
                    ),
                    row=4,
                    col=1,
                )
                colors_macd = [
                    "green" if val >= 0 else "red" for val in df["Histogram"]
                ]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df["Histogram"],
                        name="Histogram",
                        marker_color=colors_macd,
                    ),
                    row=4,
                    col=1,
                )

                # 레이아웃 업데이트
                fig.update_layout(
                    title_text=f"{ticker} 종합 기술적 분석 및 주가 예측",
                    height=900,
                    xaxis_rangeslider_visible=False,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
                fig.update_yaxes(title_text="가격", row=1, col=1)
                fig.update_yaxes(title_text="거래량", row=2, col=1)
                fig.update_yaxes(title_text="RSI", row=3, col=1)
                fig.update_yaxes(title_text="MACD", row=4, col=1)

                st.plotly_chart(fig, use_container_width=True)

                st.header("🔮 모델별 예측 결과")
                st.dataframe(forecast_df.style.format("{:,.0f}"))
                with st.expander("상세 데이터 보기"):
                    st.dataframe(df.iloc[::-1].style.format("{:,.2f}"))

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
