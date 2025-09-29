from __future__ import annotations

import datetime as dt

import FinanceDataReader as fdr  # 실시간 데이터 로드를 위해 추가
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# --- 기술적 지표 계산 함수 (기존과 동일) ---
def calculate_ma(close_prices: pd.Series, window: int) -> pd.Series:
    return close_prices.rolling(window=window).mean()


def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    delta = pd.to_numeric(close_prices.diff(1), errors="coerce")
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


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


# --- FIX: 데이터 로드 함수 수정 ---
# 실시간 데이터와 예측 데이터를 분리해서 로드합니다.
@st.cache_data(ttl=600)  # 10분 동안 캐시 유지
def load_live_data(ticker):
    """FinanceDataReader를 사용해 최신 주가 데이터를 불러옵니다."""
    today = dt.date.today()
    start_date = today - dt.timedelta(days=730)  # 최근 2년치 데이터
    try:
        df = fdr.DataReader(ticker, start_date, today)
        return df
    except Exception:
        return None


def load_forecast_data(ticker):
    """미리 생성된 예측 CSV 파일을 로드합니다."""
    try:
        forecast_df = pd.read_csv(
            f"{ticker}_forecast.csv", index_col=0, parse_dates=True
        )
        return forecast_df
    except FileNotFoundError:
        return None


# --- FIX: 상세한 기술적 분석 신호 해석 함수 ---
def get_technical_signals(df: pd.DataFrame):
    """데이터프레임을 받아 각 지표별 매매 신호와 설명을 반환합니다."""
    signals = {}

    # 1. 이동평균 (MA) 신호
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    ma_short = f"MA_{st.session_state.ma_short_window}"
    ma_long = f"MA_{st.session_state.ma_long_window}"
    df[ma_short] = calculate_ma(df["Close"], st.session_state.ma_short_window)
    df[ma_long] = calculate_ma(df["Close"], st.session_state.ma_long_window)

    if (
        df[ma_short].iloc[-1] > df[ma_long].iloc[-1]
        and df[ma_short].iloc[-2] <= df[ma_long].iloc[-2]
    ):
        signals["ma"] = (
            "🟢 매수",
            "단기 이동평균선이 장기선을 상향 돌파 (골든 크로스)",
        )
    elif (
        df[ma_short].iloc[-1] < df[ma_long].iloc[-1]
        and df[ma_short].iloc[-2] >= df[ma_long].iloc[-2]
    ):
        signals["ma"] = (
            "🔴 매도",
            "단기 이동평균선이 장기선을 하향 돌파 (데드 크로스)",
        )
    else:
        trend = "상승" if df[ma_short].iloc[-1] > df[ma_long].iloc[-1] else "하락"
        signals["ma"] = ("⚪️ 중립", f"현재 {trend} 추세 유지 중")

    # 2. RSI 신호
    rsi_val = df["RSI"].iloc[-1]
    if rsi_val > 70:
        signals["rsi"] = ("🔴 매도", f"RSI({rsi_val:.1f})가 70 이상으로 과매수 상태")
    elif rsi_val < 30:
        signals["rsi"] = ("🟢 매수", f"RSI({rsi_val:.1f})가 30 이하로 과매도 상태")
    else:
        signals["rsi"] = ("⚪️ 중립", f"RSI({rsi_val:.1f})가 중립 구간에 위치")

    # 3. MACD 신호
    if (
        df["MACD"].iloc[-1] > df["Signal"].iloc[-1]
        and df["MACD"].iloc[-2] <= df["Signal"].iloc[-2]
    ):
        signals["macd"] = ("🟢 매수", "MACD선이 시그널선을 상향 돌파 (골든 크로스)")
    elif (
        df["MACD"].iloc[-1] < df["Signal"].iloc[-1]
        and df["MACD"].iloc[-2] >= df["Signal"].iloc[-2]
    ):
        signals["macd"] = ("🔴 매도", "MACD선이 시그널선을 하향 돌파 (데드 크로스)")
    else:
        trend = "상승" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "하락"
        signals["macd"] = ("⚪️ 중립", f"현재 {trend} 추세 유지 중")

    return signals


# --- 메인 대시보드 ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="주가 예측 대시보드")
    st.title("📈 실시간 기술적 분석 및 내일 주가 예측")

    # --- 사이드바 설정 ---
    st.sidebar.header("⚙️ 설정")
    ticker = st.sidebar.selectbox("종목 코드 (Ticker)", ["005930", "000660", "035720"])

    # 세션 상태(session_state)를 사용하여 위젯 값을 저장해야 함수 내에서 접근 가능
    st.session_state.ma_short_window = st.sidebar.number_input(
        "단기 이동평균", 5, 50, 20, 1
    )
    st.session_state.ma_long_window = st.sidebar.number_input(
        "장기 이동평균", 20, 200, 60, 5
    )
    st.session_state.rsi_window = st.sidebar.number_input("RSI 기간", 5, 30, 14, 1)

    # --- 데이터 로드 ---
    df = load_live_data(ticker)
    forecast_df = load_forecast_data(ticker)

    if df is None:
        st.error(
            f"'{ticker}'에 대한 실시간 데이터를 불러오는 데 실패했습니다. 종목 코드를 확인해주세요."
        )
    else:
        # 기술적 지표 계산
        df["RSI"] = calculate_rsi(df["Close"], st.session_state.rsi_window)
        macd_df = calculate_macd(df["Close"])
        df = df.join(macd_df)

        # 신호 해석
        signals = get_technical_signals(df)

        # --- 종합 신호 요약 ---
        st.header(f"'{ticker}' 종합 기술적 분석 신호")
        cols = st.columns(3)
        with cols[0]:
            st.subheader("이동평균 (MA)")
            signal, reason = signals["ma"]
            st.markdown(f"#### 신호: **{signal}**")
            st.caption(reason)
        with cols[1]:
            st.subheader("RSI")
            signal, reason = signals["rsi"]
            st.markdown(f"#### 신호: **{signal}**")
            st.caption(reason)
        with cols[2]:
            st.subheader("MACD")
            signal, reason = signals["macd"]
            st.markdown(f"#### 신호: **{signal}**")
            st.caption(reason)

        st.divider()

        # --- 메인 가격 차트 ---
        st.header("가격 차트 및 다음 날 예측")
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="캔들차트",
            )
        )
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"MA_{st.session_state.ma_short_window}"],
                mode="lines",
                name=f"{st.session_state.ma_short_window}일 이동평균",
                line=dict(color="green"),
            )
        )
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"MA_{st.session_state.ma_long_window}"],
                mode="lines",
                name=f"{st.session_state.ma_long_window}일 이동평균",
                line=dict(color="red"),
            )
        )

        # 예측 데이터가 있을 때만 예측 범위를 표시
        if forecast_df is not None:
            next_day = forecast_df.index[0]
            min_pred = forecast_df.min(axis=1).iloc[0]
            max_pred = forecast_df.max(axis=1).iloc[0]
            mean_pred = forecast_df.mean(axis=1).iloc[0]

            # --- FIX: 마우스를 올리면 정보가 보이는 Candlestick 객체로 교체 ---
            # 기존의 add_shape, add_annotation 코드를 아래 코드로 대체합니다.
            fig_price.add_trace(
                go.Candlestick(
                    x=[next_day],
                    open=[mean_pred],
                    high=[max_pred],
                    low=[min_pred],
                    close=[mean_pred],
                    name="내일 예측 범위",
                    # 색상을 주황색 계열로 통일
                    increasing_line_color="orange",
                    decreasing_line_color="orange",
                )
            )

        else:
            st.info("다음 날 예측 데이터가 없습니다. 예측 스크립트를 실행해주세요.")

        fig_price.update_layout(
            yaxis_title="가격 (원)",
            xaxis_rangeslider_visible=False,
            height=500,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # --- 보조 지표 차트 ---
        st.header("보조 지표 상세")

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
        fig_rsi.add_hline(
            y=70, line_dash="dot", line_color="red", annotation_text="과매수 (70)"
        )
        fig_rsi.add_hline(
            y=30, line_dash="dot", line_color="blue", annotation_text="과매도 (30)"
        )
        fig_rsi.update_layout(
            title="RSI (상대강도지수)",
            yaxis_title="RSI",
            height=250,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        # 2. MACD 차트 (Corrected)
        fig_macd = go.Figure()

        # Add all traces to the figure first
        fig_macd.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="blue"),
            )
        )
        fig_macd.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Signal"],
                mode="lines",
                name="Signal Line",
                line=dict(color="orange"),
            )
        )
        fig_macd.add_trace(
            go.Bar(
                x=df.index,
                y=df["Histogram"],
                name="Histogram",
                marker_color=np.where(df["Histogram"] > 0, "green", "red"),
            )
        )

        # Then, update the layout
        fig_macd.update_layout(title="MACD", height=250, margin=dict(t=30, b=30))

        # Finally, display the chart
        st.plotly_chart(fig_macd, use_container_width=True)
