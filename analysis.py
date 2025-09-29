from __future__ import annotations

import datetime as dt

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# 1. Moving average


def calculate_ma(close_prices: pd.Series, window: int) -> pd.Series:
    """
    주어진 기간(window)의 이동 평균을 계산합니다.

    :params close_prices: 종가 데이터 (pd.Series)
    :params window: 이동 평균을 계산할 기간 (int)
    :return: 이동 평균 (pd.Series)
    """
    return close_prices.rolling(window=window).mean()


def interpret_ma_cross(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """
    단기 이동 평균과 장기 이동 평균의 교차를 해석합니다.

    :params short_ma: 단기 이동 평균 (pd.Series)
    :params long_ma: 장기 이동 평균 (pd.Series)
    :return: "Buy" (골든크로스), "Sell" (데드크로스), "Hold" 신호
    """
    signals = pd.Series(index=short_ma.index, data="Hold")
    # 골든크로스: 단기 이평선이 장기 이평선을 상향 돌파할 때
    signals[(short_ma.shift(1) < long_ma.shift(1)) & (short_ma > long_ma)] = "Buy"

    # 데드크로스: 단기 이평선이 장기 이평선을 하향 돌파할 때
    signals[(short_ma.shift(1) > long_ma.shift(1)) & (short_ma < long_ma)] = "Sell"
    return signals


# 2. 상대강도지수 (Relative Strength Index, RSI)
def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI(상대강도지수) 를 계산합니다.

    :params close_prices: 종가 데이터 (pd.Series)
    :params window: RSI를 계산할 기간 (int, 기본값=14)
    :return: RSI 값 (pd.Series)
    """
    delta = pd.to_numeric(close_prices.diff(1), errors="coerce")
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def interpret_rsi(
    rsi: pd.Series, overbought_threshold: int = 70, oversold_threshold: int = 30
) -> pd.Series:
    """
    RSI 값을 해석하여 과매수/과매도 신호를 생성합니다.

    :params rsi: RSI 값 (pd.Series)
    :params overbought_threshold: 과매수 기준 (int, 기본값=70)
    :params oversold_threshold: 과매도 기준 (int, 기본값=30)
    :return: "Buy" (과매도), "Sell" (과매수), "Hold" 신호
    """
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
    """
    MACD (이동 평균 수렴 확산 지수)를 계산합니다.

    :params close_prices: 종가 데이터 (pd.Series)
    :params short_window: 단기 EMA 기간 (int, 기본값=12)
    :params long_window: 장기 EMA 기간 (int, 기본값=26)
    :params signal_window: 신호선 EMA 기간 (int, 기본값=9)
    :return: MACD와 신호선이 포함된 DataFrame (pd.DataFrame)
    """
    short_ema = close_prices.ewm(span=short_window, adjust=False).mean()
    long_ema = close_prices.ewm(span=long_window, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram}
    )


def interpret_macd(macd_df: pd.DataFrame) -> pd.Series:
    """
    MACD 교차를 기반으로 매매 신호를 생성합니다.

    :params macd_df: MACD와 신호선이 포함된 DataFrame (pd.DataFrame)
    :return: "Buy" (MACD가 신호선을 상향 돌파), "Sell" (MACD가 신호선을 하향 돌파), "Hold" 신호
    """
    signals = pd.Series(index=macd_df.index, data="Hold")
    # 골든크로스: MACD선이 시그널선을 상향 돌파할 때
    signals[
        (macd_df["MACD"].shift(1) < macd_df["Signal"].shift(1))
        & (macd_df["MACD"] > macd_df["Signal"])
    ] = "Buy"
    # 데드크로스: MACD선이 시그널선을 하향 돌파할 때
    signals[
        (macd_df["MACD"].shift(1) > macd_df["Signal"].shift(1))
        & (macd_df["MACD"] < macd_df["Signal"])
    ] = "Sell"
    return signals


if __name__ == "__main__":
    # --- 페이지 기본 설정 ---
    st.set_page_config(
        layout="wide",
        page_title="기술적 지표 분석 툴",
    )
    st.title("📈 주식 기술적 지표 분석 대시보드")
    st.markdown(
        "관심 종목의 주가와 주요 기술적 지표들을 시각화하고, 매매 신호를 확인하세요."
    )

    # --- 사이드바 설정 ---
    st.sidebar.header("⚙️ 설정")

    # 종목 코드 입력
    ticker = st.sidebar.text_input("종목 코드 (Ticker)", "000660")

    # 기간 설정
    today = dt.date.today()
    start_date = st.sidebar.date_input("시작일", today - dt.timedelta(days=365))
    end_date = st.sidebar.date_input("종료일", today)

    # 지표 파라미터 설정
    st.sidebar.subheader("지표 파라미터")
    ma_short_window = st.sidebar.number_input(
        "단기 이동평균 기간", min_value=5, max_value=50, value=20, step=1
    )
    ma_long_window = st.sidebar.number_input(
        "장기 이동평균 기간", min_value=20, max_value=200, value=60, step=5
    )
    rsi_window = st.sidebar.number_input(
        "RSI 기간", min_value=5, max_value=30, value=14, step=1
    )

    # 분석 시작 버튼
    if st.sidebar.button("📊 분석 시작"):
        try:
            # --- 데이터 로드 ---
            df = fdr.DataReader(ticker, start_date, end_date)
            if df.empty:
                st.error(
                    "해당 기간에 대한 데이터가 없습니다. 종목 코드나 기간을 확인해주세요."
                )
            else:
                # --- 지표 계산 ---
                df["MA_Short"] = calculate_ma(df["Close"], ma_short_window)
                df["MA_Long"] = calculate_ma(df["Close"], ma_long_window)
                df["RSI"] = calculate_rsi(df["Close"], rsi_window)
                macd_df = calculate_macd(df["Close"])
                df = df.join(macd_df)

                # --- 신호 해석 ---
                macd_signals = interpret_macd(df)

                # --- 대시보드 출력 ---
                st.header(f"'{ticker}' 기술적 분석 결과")

                # 최신 정보 요약
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

                # --- 시각화 ---
                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.5, 0.1, 0.2, 0.2],
                )

                # 1. 캔들차트, 이동평균선, 매매신호
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

                buy_signals = df.loc[macd_signals == "Buy"]
                sell_signals = df.loc[macd_signals == "Sell"]
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals["Close"],
                        mode="markers",
                        name="Buy Signal",
                        marker=dict(symbol="triangle-up", color="green", size=10),
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals["Close"],
                        mode="markers",
                        name="Sell Signal",
                        marker=dict(symbol="triangle-down", color="red", size=10),
                    ),
                    row=1,
                    col=1,
                )

                # 2. 거래량
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

                # 3. RSI
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
                fig.add_hline(y=70, line_dash="dash", line_color="red")
                fig.add_hline(y=30, line_dash="dash", line_color="green")

                # 4. MACD
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
                colors = ["green" if val >= 0 else "red" for val in df["Histogram"]]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df["Histogram"],
                        name="Histogram",
                        marker_color=colors,
                    ),
                    row=4,
                    col=1,
                )

                # 레이아웃 업데이트
                fig.update_layout(
                    title_text=f"{ticker} 종합 기술적 분석",
                    height=800,
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

                # 데이터 테이블
                with st.expander("상세 데이터 보기"):
                    st.dataframe(
                        df.iloc[::-1].style.format("{:,.2f}")
                    )  # 최신 데이터가 위로 오도록

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
