from __future__ import annotations

import datetime as dt

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- CONFIGURATION (설정) ---
# 애플리케이션의 기본 설정을 관리합니다.
PAGE_CONFIG = {"layout": "wide", "page_title": "주가 예측 대시보드"}


# --- DATA HANDLING (데이터 처리) ---
@st.cache_data(ttl=600)  # 10분 동안 캐시하여 불필요한 API 호출 방지
def load_stock_data(ticker: str) -> pd.DataFrame | None:
    """
    FinanceDataReader를 사용하여 특정 종목의 최신 주가 데이터를 불러옵니다.
    최근 2년치 데이터를 가져옵니다.

    Args:
        ticker (str): 조회할 종목의 티커 (예: '005930')

    Returns:
        pd.DataFrame | None: 주가 데이터프레임 또는 실패 시 None
    """
    today = dt.date.today()
    start_date = today - dt.timedelta(days=730)
    try:
        df = fdr.DataReader(ticker, start_date, today)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None


def load_forecast_data(ticker: str) -> pd.DataFrame | None:
    """
    사전에 생성된 예측 데이터를 CSV 파일에서 로드합니다.
    파일 경로는 '티커_forecast.csv' 형식을 따릅니다.

    Args:
        ticker (str): 예측 데이터를 로드할 종목의 티커

    Returns:
        pd.DataFrame | None: 예측 데이터프레임 또는 파일이 없을 경우 None
    """
    try:
        forecast_df = pd.read_csv(
            f"{ticker}_forecast.csv", index_col=0, parse_dates=True
        )
        return forecast_df
    except FileNotFoundError:
        st.info(
            f"'{ticker}'에 대한 예측 데이터를 찾을 수 없습니다. 예측 스크립트를 먼저 실행해주세요."
        )
        return None


# --- TECHNICAL INDICATORS (기술적 지표 계산) ---
def calculate_technical_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    주어진 데이터프레임에 각종 기술적 지표를 계산하여 추가합니다.

    Args:
        df (pd.DataFrame): 원본 주가 데이터프레임
        **kwargs: 각 지표 계산에 필요한 파라미터 (예: ma_short=20, rsi_window=14)

    Returns:
        pd.DataFrame: 기술적 지표가 추가된 데이터프레임
    """
    # 이동평균
    df[f"MA_{kwargs['ma_short']}"] = (
        df["Close"].rolling(window=kwargs["ma_short"]).mean()
    )
    df[f"MA_{kwargs['ma_long']}"] = df["Close"].rolling(window=kwargs["ma_long"]).mean()

    # RSI
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=kwargs["rsi_window"]).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=kwargs["rsi_window"]).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]
    return df


# --- SIGNAL GENERATION (신호 생성) ---
def generate_signals(df: pd.DataFrame, **kwargs) -> dict:
    """
    기술적 지표를 바탕으로 매매 신호를 생성합니다.

    Args:
        df (pd.DataFrame): 기술적 지표가 포함된 데이터프레임
        **kwargs: 신호 생성에 필요한 파라미터 (예: ma_short=20, ma_long=60)

    Returns:
        dict: 각 지표별 신호와 설명을 담은 딕셔너리
    """
    signals = {}
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # 1. 이동평균 (MA) 신호
    ma_short_col = f"MA_{kwargs['ma_short']}"
    ma_long_col = f"MA_{kwargs['ma_long']}"
    if (
        latest[ma_short_col] > latest[ma_long_col]
        and previous[ma_short_col] <= previous[ma_long_col]
    ):
        signals["ma"] = (
            "🟢 매수",
            "단기 이평선이 장기 이평선을 상향 돌파 (골든 크로스)",
        )
    elif (
        latest[ma_short_col] < latest[ma_long_col]
        and previous[ma_short_col] >= previous[ma_long_col]
    ):
        signals["ma"] = (
            "🔴 매도",
            "단기 이평선이 장기 이평선을 하향 돌파 (데드 크로스)",
        )
    else:
        trend = "상승" if latest[ma_short_col] > latest[ma_long_col] else "하락"
        signals["ma"] = ("⚪️ 중립", f"현재 {trend} 추세 유지 중")

    # 2. RSI 신호
    if latest["RSI"] > 70:
        signals["rsi"] = ("🔴 매도", f"RSI({latest['RSI']:.1f})가 과매수(70) 구간 진입")
    elif latest["RSI"] < 30:
        signals["rsi"] = ("🟢 매수", f"RSI({latest['RSI']:.1f})가 과매도(30) 구간 진입")
    else:
        signals["rsi"] = ("⚪️ 중립", f"RSI({latest['RSI']:.1f})가 중립 구간에 위치")

    # 3. MACD 신호
    if latest["MACD"] > latest["Signal"] and previous["MACD"] <= previous["Signal"]:
        signals["macd"] = ("🟢 매수", "MACD선이 시그널선을 상향 돌파")
    elif latest["MACD"] < latest["Signal"] and previous["MACD"] >= previous["Signal"]:
        signals["macd"] = ("🔴 매도", "MACD선이 시그널선을 하향 돌파")
    else:
        trend = "상승" if latest["MACD"] > latest["Signal"] else "하락"
        signals["macd"] = ("⚪️ 중립", f"현재 {trend} 추세 유지 중")

    return signals


# --- PLOTTING (차트 생성) ---
def plot_price_chart(df: pd.DataFrame, forecast_df: pd.DataFrame | None, **kwargs):
    """주가, 이동평균선, 예측 범위를 포함한 메인 차트를 생성합니다."""
    fig = go.Figure()
    # 캔들차트
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="캔들차트",
        )
    )
    # 이동평균선
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"MA_{kwargs['ma_short']}"],
            mode="lines",
            name=f"{kwargs['ma_short']}일 이동평균",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"MA_{kwargs['ma_long']}"],
            mode="lines",
            name=f"{kwargs['ma_long']}일 이동평균",
            line=dict(color="purple"),
        )
    )
    # 예측 데이터 시각화
    if forecast_df is not None:
        next_day = forecast_df.index[0]
        mean_pred = forecast_df.mean(axis=1).iloc[0]
        fig.add_trace(
            go.Candlestick(
                x=[next_day],
                open=[mean_pred],
                high=[forecast_df.max(axis=1).iloc[0]],
                low=[forecast_df.min(axis=1).iloc[0]],
                close=[mean_pred],
                name="내일 예측 범위",
                increasing_line_color="royalblue",
                decreasing_line_color="royalblue",
            )
        )
    fig.update_layout(
        title="<b>가격 차트 및 다음 날 예측</b>",
        yaxis_title="가격 (원)",
        xaxis_rangeslider_visible=False,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_rsi_chart(df: pd.DataFrame):
    """RSI 보조 지표 차트를 생성합니다."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
    fig.add_hline(
        y=70, line_dash="dot", line_color="red", annotation_text="과매수 (70)"
    )
    fig.add_hline(
        y=30, line_dash="dot", line_color="blue", annotation_text="과매도 (30)"
    )
    fig.update_layout(
        title="<b>RSI (상대강도지수)</b>",
        yaxis_title="RSI",
        height=250,
        margin=dict(t=50, b=30),
    )
    return fig


def plot_macd_chart(df: pd.DataFrame):
    """MACD 보조 지표 차트를 생성합니다."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue"))
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Signal"], name="Signal Line", line=dict(color="orange")
        )
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Histogram"],
            name="Histogram",
            marker_color=np.where(df["Histogram"] > 0, "green", "red"),
        )
    )
    fig.update_layout(title="<b>MACD</b>", height=250, margin=dict(t=50, b=30))
    return fig


# --- MAIN APPLICATION (메인 대시보드) ---
def main():
    """메인 대시보드 애플리케이션을 실행합니다."""
    st.set_page_config(**PAGE_CONFIG)
    st.title("📈 실시간 기술적 분석 및 주가 예측 대시보드")

    # --- 1. 사이드바 설정 ---
    with st.sidebar:
        st.header("⚙️ 설정")
        # 주요 종목 리스트 제공 및 직접 입력 기능
        stock_list = ["005930", "035720", "000660", "TSLA", "AAPL"]
        ticker = st.selectbox("종목 선택", stock_list)
        custom_ticker = st.text_input("또는 종목 코드 직접 입력 (예: GOOGL)")
        if custom_ticker:
            ticker = custom_ticker.upper()

        st.divider()
        ma_short = st.number_input("단기 이동평균", 5, 50, 20, 1)
        ma_long = st.number_input("장기 이동평균", 20, 200, 60, 5)
        rsi_window = st.number_input("RSI 기간", 5, 30, 14, 1)
        # 지표 계산에 필요한 파라미터를 딕셔너리로 묶어 관리
        tech_params = {
            "ma_short": ma_short,
            "ma_long": ma_long,
            "rsi_window": rsi_window,
        }

    # --- 2. 데이터 로드 및 처리 ---
    data = load_stock_data(ticker)
    if data is None:
        st.error(f"'{ticker}' 종목 데이터를 불러오지 못했습니다. 코드를 확인해주세요.")
        return  # 데이터가 없으면 실행 중단

    data_with_indicators = calculate_technical_indicators(data.copy(), **tech_params)
    signals = generate_signals(data_with_indicators, **tech_params)
    forecast_data = load_forecast_data(ticker)

    # --- 3. UI 렌더링 ---
    st.header(f"'{ticker}' 종합 기술적 분석 신호")
    cols = st.columns(3)
    signal_map = {"ma": "이동평균 (MA)", "rsi": "RSI", "macd": "MACD"}
    for i, (key, name) in enumerate(signal_map.items()):
        with cols[i]:
            st.subheader(name)
            signal, reason = signals[key]
            st.markdown(f"#### 신호: **{signal}**")
            st.caption(reason)

    st.divider()

    # 차트 시각화
    st.plotly_chart(
        plot_price_chart(data_with_indicators, forecast_data, **tech_params),
        use_container_width=True,
    )

    st.header("보조 지표 상세")
    st.plotly_chart(plot_rsi_chart(data_with_indicators), use_container_width=True)
    st.plotly_chart(plot_macd_chart(data_with_indicators), use_container_width=True)


if __name__ == "__main__":
    main()
