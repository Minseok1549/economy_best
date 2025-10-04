from __future__ import annotations

import datetime as dt
import warnings

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as pgo
import streamlit as st
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots

# 경고 메시지 무시
warnings.simplefilter(action="ignore", category=FutureWarning)


# --- 2. 페이지 기본 설정 ---
PAGE_CONFIG = {
    "layout": "wide",
    "page_title": "종합 주식 분석 대시보드",
    "page_icon": "🚀",
}
st.set_page_config(**PAGE_CONFIG)


# --- 3. 데이터 로딩 및 처리 함수 ---
@st.cache_data(ttl=600)  # 10분 캐시
def load_stock_data(ticker: str) -> pd.DataFrame | None:
    """FinanceDataReader를 사용하여 주식 데이터를 로드합니다."""
    try:
        df = fdr.DataReader(ticker, start=dt.date.today() - relativedelta(years=2))
        return None if df.empty else df
    except Exception as e:
        st.error(f"'{ticker}' 데이터 로드 중 오류 발생: {e}")
        return None


def load_forecast_data(ticker: str) -> pd.DataFrame | None:
    """사전에 생성된 예측 데이터를 CSV 파일에서 로드합니다."""
    try:
        return pd.read_csv(f"{ticker}_forecast.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        return None


@st.cache_data(ttl=3600)  # 1시간 캐시
def get_company_info(ticker: str) -> dict | None:
    """종목의 기본 정보를 가져옵니다."""
    try:
        market = "KRX" if ticker.isdigit() else "NASDAQ"
        listing = fdr.StockListing(market)
        # Symbol이 ticker와 일치하는 행을 찾아 사전으로 변환
        info_series = listing[listing["Symbol"] == ticker].iloc[0]
        return info_series.to_dict()
    except Exception:
        return None


def add_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """데이터프레임에 모든 기술적 지표를 추가합니다."""
    # 이동평균
    df[f"MA_{params['ma_short']}"] = (
        df["Close"].rolling(window=params["ma_short"]).mean()
    )
    df[f"MA_{params['ma_long']}"] = df["Close"].rolling(window=params["ma_long"]).mean()
    # RSI
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0.0).rolling(window=params["rsi_window"]).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=params["rsi_window"]).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]
    return df


def get_all_signals(df: pd.DataFrame, params: dict) -> dict:
    """모든 기술적 지표에 대한 신호를 생성합니다."""
    signals = {}
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # MA Signal
    ma_short_col = f"MA_{params['ma_short']}"
    ma_long_col = f"MA_{params['ma_long']}"
    if (
        latest[ma_short_col] > latest[ma_long_col]
        and previous[ma_short_col] <= previous[ma_long_col]
    ):
        signals["ma"] = ("🟢 매수", "골든 크로스 발생")
    elif (
        latest[ma_short_col] < latest[ma_long_col]
        and previous[ma_short_col] >= previous[ma_long_col]
    ):
        signals["ma"] = ("🔴 매도", "데드 크로스 발생")
    else:
        trend = "상승" if latest[ma_short_col] > latest[ma_long_col] else "하락"
        signals["ma"] = ("⚪️ 중립", f"현재 {trend} 추세")
    # RSI Signal
    if latest["RSI"] > 70:
        signals["rsi"] = ("🔴 매도", f"과매수 구간 ({latest['RSI']:.1f})")
    elif latest["RSI"] < 30:
        signals["rsi"] = ("🟢 매수", f"과매도 구간 ({latest['RSI']:.1f})")
    else:
        signals["rsi"] = ("⚪️ 중립", f"중립 구간 ({latest['RSI']:.1f})")
    # MACD Signal
    if latest["MACD"] > latest["Signal"] and previous["MACD"] <= previous["Signal"]:
        signals["macd"] = ("🟢 매수", "MACD선이 시그널선 상향 돌파")
    elif latest["MACD"] < latest["Signal"] and previous["MACD"] >= previous["Signal"]:
        signals["macd"] = ("🔴 매도", "MACD선이 시그널선 하향 돌파")
    else:
        trend = "상승" if latest["MACD"] > latest["Signal"] else "하락"
        signals["macd"] = ("⚪️ 중립", f"현재 {trend} 추세")
    return signals


# --- 4. UI 컴포넌트 함수 ---
def setup_sidebar() -> tuple[str, dict, bool]:
    """사이드바 UI를 구성하고 사용자 입력을 받습니다."""
    with st.sidebar:
        st.header("⚙️ 설정")
        stock_list = ["005930", "035720", "000660", "TSLA", "AAPL"]
        ticker = st.selectbox(
            "종목 선택",
            stock_list,
            help="리스트에서 종목을 선택하거나 직접 입력하세요.",
        )
        custom_ticker = st.text_input("또는 종목 코드 직접 입력 (예: GOOGL)")
        if custom_ticker:
            ticker = custom_ticker.upper()

        st.divider()
        st.subheader("차트 설정")
        params = {
            "ma_short": st.number_input("단기 이동평균", 5, 50, 20, 1),
            "ma_long": st.number_input("장기 이동평균", 20, 200, 60, 5),
            "rsi_window": st.number_input("RSI 기간", 5, 30, 14, 1),
        }
        show_bb = st.checkbox("볼린저 밴드 표시", value=True)
    return ticker, params, show_bb


def display_summary(ticker: str, signals: dict, forecast_data: pd.DataFrame | None):
    """'종합 분석' 탭의 내용을 표시합니다."""
    info = get_company_info(ticker)
    company_name = info.get("Name", ticker) if info else ticker

    st.header(f"종합 분석: {company_name}")

    # 1. 기업 정보 및 예측 요약
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏢 기업 기본 정보")
        if info:
            market_cap = info.get("Marcap", 0)
            st.metric("회사명", info.get("Name", "N/A"))
            st.metric("시가총액", f"{(market_cap / 100000000):,.0f} 억원")
            st.caption(f"소속: {info.get('Market', 'N/A')}")
        else:
            st.info("기업 정보를 불러올 수 없습니다.")

    with col2:
        st.subheader("🤖 AI 예측 요약")
        if forecast_data is not None:
            mean_pred = forecast_data.mean(axis=1).iloc[0]
            std_pred = forecast_data.std(axis=1).iloc[0]
            st.metric("다음 거래일 평균 예측가", f"{mean_pred:,.2f}")

            confidence = max(0, 100 - (std_pred / mean_pred * 200))
            st.write("예측 신뢰도:")
            st.progress(int(confidence), text=f"{confidence:.1f} / 100")
        else:
            st.info("예측 데이터가 없습니다.")

    st.divider()

    # 2. 기술적 분석 신호
    st.subheader("📊 종합 기술적 분석 신호")
    cols = st.columns(3)
    signal_map = {"ma": "이동평균 (MA)", "rsi": "RSI", "macd": "MACD"}
    for i, (key, name) in enumerate(signal_map.items()):
        with cols[i]:
            st.metric_label = f"**{name}**"
            signal, reason = signals[key]
            st.markdown(f"##### {name}")
            st.markdown(f"## {signal}")
            st.caption(reason)


def display_charts(
    df: pd.DataFrame, forecast_data: pd.DataFrame | None, params: dict, show_bb: bool
):
    """'상세 차트' 탭의 내용을 표시합니다."""
    st.header("📈 상세 차트 분석")

    # 1. 가격 및 거래량 차트
    fig_price = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2]
    )
    fig_price.add_trace(
        pgo.Candlestick(
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
    fig_price.add_trace(
        pgo.Scatter(
            x=df.index,
            y=df[f"MA_{params['ma_short']}"],
            name=f"{params['ma_short']}일 MA",
            line=dict(color="orange"),
        ),
        row=1,
        col=1,
    )
    fig_price.add_trace(
        pgo.Scatter(
            x=df.index,
            y=df[f"MA_{params['ma_long']}"],
            name=f"{params['ma_long']}일 MA",
            line=dict(color="purple"),
        ),
        row=1,
        col=1,
    )
    if show_bb:
        ma20 = df["Close"].rolling(window=20).mean()
        std20 = df["Close"].rolling(window=20).std()
        upper = ma20 + (std20 * 2)
        lower = ma20 - (std20 * 2)
        fig_price.add_trace(
            pgo.Scatter(
                x=df.index,
                y=upper,
                name="상단밴드",
                line=dict(color="rgba(168, 162, 255, 0.4)"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig_price.add_trace(
            pgo.Scatter(
                x=df.index,
                y=lower,
                name="하단밴드",
                line=dict(color="rgba(168, 162, 255, 0.4)"),
                fill="tonexty",
                fillcolor="rgba(168, 162, 255, 0.1)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if forecast_data is not None:
        next_day = forecast_data.index[0]
        mean_pred = forecast_data.mean(axis=1).iloc[0]
        fig_price.add_trace(
            pgo.Candlestick(
                x=[next_day],
                open=[mean_pred],
                high=[forecast_data.max(axis=1).iloc[0]],
                low=[forecast_data.min(axis=1).iloc[0]],
                close=[mean_pred],
                name="내일 예측",
                increasing_line_color="royalblue",
                decreasing_line_color="royalblue",
            ),
            row=1,
            col=1,
        )
    fig_price.add_trace(
        pgo.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color="lightgray"),
        row=2,
        col=1,
    )
    fig_price.update_layout(
        title_text="<b>가격 및 거래량 차트</b>",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_price.update_yaxes(title_text="가격", row=1, col=1)
    fig_price.update_yaxes(title_text="거래량", row=2, col=1)
    st.plotly_chart(fig_price, use_container_width=True)

    # 2. 보조 지표 차트
    st.subheader("보조 지표")
    col1, col2 = st.columns(2)
    with col1:
        fig_rsi = pgo.Figure()
        fig_rsi.add_trace(pgo.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="blue")
        fig_rsi.update_layout(
            title_text="<b>RSI (상대강도지수)</b>", height=300, margin=dict(t=50, b=30)
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    with col2:
        fig_macd = pgo.Figure()
        fig_macd.add_trace(
            pgo.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue"))
        )
        fig_macd.add_trace(
            pgo.Scatter(
                x=df.index, y=df["Signal"], name="Signal", line=dict(color="orange")
            )
        )
        fig_macd.add_trace(
            pgo.Bar(
                x=df.index,
                y=df["Histogram"],
                name="Histogram",
                marker_color=np.where(df["Histogram"] > 0, "mediumseagreen", "salmon"),
            )
        )
        fig_macd.update_layout(
            title_text="<b>MACD</b>", height=300, margin=dict(t=50, b=30)
        )
        st.plotly_chart(fig_macd, use_container_width=True)


def display_news(ticker: str):
    """'기업 정보 및 뉴스' 탭의 내용을 표시합니다."""
    st.header("📰 관련 뉴스 및 정보")
    st.info("뉴스 정보 연동 기능은 현재 개발 중입니다.", icon="🚧")
    # 향후 뉴스 API 또는 크롤링을 통해 관련 정보를 여기에 표시할 수 있습니다.


# --- 5. 메인 애플리케이션 실행 ---
def main():
    """메인 대시보드 애플리케이션을 실행합니다."""
    info = get_company_info("005930")  # 초기 로딩용
    st.title(f"🚀 종합 주식 분석 대시보드")

    # 1. 사용자 설정 입력 (사이드바)
    ticker, params, show_bb = setup_sidebar()

    # 2. 데이터 로드 및 처리
    stock_data = load_stock_data(ticker)
    if stock_data is None:
        st.warning(f"'{ticker}' 종목 데이터를 찾을 수 없습니다. 코드를 확인해주세요.")
        return  # 데이터 없으면 실행 중단

    forecast_data = load_forecast_data(ticker)
    data_with_indicators = add_all_indicators(stock_data.copy(), params)
    signals = get_all_signals(data_with_indicators, params)

    # 3. 화면에 탭(Tab) 구성 및 표시
    tab1, tab2, tab3 = st.tabs(["📊 종합 분석", "📈 상세 차트", "📰 기업 정보 및 뉴스"])

    with tab1:
        display_summary(ticker, signals, forecast_data)

    with tab2:
        display_charts(data_with_indicators, forecast_data, params, show_bb)

    with tab3:
        display_news(ticker)


if __name__ == "__main__":
    main()
