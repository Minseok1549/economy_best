from __future__ import annotations

import datetime as dt
import warnings

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as pgo
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots
from pykrx import stock

# 경고 메시지 무시
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- 2. 페이지 기본 설정 ---
PAGE_CONFIG = {
    "layout": "wide",
    "page_title": "개인 투자 전략 어시스턴트",
    "page_icon": "💡",
}
st.set_page_config(**PAGE_CONFIG)


# --- 3. 핵심 데이터 처리 및 분석 함수 ---
@st.cache_data(ttl=600)  # 10분 캐시
def load_stock_data(ticker: str) -> pd.DataFrame | None:
    """주가 데이터를 로드합니다."""
    try:
        df = fdr.DataReader(ticker, start=dt.date.today() - dt.timedelta(days=730))
        return None if df.empty else df
    except Exception:
        return None


def load_forecast_data(ticker: str) -> pd.DataFrame | None:
    """사전에 생성된 예측 데이터를 로드합니다."""
    try:
        return pd.read_csv(f"{ticker}_forecast.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        return None


@st.cache_data(ttl=3600)  # 1시간 캐시
def get_usd_krw_rate() -> float:
    """최신 USD/KRW 환율 정보를 가져옵니다."""
    try:
        df = fdr.DataReader("USD/KRW", start=dt.date.today() - dt.timedelta(days=30))
        return df["Close"].iloc[-1]
    except Exception:
        return 1300.0  # 실패 시 기본값


@st.cache_data(ttl=3600)
def get_fundamental_info(ticker: str) -> dict:
    """국내/해외 주식의 펀더멘털 및 뉴스 정보를 통합하여 가져옵니다."""
    info = {
        "name": ticker,
        "summary": "N/A",
        "marcap": 0,
        "per": 0,
        "pbr": 0,
        "news": [],
    }

    if ticker.isdigit():  # 국내 주식 (pykrx)
        try:
            today_str = dt.date.today().strftime("%Y%m%d")
            info["name"] = stock.get_market_ticker_name(ticker)
            f_df = stock.get_market_fundamental(today_str, market="ALL")
            f_info = f_df.loc[ticker]
            info.update({"per": f_info.get("PER", 0), "pbr": f_info.get("PBR", 0)})
            mc_df = stock.get_market_cap(today_str, market="ALL")
            info["marcap"] = mc_df.loc[ticker, "시가총액"]
            info["news"] = get_naver_news(ticker)
        except Exception:
            pass  # 정보가 없는 경우 기본값 사용
    else:  # 해외 주식 (yfinance)
        try:
            yticker = yf.Ticker(ticker)
            yinfo = yticker.info
            info["name"] = yinfo.get("shortName", ticker)
            info["summary"] = yinfo.get("longBusinessSummary", "상세 정보 없음.")
            info["marcap"] = yinfo.get("marketCap", 0)
            info["per"] = yinfo.get("trailingPE", 0)
            info["pbr"] = yinfo.get("priceToBook", 0)
            news_raw = yticker.news
            info["news"] = [
                {"title": n["title"], "link": n["link"]} for n in news_raw[:3]
            ]
        except Exception:
            pass
    return info


@st.cache_data(ttl=600)
def get_naver_news(ticker: str) -> list:
    """네이버 금융에서 최신 뉴스 헤드라인 3개를 스크레이핑합니다."""
    news_list = []
    try:
        url = f"https://finance.naver.com/item/news_news.naver?code={ticker}&page=1"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        for row in soup.select("table.type5 tr")[:3]:
            title_tag = row.select_one("td.title a")
            if title_tag:
                news_list.append(
                    {
                        "title": title_tag.text.strip(),
                        "link": "https://finance.naver.com" + title_tag["href"],
                    }
                )
    except Exception:
        pass
    return news_list


def add_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """모든 기술적 지표를 계산하고 데이터프레임에 추가합니다."""
    df[f"MA_{params['ma_short']}"] = (
        df["Close"].rolling(window=params["ma_short"]).mean()
    )
    df[f"MA_{params['ma_long']}"] = df["Close"].rolling(window=params["ma_long"]).mean()
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0.0).rolling(window=params["rsi_window"]).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=params["rsi_window"]).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]
    return df


def get_final_signal_and_targets(df: pd.DataFrame, params: dict) -> dict:
    """기술적 지표를 종합하여 최종 신호와 목표가를 생성합니다."""
    # 1. 개별 신호 생성
    signals, latest, previous = {}, df.iloc[-1], df.iloc[-2]
    ma_short, ma_long = f"MA_{params['ma_short']}", f"MA_{params['ma_long']}"
    if latest[ma_short] > latest[ma_long] and previous[ma_short] <= previous[ma_long]:
        signals["ma"] = 1
    elif latest[ma_short] < latest[ma_long] and previous[ma_short] >= previous[ma_long]:
        signals["ma"] = -1
    if latest["RSI"] < 30:
        signals["rsi"] = 1
    elif latest["RSI"] > 70:
        signals["rsi"] = -1
    if latest["MACD"] > latest["Signal"] and previous["MACD"] <= previous["Signal"]:
        signals["macd"] = 1
    elif latest["MACD"] < latest["Signal"] and previous["MACD"] >= previous["Signal"]:
        signals["macd"] = -1

    # 2. 최종 신호 결정
    score = sum(signals.values())
    if score >= 2:
        final_signal = ("🟢 강력 매수", f"{score}개 지표 매수")
    elif score <= -2:
        final_signal = ("🔴 적극 매도", f"{-score}개 지표 매도")
    elif score == 1:
        final_signal = ("🟡 매수 고려", "매수 우위")
    elif score == -1:
        final_signal = ("🟡 매도 고려", "매도 우위")
    else:
        final_signal = ("⚪️ 관망", "신호 혼재")

    # 3. 목표가 및 손절가 계산
    recent_60 = df.iloc[-60:]
    resistance = recent_60["High"].max()
    support = recent_60["Low"].min()

    return {"signal": final_signal, "target": resistance, "loss_cut": support}


# --- 4. UI 렌더링 함수 ---
def setup_sidebar() -> tuple[str, dict, bool]:
    """사이드바 UI 구성"""
    with st.sidebar:
        st.header("⚙️ 설정")
        ticker = st.selectbox(
            "종목 선택", ["005930", "TSLA", "035720", "AAPL", "000660"]
        )
        custom_ticker = st.text_input(
            "또는 종목 코드 직접 입력", placeholder="예: GOOGL"
        )
        if custom_ticker:
            ticker = custom_ticker.upper()
        st.divider()
        st.subheader("차트 설정")
        params = {
            "ma_short": st.number_input("단기 이평선", 5, 50, 20),
            "ma_long": st.number_input("장기 이평선", 20, 200, 60),
            "rsi_window": st.number_input("RSI 기간", 5, 30, 14),
        }
        show_bb = st.checkbox("볼린저 밴드 표시", True)
    return ticker, params, show_bb


def display_strategy_panel(
    df: pd.DataFrame,
    strategy: dict,
    forecast: pd.DataFrame | None,
    krw_rate: float,
    is_kr_stock: bool,
):
    """'투자 전략' 탭의 내용을 표시합니다."""
    st.header("💡 투자 전략 어시스턴트")
    current_price = df.iloc[-1]["Close"]
    unit = "원" if is_kr_stock else "달러"

    col1, col2 = st.columns(2)
    with col1:  # 최종 신호
        st.metric("최종 투자 신호", strategy["signal"][0], help=strategy["signal"][1])
        st.write(f"현재가: **{current_price:,.2f} {unit}**")
        if not is_kr_stock:
            st.caption(f"KRW: 약 {(current_price * krw_rate):,.0f} 원")

    with col2:  # 목표가 및 손절가
        target_delta = f"{(strategy['target'] / current_price - 1) * 100:.1f}%"
        loss_cut_delta = f"{(strategy['loss_cut'] / current_price - 1) * 100:.1f}%"
        st.metric(
            "🎯 1차 목표가 (저항선)",
            f"{strategy['target']:,.2f} {unit}",
            delta=target_delta,
        )
        st.metric(
            "🛡️ 손절가 (지지선)",
            f"{strategy['loss_cut']:,.2f} {unit}",
            delta=loss_cut_delta,
            delta_color="inverse",
        )

    st.divider()
    st.subheader("🤖 AI 예측 요약")
    if forecast is not None:
        mean_pred = forecast.mean(axis=1).iloc[0]
        std_pred = forecast.std(axis=1).iloc[0]
        st.metric("다음 거래일 평균 예측가", f"{mean_pred:,.2f} {unit}")
        confidence = max(0, 100 - (std_pred / mean_pred * 200))
        st.progress(int(confidence), text=f"예측 신뢰도: {confidence:.1f} / 100")
    else:
        st.info("예측 데이터를 찾을 수 없습니다.")
    st.caption("주의: 본 정보는 알고리즘 분석 결과이며, 투자 추천이 아닙니다.")


def display_charts(
    df: pd.DataFrame,
    forecast: pd.DataFrame | None,
    params: dict,
    show_bb: bool,
    krw_rate: float,
    is_kr_stock: bool,
):
    """'상세 차트' 탭의 내용을 표시합니다."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.8, 0.2]
    )
    # 가격 차트
    fig.add_trace(
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
    fig.add_trace(
        pgo.Scatter(
            x=df.index,
            y=df[f"MA_{params['ma_short']}"],
            name=f"{params['ma_short']}일 MA",
            line=dict(color="orange"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
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
        fig.add_trace(
            pgo.Scatter(
                x=df.index, y=ma20 + (std20 * 2), line=dict(width=0), showlegend=False
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            pgo.Scatter(
                x=df.index,
                y=ma20 - (std20 * 2),
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(168,162,255,0.1)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if forecast is not None:
        next_day = forecast.index[0]
        mean_pred = forecast.mean(axis=1).iloc[0]
        fig.add_trace(
            pgo.Candlestick(
                x=[next_day],
                open=[mean_pred],
                high=[forecast.max(axis=1).iloc[0]],
                low=[forecast.min(axis=1).iloc[0]],
                close=[mean_pred],
                name="내일 예측",
                increasing_line_color="royalblue",
            ),
            row=1,
            col=1,
        )
    # 거래량 차트
    fig.add_trace(
        pgo.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color="lightgray"),
        row=2,
        col=1,
    )
    unit = "원" if is_kr_stock else "USD"
    fig.update_layout(
        title_text="<b>가격 및 거래량 차트</b>",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text=f"가격 ({unit})", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def display_info_and_news(info: dict):
    """'기업 정보 및 뉴스' 탭의 내용을 표시합니다."""
    st.header(f"🏢 {info.get('name', 'N/A')} 기업 정보")

    col1, col2, col3 = st.columns(3)
    col1.metric("시가총액", f"{(info.get('marcap', 0) / 100000000):,.0f} 억원")
    col2.metric("PER", f"{info.get('per', 0):.2f}")
    col3.metric("PBR", f"{info.get('pbr', 0):.2f}")

    with st.expander("사업 요약 보기"):
        st.write(info.get("summary", "제공된 요약 정보가 없습니다."))

    st.divider()
    st.subheader("📰 최신 뉴스")
    if info["news"]:
        for item in info["news"]:
            st.markdown(f"- [{item['title']}]({item['link']})")
    else:
        st.info("최신 뉴스를 불러올 수 없습니다.")


# --- 5. 메인 애플리케이션 실행부 ---
def main():
    """메인 애플리케이션을 실행합니다."""
    st.title("💡 개인 투자 전략 어시스턴트")

    ticker, params, show_bb = setup_sidebar()
    stock_data = load_stock_data(ticker)

    if stock_data is None:
        st.warning(f"'{ticker}' 종목 데이터를 찾을 수 없습니다.")
        return

    is_kr_stock = ticker.isdigit()
    krw_rate = 1.0 if is_kr_stock else get_usd_krw_rate()

    # 데이터 처리
    forecast_data = load_forecast_data(ticker)
    data_with_indicators = add_all_indicators(stock_data.copy(), params)
    strategy = get_final_signal_and_targets(data_with_indicators, params)
    fundamental_info = get_fundamental_info(ticker)

    # 탭 렌더링
    tab1, tab2, tab3 = st.tabs(["💡 투자 전략", "📈 상세 차트", "📰 기업 정보"])
    with tab1:
        display_strategy_panel(
            data_with_indicators, strategy, forecast_data, krw_rate, is_kr_stock
        )
    with tab2:
        display_charts(
            data_with_indicators, forecast_data, params, show_bb, krw_rate, is_kr_stock
        )
    with tab3:
        display_info_and_news(fundamental_info)


if __name__ == "__main__":
    main()
