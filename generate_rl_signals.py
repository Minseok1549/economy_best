# generate_rl_signals.py
import json
import os

import FinanceDataReader as fdr
import numpy as np
from stable_baselines3 import PPO

SIGNAL_DATA_PATH = "data/signals"
if not os.path.exists(SIGNAL_DATA_PATH):
    os.makedirs(SIGNAL_DATA_PATH)


def generate_signal(ticker):
    print(f"[{ticker}] 강화학습 에이전트의 매매 신호를 생성합니다...")
    try:
        model = PPO.load(f"models/{ticker}/rl_agent")

        df = fdr.DataReader(ticker, start="2024-01-01")
        delta = df["Close"].diff(1)
        gain = delta.where(delta > 0, 0).rolling(window=14).mean().fillna(0)
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean().fillna(0)
        df["RSI"] = 100 - (100 / (1 + (gain / loss)))
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = macd - signal_line
        df.dropna(inplace=True)

        latest_obs = np.array(
            [0, 0, df.iloc[-1]["Close"], df.iloc[-1]["RSI"], df.iloc[-1]["MACD_Hist"]]
        ).astype(np.float32)

        action, _ = model.predict(latest_obs, deterministic=True)

        action_map = {0: "🟢 매수", 1: "🔴 매도", 2: "⚪️ 관망"}
        signal = action_map[int(action)]

        output_path = os.path.join(SIGNAL_DATA_PATH, f"{ticker}_rl_signal.json")
        with open(output_path, "w") as f:
            json.dump({"signal": signal, "reason": "RL 에이전트 결정"}, f)

        print(f"[{ticker}] 신호 생성 완료: {signal}")

    except Exception as e:
        print(f"[{ticker}] 신호 생성 실패: {e}")


if __name__ == "__main__":
    TICKERS_TO_SIGNAL = ["005930", "TSLA", "AAPL", "NVDA"]
    for ticker in TICKERS_TO_SIGNAL:
        generate_signal(ticker)
