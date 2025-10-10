# train_rl_agent.py

import os

import FinanceDataReader as fdr
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


# --- 1. 가상 주식 거래 환경 (StockTradingEnv) ---
class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = 10000000  # 초기 자본금 1천만원
        self.transaction_fee_percent = 0.0015  # 거래 수수료

        self.action_space = gym.spaces.Discrete(3)  # 0: 매수, 1: 매도, 2: 관망
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        prev_net_worth = self.net_worth

        current_price = self.df.iloc[self.current_step]["Close"]
        if action == 0:  # 매수
            if self.balance > current_price:
                num_shares_to_buy = self.balance // current_price
                self.shares_held += num_shares_to_buy
                self.balance -= (
                    num_shares_to_buy
                    * current_price
                    * (1 + self.transaction_fee_percent)
                )
        elif action == 1:  # 매도
            if self.shares_held > 0:
                self.balance += (
                    self.shares_held
                    * current_price
                    * (1 - self.transaction_fee_percent)
                )
                self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - prev_net_worth

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        obs = np.array(
            [
                self.balance,
                self.shares_held,
                self.df.iloc[self.current_step]["Close"],
                self.df.iloc[self.current_step]["RSI"],
                self.df.iloc[self.current_step]["MACD_Hist"],
            ]
        )
        return obs.astype(np.float32)


# --- 2. 학습 실행 ---
def train_agent(ticker, start_date, end_date):
    print(f"[{ticker}] 강화학습 에이전트 훈련을 시작합니다...")

    df = fdr.DataReader(ticker, start=start_date, end=end_date)
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=14).mean().fillna(0)
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean().fillna(0)
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd - signal
    df.dropna(inplace=True)

    env = StockTradingEnv(df)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=20000)

    save_path = f"models/{ticker}"
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/rl_agent")
    print(f"[{ticker}] 에이전트 훈련 완료 및 저장: {save_path}/rl_agent.zip")


if __name__ == "__main__":
    TICKERS_TO_TRAIN = ["005930", "TSLA", "AAPL", "NVDA"]
    for ticker in TICKERS_TO_TRAIN:
        train_agent(ticker, "2021-01-01", "2025-10-01")
