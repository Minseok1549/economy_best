import datetime as dt  # datetime 라이브러리 추가
import os

import generate_forecasts
import generate_rl_signals
import train_models
import train_rl_agent

# --- 설정 ---
# 분석할 모든 종목 리스트
ALL_TICKERS = ["TSLA", "IONQ", "RGTI", "PLTR", "ALB", "RR"]

# 강화학습 모델 학습 기간
RL_TRAIN_START = "2021-01-01"

# 변경 전: RL_TRAIN_END = "2025-10-10"
# 변경 후: 오늘 날짜를 동적으로 가져오도록 수정
RL_TRAIN_END = dt.date.today().strftime("%Y-%m-%d")


def main():
    """로컬에서 모델 학습과 예측 생성을 모두 수행하는 메인 함수"""
    print("===== AI 투자 전략 어시턴트 로컬 배치 작업을 시작합니다. =====")
    print(f"데이터 기준일: {RL_TRAIN_END}")  # 오늘 날짜를 출력하여 확인

    # 1. 필요한 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/forecasts", exist_ok=True)
    os.makedirs("data/rl_signals", exist_ok=True)
    print("\n[단계 1/4] 작업 폴더 준비 완료.")

    # 2. 지도학습 모델 훈련
    print("\n[단계 2/4] 지도학습 모델 훈련을 시작합니다...")
    for ticker in ALL_TICKERS:
        train_models.train_and_save_models(ticker)
    print("지도학습 모델 훈련 완료.")

    # 3. 강화학습 에이전트 훈련
    print("\n[단계 3/4] 강화학습 에이전트 훈련을 시작합니다...")
    for ticker in ALL_TICKERS:
        train_rl_agent.train_agent(ticker, RL_TRAIN_START, RL_TRAIN_END)
    print("강화학습 에이전트 훈련 완료.")

    # 4. 예측 및 신호 생성
    print("\n[단계 4/4] 최신 데이터 기반 예측 및 신호 생성을 시작합니다...")
    for ticker in ALL_TICKERS:
        generate_forecasts.generate_forecasts_from_saved_models(ticker)
        generate_rl_signals.generate_signal(ticker)
    print("예측 및 신호 생성 완료.")

    print("\n===== 모든 로컬 배치 작업이 성공적으로 완료되었습니다. =====")
    print("이제 streamlit app을 실행하세요: streamlit run app.py")


if __name__ == "__main__":
    main()
