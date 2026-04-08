from __future__ import annotations

from env.trading_env_baseline import MyTradingEnv
from env.trading_env_lstm import MyTradingEnvLSTM


class DrawdownAwareReward(MyTradingEnv):
    def __init__(self, *args, penalty_lambda: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.penalty_lambda = float(penalty_lambda)

    def _calculate_reward(self, done: bool) -> float:
        if self.prev_portfolio_value <= 0.0:
            return 0.0
        history = list(self.portfolio_history) + [float(self.portfolio_value)]
        peak = max(history) if history else float(self.portfolio_value)
        drawdown = (peak - self.portfolio_value) / peak if peak > 0 else 0.0
        step_return = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        return float(step_return - self.penalty_lambda * drawdown)


class DrawdownAwareRewardLSTM(MyTradingEnvLSTM):
    def __init__(self, *args, penalty_lambda: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.penalty_lambda = float(penalty_lambda)

    def _calculate_reward(self, done: bool) -> float:
        if self.prev_portfolio_value <= 0.0:
            return 0.0
        history = list(self.portfolio_history) + [float(self.portfolio_value)]
        peak = max(history) if history else float(self.portfolio_value)
        drawdown = (peak - self.portfolio_value) / peak if peak > 0 else 0.0
        step_return = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        return float(step_return - self.penalty_lambda * drawdown)
