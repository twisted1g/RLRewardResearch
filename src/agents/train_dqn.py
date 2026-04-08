from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .experiment import Experiment


@dataclass
class DQNConfig:
    total_timesteps: int = 50_000
    learning_rate: float = 1e-3
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.2
    exploration_final_eps: float = 0.05
    seed: int = 42
    verbose: int = 1


def train_dqn(
    env,
    config: DQNConfig | None = None,
    experiment: Experiment | None = None,
) -> Dict[str, object]:
    if config is None:
        config = DQNConfig()

    if experiment is not None:
        env = Monitor(env, filename=str(experiment.dir / "monitor.csv"))
        experiment.save_config(config)

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config.gamma,
    )
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        train_freq=config.train_freq,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        seed=config.seed,
        verbose=config.verbose,
    )

    if experiment is not None:
        model.set_logger(experiment.setup_logger())
    model.learn(total_timesteps=config.total_timesteps)
    if experiment is not None:
        experiment.save_vecnormalize(vec_env)
        experiment.save_model(model)

    return {
        "model": model,
        "config": config,
    }


def load_dqn_agent(
    model_path: str | Path,
    env,
    vecnorm_path: str | Path | None = None,
) -> tuple[DQN, DummyVecEnv]:
    vec_env = DummyVecEnv([lambda: env])
    if vecnorm_path is not None:
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    model = DQN.load(str(model_path), env=vec_env)
    return model, vec_env
