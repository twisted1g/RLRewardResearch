from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .experiment import Experiment


@dataclass
class A2CConfig:
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    n_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cpu"
    seed: int = 42
    verbose: int = 1


def train_a2c(
    env,
    config: A2CConfig | None = None,
    experiment: Experiment | None = None,
) -> Dict[str, object]:
    if config is None:
        config = A2CConfig()

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
    model = A2C(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        device=config.device,
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


def load_a2c_agent(
    model_path: str | Path,
    env,
    vecnorm_path: str | Path | None = None,
) -> tuple[A2C, DummyVecEnv]:
    vec_env = DummyVecEnv([lambda: env])
    if vecnorm_path is not None:
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    model = A2C.load(str(model_path), env=vec_env)
    return model, vec_env
