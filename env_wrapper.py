from __future__ import annotations

"""
Thin wrapper around Gymnasium's LunarLander-v3.

No custom physics — we just relay observations, rewards, and done flags
exactly as Gymnasium provides them, plus a safe/crash label at episode end.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from config import CONFIG


@dataclass
class StepResult:
    obs: np.ndarray       # float64[8]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class GymLunarLander:
    """Standard Gymnasium LunarLander-v3 interface."""

    ACTIONS = [0, 1, 2, 3]  # NOOP, Left-engine, Main-engine, Right-engine

    def __init__(self) -> None:
        cfg = CONFIG.env
        render_mode = "human" if cfg.render else None
        self.env = gym.make(cfg.env_id, render_mode=render_mode)
        self._seeded = False

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment. Only the first call uses the configured seed."""
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        elif not self._seeded:
            obs, info = self.env.reset(seed=CONFIG.env.seed)
            self._seeded = True
        else:
            obs, info = self.env.reset()
        return np.asarray(obs, dtype=np.float64), info

    def step(self, action: int) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float64)

        if terminated or truncated:
            info = dict(info)
            info["safe_landing"] = self._is_safe(obs, reward)
            info["crash"] = not info["safe_landing"]

        return StepResult(
            obs=obs,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        self.env.close()

    @staticmethod
    def _is_safe(obs: np.ndarray, reward: float) -> bool:
        left_leg = obs[6] > 0.5
        right_leg = obs[7] > 0.5
        return bool(left_leg and right_leg and reward >= 0)
