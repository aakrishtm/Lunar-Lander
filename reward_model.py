from __future__ import annotations

"""
Reward decomposition for the ALE Lunar Lander.

Breaks down per-episode economics into CS109-style random-variable
components (fuel cost, landing reward, crash penalty).  Used for
logging and analysis only — does NOT replace the ALE reward signal.
"""

from dataclasses import dataclass

from config import CONFIG


@dataclass
class RewardComponents:
    fuel_cost: float
    landing_reward: float
    crash_penalty: float

    @property
    def total(self) -> float:
        return self.fuel_cost + self.landing_reward + self.crash_penalty


def fuel_cost(main_frames: int, side_frames: int) -> float:
    """Cumulative fuel cost from frame counts."""
    w = CONFIG.utility
    return -(w.fuel_main * main_frames + w.fuel_side * side_frames)


def decompose_episode(
    total_reward: float,
    main_frames: int,
    side_frames: int,
    crashed: bool,
) -> RewardComponents:
    """Split an episode's total reward into interpretable components."""
    fc = fuel_cost(main_frames, side_frames)
    cp = -abs(total_reward) if crashed else 0.0
    lr = total_reward - fc - cp
    return RewardComponents(fuel_cost=fc, landing_reward=lr, crash_penalty=cp)
