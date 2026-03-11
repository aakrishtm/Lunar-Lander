from __future__ import annotations

"""
All hyperparameters in one place for reproducibility and easy tuning.
"""

import dataclasses


@dataclasses.dataclass
class ALEConfig:
    rom_name: str = "lunar_lander"
    seed: int = 42
    repeat_action_probability: float = 0.0
    display_screen: bool = False


@dataclasses.dataclass
class RAMAddresses:
    """Byte offsets into the 128-byte Atari 2600 RAM for Lunar Lander."""
    y_pos: int = 0xA9   # altitude (unsigned)
    y_vel: int = 0xA4   # vertical velocity (signed)
    x_vel: int = 0xA2   # horizontal velocity (signed)
    angle: int = 0x90   # orientation (signed, 0 = upright)


@dataclasses.dataclass
class KalmanConfig:
    dt: float = 1.0                 # one frame
    process_noise_std: float = 1.5  # Q diagonal
    obs_noise_std: float = 3.0      # R diagonal
    gravity: float = 0.5            # downward accel per frame (RAM-byte units)
    thrust_main: float = 1.2        # upward accel from main engine
    thrust_lateral: float = 0.4     # horizontal accel from side thrusters
    torque: float = 0.3             # angular change from side thrusters


@dataclasses.dataclass
class UtilityWeights:
    w_vy: float = 2.0       # penalise vertical speed
    w_vx: float = 1.0       # penalise horizontal speed
    w_angle: float = 1.5    # penalise tilt away from upright
    fuel_main: float = 0.30
    fuel_side: float = 0.03


@dataclasses.dataclass
class TrainingConfig:
    num_episodes: int = 500
    max_steps: int = 2000
    log_every: int = 25
    safe_reward_threshold: float = 0.0


@dataclasses.dataclass
class MetricsConfig:
    calib_bins: int = 10
    pr_thresholds: int = 21


@dataclasses.dataclass
class ProjectConfig:
    ale: ALEConfig = dataclasses.field(default_factory=ALEConfig)
    ram: RAMAddresses = dataclasses.field(default_factory=RAMAddresses)
    kalman: KalmanConfig = dataclasses.field(default_factory=KalmanConfig)
    utility: UtilityWeights = dataclasses.field(default_factory=UtilityWeights)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    metrics: MetricsConfig = dataclasses.field(default_factory=MetricsConfig)


CONFIG = ProjectConfig()
