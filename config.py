from __future__ import annotations

"""
All hyperparameters in one place for reproducibility and easy tuning.
"""

import dataclasses


@dataclasses.dataclass
class EnvConfig:
    env_id: str = "LunarLander-v3"
    seed: int = 42
    render: bool = False


@dataclasses.dataclass
class KalmanConfig:
    """
    Calibrated against LunarLander-v3 frame deltas:
        gravity  ≈ -0.027 Δvy / frame
        main eng ≈ +0.030 Δvy / frame (above gravity)
        side eng ≈ ±0.010 Δvx / frame
        torque   ≈ ±0.040 Δω  / frame
    """
    dt: float = 0.02
    process_noise_std: float = 0.05
    obs_noise_std: float = 0.02
    obs_noise_x: float = 0.05          # inflated x-noise → skeptical of small x jitter
    gravity: float = -0.027
    thrust_main: float = 0.030
    thrust_lateral: float = 0.010
    torque: float = 0.040


@dataclasses.dataclass
class UtilityWeights:
    w_vy: float = 6.0                # descent-speed tracking
    w_vx: float = 6.0                # horizontal velocity damping (overdamp oscillation)
    w_angle: float = 5.0             # upright orientation
    w_ang_vel: float = 5.0           # angular velocity damping
    w_x: float = 0.3                 # gentle position pull (low to avoid overshoot)
    descent_gain: float = 0.35       # target_vy = -descent_gain * y
    center_gain: float = 0.10        # target_vx = -center_gain * x (very gentle)
    fuel_main: float = 0.01
    fuel_side: float = 0.003
    braking_g_eff: float = 0.003     # net vy deceleration per frame from main engine
    braking_safety: float = 1.0      # fire exactly at physics limit
    safety_scale: float = 8.0
    safety_shift: float = 0.5


@dataclasses.dataclass
class TrainingConfig:
    num_episodes: int = 500
    max_steps: int = 1000
    log_every: int = 25
    safe_reward_threshold: float = 50.0


@dataclasses.dataclass
class MetricsConfig:
    calib_bins: int = 10
    pr_thresholds: int = 21


@dataclasses.dataclass
class ProjectConfig:
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    kalman: KalmanConfig = dataclasses.field(default_factory=KalmanConfig)
    utility: UtilityWeights = dataclasses.field(default_factory=UtilityWeights)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    metrics: MetricsConfig = dataclasses.field(default_factory=MetricsConfig)


CONFIG = ProjectConfig()
