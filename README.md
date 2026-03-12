# Bayesian Lunar Lander (Gymnasium)

A **Physics-First** autonomous agent for Gymnasium's `LunarLander-v3`.
Bypasses Deep RL (PPO/DQN) to implement explicit Bayesian inference
(Kalman Filter) and Decision-Theoretic control.

## Architecture

| Layer | Purpose |
|---|---|
| **Kalman Filter** | Treats the 6 continuous observation fields as noisy sensors; maintains a Gaussian posterior N(mu, P) over `[x, y, vx, vy, angle, ang_vel]`. |
| **Heuristic Policy** | `U(a) = -10|vy| - 5|vx| - 10|angle| - 5|x| - fuel(a)`. Evaluated on the Kalman-predicted next state; picks `a* = argmax U(a)`. Fires main engine immediately when `vy < -0.2`. |
| **Crash-Biased Safety** | Maps Kalman state to P(Safe) via a sigmoid with a Laplace-style crash prior (Beta(3,1) equivalent). P(Safe) starts below 0.5 for most states. |
| **Validation Suite** | Calibration diagram, Brier/ECE, Precision/Recall, Poisson analysis, Negative Binomial fuel fit. |

## Key Constraint

**No PyTorch, TensorFlow, or scikit-learn.**  All Bayesian updates, matrix math
(Kalman equations), and statistical models (Poisson PMF, Negative Binomial,
calibration, Brier score) are implemented from scratch with NumPy.

## Prerequisites

- **Python 3.10 -- 3.12** (3.12 recommended)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
source venv/bin/activate
python train.py
```

### What happens

The agent runs **500 episodes** (configurable in `config.py`).
Every 25 episodes it prints a status line:

```
[Ep   25/500]  ret= -112.3  crash_rate=0.76  Brier=0.1832  ECE=0.0401  P@0.5=0.71  R@0.5=0.63  NB(r=14.2,p=0.038)
```

| Field | Meaning |
|---|---|
| `ret` | Total Gymnasium reward for that episode |
| `crash_rate` | Fraction of episodes so far that crashed |
| `Brier` | Brier score (lower is better) |
| `ECE` | Expected Calibration Error |
| `P@0.5 / R@0.5` | Precision and Recall at threshold tau = 0.5 |
| `NB(r, p)` | Negative Binomial fit to main-engine frame counts |

At the end it saves two plots:
- **`calibration.png`** -- reliability diagram
- **`precision_recall.png`** -- PR curve for safe-landing classification

### Watch the game visually

Set `render` to `True` in `config.py`:

```python
@dataclasses.dataclass
class EnvConfig:
    ...
    render: bool = True   # opens a Pygame window
```

## Tuning

All hyperparameters live in `config.py`:

| Parameter | Location | Effect |
|---|---|---|
| `w_vy = 10.0` | `UtilityWeights` | Descent speed penalty (high = brake hard) |
| `w_vx = 5.0` | `UtilityWeights` | Horizontal drift penalty |
| `w_angle = 10.0` | `UtilityWeights` | Tilt penalty (high = stay upright) |
| `w_x = 5.0` | `UtilityWeights` | Pad targeting (high = stay centred) |
| `fuel_main = 0.30` | `UtilityWeights` | Main engine cost per frame |
| `fuel_side = 0.03` | `UtilityWeights` | Side thruster cost per frame |
| `vy_fire_threshold = -0.2` | `UtilityWeights` | Fire main engine when vy below this |
| `safety_scale / safety_shift` | `UtilityWeights` | Crash-biased sigmoid parameters |
| `process_noise_std` | `KalmanConfig` | Higher = trust physics model less |
| `obs_noise_std` | `KalmanConfig` | Higher = trust observations less |

## Files

| File | Description |
|---|---|
| `config.py` | All hyperparameters. |
| `env_wrapper.py` | Thin wrapper around `gym.make("LunarLander-v3")`. |
| `kalman_filter.py` | 6-D Kalman filter with Newtonian dynamics. |
| `bayes_agent.py` | Agent: Kalman + utility + vy override + crash-biased P(Safe). |
| `reward_model.py` | Reward decomposition (fuel, landing, crash penalty). |
| `fuel_model.py` | Negative Binomial fit via method-of-moments. |
| `poisson_utils.py` | Poisson PMF and Knuth sampler (offline analysis). |
| `metrics.py` | Calibration, Brier, ECE, Precision/Recall, EV-vs-return. |
| `train.py` | Main loop: run episodes, log data, print metrics, save plots. |
