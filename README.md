# Bayesian Atari Lunar Lander (ALE)

A **Physics-First** autonomous agent for the raw Atari 2600 Lunar Lander ROM
using `ale-py`.  Bypasses Gymnasium and Deep RL to implement explicit Bayesian
inference (Kalman Filter) and Decision-Theoretic control.

## Architecture

| Layer | Purpose |
|---|---|
| **Kalman Filter** | Treats 8-bit RAM bytes as noisy observations; maintains a Gaussian posterior N(μ, P) over the continuous state `[y, y_dot, x_dot, angle]`. |
| **Heuristic Policy** | For each candidate action, runs a hypothetical Kalman predict and picks `a* = argmax U(a)` where `U = −w₁|ẏ| − w₂|ẋ| − w₃|θ| − fuel_cost`. |
| **Validation Suite** | Calibration diagram, Precision/Recall, Poisson modelling, Negative Binomial fuel fit. |

## Key Constraint

**No PyTorch, TensorFlow, or scikit-learn.**  All Bayesian updates, matrix math
(Kalman equations), and statistical models (Poisson PMF, Negative Binomial,
calibration, Brier score) are implemented from scratch with NumPy.

## Files

| File | Description |
|---|---|
| `config.py` | All hyperparameters (ALE settings, RAM addresses, Kalman noise, utility weights). |
| `env_wrapper.py` | Thin wrapper around `ALEInterface`; extracts RAM bytes to float observations. |
| `kalman_filter.py` | 4-D Kalman filter with Newtonian dynamics.  Explicit P(State \| Obs) update. |
| `bayes_agent.py` | Agent: Kalman state estimation + utility-based action selection. |
| `reward_model.py` | Reward decomposition (fuel cost, landing reward, crash penalty). |
| `fuel_model.py` | Negative Binomial fit via method-of-moments (from scratch). |
| `poisson_utils.py` | Poisson PMF and Knuth sampler for offline event-rate analysis. |
| `metrics.py` | Calibration, Brier score, ECE, Precision/Recall, EV-vs-return correlation. |
| `train.py` | Main loop: run episodes, log data, print metrics, save plots. |

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python train.py
```

The agent reads raw RAM from the Atari 2600 Lunar Lander ROM, smooths state
via Kalman filtering, and picks actions to maximise expected utility each frame.
