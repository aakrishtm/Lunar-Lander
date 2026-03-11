from __future__ import annotations

"""
Kalman Filter for Atari Lunar Lander state estimation.

Treats the 8-bit RAM bytes as noisy observations Z of an underlying
continuous physics state.  Maintains a Gaussian posterior belief
    N(mu, P)
and updates it every frame using the standard Kalman equations.

This IS the explicit Bayesian update  P(S | O):

    Prior      :  N(mu_pred, P_pred)       ← Prediction step (kinematics)
    Likelihood :  N(z | H·x, R)            ← Observation model (RAM reading)
    Posterior  :  N(mu_upd, P_upd)         ← Update step (Kalman gain)

    P(State | Observation) ∝ P(Observation | State) · P(State)
"""

import numpy as np

from config import CONFIG


class KalmanFilter:
    """
    4-D Kalman filter over state  x = [y, y_dot, x_dot, angle].

    All matrix operations are pure NumPy — no scipy or filterpy.
    """

    STATE_DIM = 4
    OBS_DIM = 4

    def __init__(self) -> None:
        cfg = CONFIG.kalman
        self.dt = cfg.dt

        # ---- Dynamics: x_{t+1} = A x_t + b(action) ----
        self.A = np.array([
            [1.0, cfg.dt, 0.0, 0.0],   # y += dt · y_dot
            [0.0, 1.0,    0.0, 0.0],   # y_dot (gravity + thrust in b)
            [0.0, 0.0,    1.0, 0.0],   # x_dot (thrust in b)
            [0.0, 0.0,    0.0, 1.0],   # angle  (torque in b)
        ], dtype=np.float64)

        g  = cfg.gravity
        tm = cfg.thrust_main
        tl = cfg.thrust_lateral
        tq = cfg.torque

        self._b = {
            0: np.array([0.0, -g,       0.0,  0.0]),   # nothing
            1: np.array([0.0, -g,      -tl,  -tq]),    # left thruster
            2: np.array([0.0, -g + tm,  0.0,  0.0]),   # main engine
            3: np.array([0.0, -g,       tl,   tq]),    # right thruster
        }

        # ---- Observation model: z = H x + noise ----
        self.H = np.eye(self.OBS_DIM, dtype=np.float64)

        # ---- Noise covariances ----
        self.Q = np.eye(self.STATE_DIM) * cfg.process_noise_std ** 2
        self.R = np.eye(self.OBS_DIM)   * cfg.obs_noise_std ** 2

        # ---- Belief state ----
        self.mu = np.zeros(self.STATE_DIM, dtype=np.float64)
        self.P  = np.eye(self.STATE_DIM, dtype=np.float64) * 100.0

    # ------------------------------------------------------------------ #
    #  Initialise                                                         #
    # ------------------------------------------------------------------ #
    def reset(self, z0: np.ndarray) -> None:
        """Set the initial belief to the first observation with high uncertainty."""
        self.mu = z0.copy().astype(np.float64)
        self.P  = np.eye(self.STATE_DIM, dtype=np.float64) * 50.0

    # ================================================================== #
    #  PREDICT  —  the PRIOR   P(x_t | z_{1:t-1}, a_{t-1})              #
    # ================================================================== #
    def predict(self, action: int) -> None:
        """
        Propagate the belief one step forward under Newtonian kinematics.

            mu_pred = A · mu  +  b(action)
            P_pred  = A · P · A^T  +  Q
        """
        b = self._b.get(action, self._b[0])
        self.mu = self.A @ self.mu + b
        self.P  = self.A @ self.P @ self.A.T + self.Q

    # ================================================================== #
    #  UPDATE  —  the POSTERIOR   P(x_t | z_{1:t})                       #
    #                                                                     #
    #  This is the Bayesian step:                                         #
    #      posterior ∝ likelihood  ×  prior                               #
    #  where                                                              #
    #      prior      = N(mu_pred, P_pred)                                #
    #      likelihood = N(z | H · x, R)                                   #
    #      posterior   = N(mu_upd,  P_upd)                                #
    # ================================================================== #
    def update(self, z: np.ndarray) -> None:
        """
        Fuse the RAM observation with the predicted prior.

        Innovation:          y = z  -  H · mu_pred
        Innovation cov:      S = H · P_pred · H^T  +  R
        Kalman Gain:         K = P_pred · H^T · S^{-1}
        Posterior mean:      mu = mu_pred  +  K · y
        Posterior cov:       P  = (I - K · H) · P_pred
        """
        z = np.asarray(z, dtype=np.float64)

        y = z - self.H @ self.mu                              # innovation
        S = self.H @ self.P @ self.H.T + self.R               # innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)              # Kalman gain

        self.mu = self.mu + K @ y                              # posterior mean
        I = np.eye(self.STATE_DIM)
        self.P  = (I - K @ self.H) @ self.P                   # posterior cov

    # ------------------------------------------------------------------ #
    #  Hypothetical predict (does NOT mutate state)                       #
    # ------------------------------------------------------------------ #
    def predict_state(self, action: int) -> np.ndarray:
        """Return predicted mean for a candidate action."""
        b = self._b.get(action, self._b[0])
        return self.A @ self.mu + b

    # ------------------------------------------------------------------ #
    #  Read-only accessors                                                #
    # ------------------------------------------------------------------ #
    @property
    def state_mean(self) -> np.ndarray:
        return self.mu.copy()

    @property
    def state_cov(self) -> np.ndarray:
        return self.P.copy()
