from __future__ import annotations

"""
Kalman Filter for LunarLander-v3 state estimation.

Treats the six continuous observation fields
    [x, y, vx, vy, angle, angular_vel]
as noisy sensor readings Z_t and maintains a Gaussian posterior belief

    N(mu_t, P_t)

updated every frame via the standard Kalman equations.

This IS the explicit Bayesian update  P(S | O):

    Prior      :  N(mu_pred, P_pred)       <- Prediction step (kinematics)
    Likelihood :  N(z | H x, R)            <- Observation model
    Posterior  :  N(mu_upd, P_upd)         <- Update step (Kalman gain)

    P(State | Observation)  ∝  P(Observation | State) · P(State)

All matrix math is pure NumPy — no scipy, no filterpy.
"""

import numpy as np

from config import CONFIG


class KalmanFilter:
    """
    6-D Kalman filter over state  x = [x, y, vx, vy, angle, ang_vel].
    """

    STATE_DIM = 6
    OBS_DIM = 6

    def __init__(self) -> None:
        cfg = CONFIG.kalman
        dt = cfg.dt

        # ---- Dynamics: x_{t+1} = A x_t + b(action) ----
        self.A = np.array([
            [1, 0, dt, 0,  0,  0 ],   # x  += dt · vx
            [0, 1, 0,  dt, 0,  0 ],   # y  += dt · vy
            [0, 0, 1,  0,  0,  0 ],   # vx  (modified by b)
            [0, 0, 0,  1,  0,  0 ],   # vy  (gravity + thrust in b)
            [0, 0, 0,  0,  1,  dt],   # θ  += dt · ω
            [0, 0, 0,  0,  0,  1 ],   # ω   (torque in b)
        ], dtype=np.float64)

        g  = cfg.gravity
        tm = cfg.thrust_main
        tl = cfg.thrust_lateral
        tq = cfg.torque

        # Control vectors calibrated against LunarLander-v3 frame deltas.
        # Action 1 = left orientation engine  → pushes vx negative, ang_vel positive
        # Action 3 = right orientation engine → pushes vx positive, ang_vel negative
        self._b = {
            0: np.array([0, 0,   0,  g,       0,   0 ]),   # nothing
            1: np.array([0, 0, -tl,  g,       0,  tq]),    # left engine
            2: np.array([0, 0,   0,  g + tm,  0,   0 ]),   # main engine
            3: np.array([0, 0,  tl,  g,       0, -tq]),    # right engine
        }

        # ---- Observation model: z = H x + noise ----
        self.H = np.eye(self.OBS_DIM, dtype=np.float64)

        # ---- Noise covariances ----
        self.Q = np.eye(self.STATE_DIM) * cfg.process_noise_std ** 2
        self.R = np.eye(self.OBS_DIM)   * cfg.obs_noise_std ** 2

        # ---- Belief state ----
        self.mu = np.zeros(self.STATE_DIM, dtype=np.float64)
        self.P  = np.eye(self.STATE_DIM, dtype=np.float64) * 1.0

    # ------------------------------------------------------------------ #
    def reset(self, z0: np.ndarray) -> None:
        """Initialise belief from the first observation."""
        self.mu = z0.copy().astype(np.float64)
        self.P  = np.eye(self.STATE_DIM, dtype=np.float64) * 0.1

    # ================================================================== #
    #  PREDICT  —  the PRIOR   P(x_t | z_{1:t-1}, a_{t-1})              #
    # ================================================================== #
    def predict(self, action: int) -> None:
        """
        Propagate belief one step forward via Newtonian kinematics.

            mu_pred = A · mu  +  b(action)
            P_pred  = A · P · A^T  +  Q
        """
        b = self._b.get(action, self._b[0])
        self.mu = self.A @ self.mu + b
        self.P  = self.A @ self.P @ self.A.T + self.Q

    # ================================================================== #
    #  UPDATE  —  the POSTERIOR   P(x_t | z_{1:t})                       #
    #                                                                     #
    #      posterior ∝ likelihood  ×  prior                               #
    # ================================================================== #
    def update(self, z: np.ndarray) -> None:
        """
        Fuse the Gymnasium observation with the predicted prior.

        Innovation:      y = z  −  H · mu_pred
        Innovation cov:  S = H · P_pred · H^T  +  R
        Kalman Gain:     K = P_pred · H^T · S^{-1}
        Posterior mean:  mu = mu_pred  +  K · y
        Posterior cov:   P  = (I − K · H) · P_pred
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
    @property
    def state_mean(self) -> np.ndarray:
        return self.mu.copy()

    @property
    def state_cov(self) -> np.ndarray:
        return self.P.copy()
