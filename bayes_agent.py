from __future__ import annotations

"""
Bayesian Lunar Lander agent — Physics-First control law.

Architecture
------------
1. **KalmanFilter** maintains a smoothed Gaussian posterior over
   [y, y_dot, x_dot, angle] from noisy 8-bit RAM.
2. For each candidate action a ∈ {0,1,2,3} we run a *hypothetical*
   Kalman predict step to obtain the predicted next-state mean.
3. We evaluate a utility function on each predicted state and pick
   a* = argmax_a  U(a).
4. A sigmoid-transformed "danger" metric provides a continuous
   P(Safe) estimate used by the calibration / PR validation suite.
"""

import numpy as np

from config import CONFIG
from kalman_filter import KalmanFilter


class BayesianLanderAgent:

    NUM_ACTIONS = 4  # 0=NOOP, 1=Left, 2=Main, 3=Right

    def __init__(self) -> None:
        self.kf = KalmanFilter()
        self.w  = CONFIG.utility

    def reset(self, z0: np.ndarray) -> None:
        """Initialise the Kalman filter with the first RAM observation."""
        self.kf.reset(z0)

    def observe(self, action: int, z: np.ndarray) -> None:
        """Run one predict → update cycle on the real observation."""
        self.kf.predict(action)
        self.kf.update(z)

    # ================================================================== #
    #  Action selection  —  Expected Utility maximisation                 #
    # ================================================================== #
    def select_action(self) -> int:
        """
        For each action a:
            1. Hypothetically predict the next state.
            2. Compute U(a) = -w_vy|y_dot| - w_vx|x_dot| - w_angle|angle| - fuel.
            3. Return a* = argmax U(a).
        """
        best_action  = 0
        best_utility = -np.inf

        for a in range(self.NUM_ACTIONS):
            u = self._evaluate_action(a)
            if u > best_utility:
                best_utility = u
                best_action  = a

        return best_action

    def _evaluate_action(self, action: int) -> float:
        """Score a candidate action via the predicted next-state."""
        predicted = self.kf.predict_state(action)
        _, y_dot, x_dot, angle = predicted

        fuel = 0.0
        if action == 2:
            fuel = self.w.fuel_main
        elif action in (1, 3):
            fuel = self.w.fuel_side

        return (
            - self.w.w_vy    * abs(y_dot)
            - self.w.w_vx    * abs(x_dot)
            - self.w.w_angle * abs(angle)
            - fuel
        )

    # ================================================================== #
    #  Safety score  —  used by the validation / metrics layer            #
    # ================================================================== #
    def safety_score(self) -> float:
        """
        Map the current Kalman state to P(Safe) ∈ (0, 1) via sigmoid.

        danger = w_vy·|y_dot| + w_vx·|x_dot| + w_angle·|angle|
        P(Safe) = σ(shift − danger / scale)

        Calm states → high P(Safe),  violent states → low P(Safe).
        """
        _, y_dot, x_dot, angle = self.kf.state_mean

        danger = (
              self.w.w_vy    * abs(y_dot)
            + self.w.w_vx    * abs(x_dot)
            + self.w.w_angle * abs(angle)
        )

        scale, shift = 20.0, 2.0
        p_safe = 1.0 / (1.0 + np.exp(danger / scale - shift))
        return float(np.clip(p_safe, 0.001, 0.999))
