from __future__ import annotations

"""
Bayesian Lunar Lander agent — Physics-First control law.

1. KalmanFilter maintains a smoothed posterior over
   [x, y, vx, vy, angle, ang_vel] from Gymnasium observations.
2. For each candidate action we run a hypothetical Kalman predict
   and evaluate a utility function on the predicted next-state.
3. Pick a* = argmax_a U(a), with a descent-rate safety override.
"""

import numpy as np

from config import CONFIG
from kalman_filter import KalmanFilter


class BayesianLanderAgent:

    NUM_ACTIONS = 4  # 0=NOOP, 1=Left, 2=Main, 3=Right

    def __init__(self) -> None:
        self.kf = KalmanFilter()
        self.w  = CONFIG.utility

    def reset(self, obs: np.ndarray) -> None:
        self.kf.reset(obs[:6])

    def observe(self, action: int, obs: np.ndarray) -> None:
        self.kf.predict(action)
        self.kf.update(obs[:6])

    # ================================================================== #
    #  Action selection                                                   #
    # ================================================================== #
    def select_action(self) -> int:
        """
        If the lander is falling too fast, fire main engine immediately.
        Otherwise pick the action with highest predicted utility.
        """
        vy = self.kf.state_mean[3]

        if vy < self.w.vy_fire_threshold:
            return 2

        best_action  = 0
        best_utility = -np.inf

        for a in range(self.NUM_ACTIONS):
            u = self._evaluate_action(a)
            if u > best_utility:
                best_utility = u
                best_action  = a

        return best_action

    # ================================================================== #
    #  Utility function (altitude-aware descent profile)                   #
    #                                                                      #
    #  Target velocities scale with position so the lander follows a       #
    #  controlled descent toward the pad rather than hovering.             #
    #    target_vy = -descent_gain * y   (fall faster when high)           #
    #    target_vx = -center_gain  * x   (move toward pad center)          #
    # ================================================================== #
    def _evaluate_action(self, action: int) -> float:
        predicted = self.kf.predict_state(action)
        x, y, vx, vy, angle, ang_vel = predicted

        target_vy = -self.w.descent_gain * max(y, 0.0)
        target_vx = -self.w.center_gain * x

        fuel = 0.0
        if action == 2:
            fuel = self.w.fuel_main
        elif action in (1, 3):
            fuel = self.w.fuel_side

        return (
            - self.w.w_vy      * (vy - target_vy) ** 2
            - self.w.w_vx      * (vx - target_vx) ** 2
            - self.w.w_angle   * angle ** 2
            - self.w.w_ang_vel * ang_vel ** 2
            - self.w.w_x       * x ** 2
            - fuel
        )

    # ================================================================== #
    #  P(Safe) — crash-biased sigmoid                                    #
    # ================================================================== #
    def safety_score(self) -> float:
        x, _y, vx, vy, angle, _ang_vel = self.kf.state_mean

        danger = (
              self.w.w_vy    * abs(vy)
            + self.w.w_vx    * abs(vx)
            + self.w.w_angle * abs(angle)
            + self.w.w_x     * abs(x)
        )

        p_safe = 1.0 / (1.0 + np.exp(danger / self.w.safety_scale
                                       - self.w.safety_shift))
        return float(np.clip(p_safe, 0.001, 0.999))
