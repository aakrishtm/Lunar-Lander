from __future__ import annotations

"""
Thin wrapper around ale-py's ALEInterface for the Atari 2600 Lunar Lander ROM.

Responsibilities
- Load the ROM and configure ALE settings.
- Each frame, extract the four physics-relevant RAM bytes.
- Convert raw 8-bit integers to continuous floats for the Kalman filter.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from ale_py import ALEInterface, roms

from config import CONFIG


@dataclass
class StepResult:
    obs: np.ndarray       # [y, y_dot, x_dot, angle]  float64
    reward: float
    game_over: bool
    ram: np.ndarray       # full 128-byte snapshot (uint8)


def _signed(val: int) -> float:
    """Interpret an unsigned byte as signed two's-complement (-128..127)."""
    return float(val) if val < 128 else float(val - 256)


class AtariLunarLander:
    """Direct ALE interface — no Gymnasium dependency."""

    ACTIONS = [0, 1, 2, 3]  # NOOP, Left, Main, Right

    def __init__(self) -> None:
        cfg = CONFIG.ale
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", cfg.seed)
        self.ale.setFloat("repeat_action_probability",
                          cfg.repeat_action_probability)
        self.ale.setBool("display_screen", cfg.display_screen)

        self.ale.loadROM(roms.get_rom_path(cfg.rom_name))

        self._actions = list(self.ale.getMinimalActionSet())
        while len(self._actions) < 4:
            self._actions.append(0)

        r = CONFIG.ram
        self._addrs = [r.y_pos, r.y_vel, r.x_vel, r.angle]

    # ------------------------------------------------------------------ #
    #  Environment interface                                              #
    # ------------------------------------------------------------------ #
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the game and return (observation, raw_ram)."""
        self.ale.reset_game()
        ram = self.ale.getRAM()
        return self._extract(ram), ram

    def step(self, action: int) -> StepResult:
        """Execute *action* for one frame and return the result."""
        ale_action = int(self._actions[action % len(self._actions)])
        reward = self.ale.act(ale_action)
        ram = self.ale.getRAM()
        return StepResult(
            obs=self._extract(ram),
            reward=float(reward),
            game_over=self.ale.game_over(),
            ram=ram,
        )

    def game_over(self) -> bool:
        return self.ale.game_over()

    def lives(self) -> int:
        return self.ale.lives()

    # ------------------------------------------------------------------ #
    #  RAM → float conversion                                            #
    # ------------------------------------------------------------------ #
    def _extract(self, ram: np.ndarray) -> np.ndarray:
        """
        Pull four bytes from RAM and convert to floats.

        y_pos  – unsigned  (higher = higher altitude)
        y_vel  – signed    (positive = ascending)
        x_vel  – signed    (positive = rightward)
        angle  – signed    (0 = upright)
        """
        y = float(ram[self._addrs[0]])
        y_dot = _signed(int(ram[self._addrs[1]]))
        x_dot = _signed(int(ram[self._addrs[2]]))
        angle = _signed(int(ram[self._addrs[3]]))
        return np.array([y, y_dot, x_dot, angle], dtype=np.float64)
