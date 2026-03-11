from __future__ import annotations

"""
Main training / data-collection loop for the Bayesian Atari Lunar Lander.

1. Creates the ALE environment and Bayesian agent (Kalman + utility).
2. Runs episodes: agent picks actions via expected-utility maximisation,
   Kalman filter smooths the noisy RAM observations every frame.
3. Logs per-step crash probabilities and per-episode fuel usage / returns.
4. Periodically computes and prints calibration, Brier, ECE, and PR metrics.
5. At termination, fits a Negative Binomial to fuel counts and saves plots.
"""

from typing import List

import numpy as np

from bayes_agent import BayesianLanderAgent
from config import CONFIG
from env_wrapper import AtariLunarLander
from fuel_model import fit_negative_binomial, prob_fuel_depleted
from metrics import (
    brier_score,
    compare_ev_vs_return,
    compute_calibration_bins,
    expected_calibration_error,
    plot_precision_recall,
    plot_reliability_diagram,
    precision_recall_curve,
)


def run() -> None:
    cfg = CONFIG.training

    env   = AtariLunarLander()
    agent = BayesianLanderAgent()

    # ---- accumulators ----
    all_crash_probs:    List[float] = []
    all_crash_outcomes: List[int]   = []
    steps_per_episode:  List[int]   = []
    episode_returns:    List[float] = []
    episode_mean_evs:   List[float] = []
    fuel_counts:        List[int]   = []

    for ep in range(cfg.num_episodes):
        obs, _ = env.reset()
        agent.reset(obs)

        total_reward  = 0.0
        main_frames   = 0
        side_frames   = 0
        step_count    = 0
        step_utils: List[float] = []

        for _ in range(cfg.max_steps):
            # Record predicted P(Crash) for this step
            p_crash = 1.0 - agent.safety_score()
            all_crash_probs.append(p_crash)

            action = agent.select_action()
            result = env.step(action)

            # Kalman predict + update on the new observation
            agent.observe(action, result.obs)

            total_reward += result.reward
            step_count   += 1

            if action == 2:
                main_frames += 1
            elif action in (1, 3):
                side_frames += 1

            step_utils.append(agent._evaluate_action(action))

            if result.game_over:
                break

        # ---- end of episode bookkeeping ----
        crashed = total_reward < cfg.safe_reward_threshold
        all_crash_outcomes.append(1 if crashed else 0)
        steps_per_episode.append(step_count)
        episode_returns.append(total_reward)
        episode_mean_evs.append(
            float(np.mean(step_utils)) if step_utils else 0.0
        )
        fuel_counts.append(main_frames)

        # ---- periodic console report ----
        if (ep + 1) % cfg.log_every == 0 and all_crash_probs:
            step_outcomes = _expand_outcomes(all_crash_outcomes,
                                            steps_per_episode)
            preds = np.array(all_crash_probs)
            outs  = step_outcomes

            bins  = compute_calibration_bins(preds, outs)
            brier = brier_score(preds, outs)
            ece   = expected_calibration_error(bins)
            _, precs, recs = precision_recall_curve(preds, outs)
            mid = len(precs) // 2

            r_nb, p_nb = fit_negative_binomial(np.array(fuel_counts))

            print(
                f"[Ep {ep+1:>4d}/{cfg.num_episodes}]  "
                f"ret={total_reward:>7.1f}  "
                f"crash_rate={np.mean(all_crash_outcomes):.2f}  "
                f"Brier={brier:.4f}  ECE={ece:.4f}  "
                f"P@0.5={precs[mid]:.2f}  R@0.5={recs[mid]:.2f}  "
                f"NB(r={r_nb:.1f},p={p_nb:.3f})"
            )

    # ================================================================== #
    #  Final reporting + plots                                            #
    # ================================================================== #
    if all_crash_probs:
        step_outcomes = _expand_outcomes(all_crash_outcomes, steps_per_episode)
        preds = np.array(all_crash_probs)

        bins = compute_calibration_bins(preds, step_outcomes)
        plot_reliability_diagram(bins)

        _, precs, recs = precision_recall_curve(preds, step_outcomes)
        plot_precision_recall(precs, recs)

        corr = compare_ev_vs_return(
            np.array(episode_mean_evs), np.array(episode_returns)
        )
        print(f"\nEV vs Return correlation: {corr:.3f}")

    if fuel_counts:
        r_nb, p_nb = fit_negative_binomial(np.array(fuel_counts))
        p_dep = prob_fuel_depleted(r_nb, p_nb, budget=500)
        print(f"Fuel NegBin fit:  r={r_nb:.2f}  p={p_nb:.4f}")
        print(f"P(fuel > 500 frames) = {p_dep:.4f}")

    print("Done.")


# ------------------------------------------------------------------ #
#  Helper: expand per-episode outcomes to per-step arrays             #
# ------------------------------------------------------------------ #
def _expand_outcomes(
    episode_outcomes: List[int],
    steps_per_episode: List[int],
) -> np.ndarray:
    """Replicate each episode's crash label across all its steps."""
    parts = [
        np.full(n, outcome, dtype=np.float64)
        for outcome, n in zip(episode_outcomes, steps_per_episode)
    ]
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


if __name__ == "__main__":
    run()
