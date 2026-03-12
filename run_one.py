import time

import gymnasium as gym

from bayes_agent import BayesianLanderAgent


def run_single_sim():
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = BayesianLanderAgent()

    obs, info = env.reset()
    agent.reset(obs)

    total_reward = 0.0
    terminated = False
    truncated = False

    print("--- Starting Single Simulation ---")
    while not (terminated or truncated):
        action = agent.select_action()
        obs, reward, terminated, truncated, info = env.step(action)
        agent.observe(action, obs)
        total_reward += reward
        time.sleep(0.01)

    print(f"Simulation Finished! Final Reward: {total_reward:.2f}")
    if total_reward >= 100:
        print("SUCCESSFUL LANDING")
    else:
        print("CRASH OR HARD LANDING")

    env.close()


if __name__ == "__main__":
    run_single_sim()
