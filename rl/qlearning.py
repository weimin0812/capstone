import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Environment
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Hyperparameters
LR = 0.1
GAMMA = 0.99
EPS = 1.0
EPS_MIN = 0.01
EPS_DECAY = 0.001
EPISODES = 10000

q_table = np.zeros((n_states, n_actions))
reward_history = []  # Record reward for plotting

# Training
for ep in range(EPISODES):
    state, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:
        # Epsilon-greedy
        if np.random.random() < EPS:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)
        # Q update
        q_table[state, action] += LR * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
        ep_reward += reward

    # Decay epsilon
    EPS = EPS_MIN + (1.0 - EPS_MIN) * np.exp(-EPS_DECAY * ep)
    reward_history.append(ep_reward)

# Plot convergence
plt.figure(figsize=(10, 4))
plt.plot(reward_history, label="Episode Reward", alpha=0.7)
# Rolling average for smoother curve
window = 100
avg_reward = np.convolve(reward_history, np.ones(window)/window, mode="valid")
plt.plot(avg_reward, label=f"Average Reward (window={window})", linewidth=2)

plt.title("Q-Learning Convergence (FrozenLake)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

env.close()