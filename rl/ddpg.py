import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Env: Pendulum (continuous action)
env = gym.make("Pendulum-v1")
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
max_act = float(env.action_space.high[0])

# Hyperparams
LR_ACT = 0.001
LR_CRIT = 0.002
GAMMA = 0.99
TAU = 0.005
MEM_SIZE = 100000
BATCH = 128
EPISODES = 200

# Actor & Critic
class Actor(nn.Module):
    def __init__(self, s_d, a_d, max_a):
        super().__init__()
        self.fc1 = nn.Linear(s_d, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, a_d)
        self.max_a = max_a
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.max_a

class Critic(nn.Module):
    def __init__(self, s_d, a_d):
        super().__init__()
        self.fc1 = nn.Linear(s_d + a_d, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
    def forward(self, s, a):
        x = torch.cat([s,a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)
    def add(self, s,a,r,ns,d):
        self.buf.append((s,a,r,ns,d))
    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s,a,r,ns,d = zip(*batch)
        return np.array(s),np.array(a),np.array(r),np.array(ns),np.array(d)
    def __len__(self):
        return len(self.buf)

# DDPG Agent
class DDPG:
    def __init__(self, s_d, a_d, max_a):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(s_d, a_d, max_a).to(self.device)
        self.actor_tar = Actor(s_d, a_d, max_a).to(self.device)
        self.actor_tar.load_state_dict(self.actor.state_dict())
        self.opt_act = optim.Adam(self.actor.parameters(), lr=LR_ACT)

        self.critic = Critic(s_d, a_d).to(self.device)
        self.critic_tar = Critic(s_d, a_d).to(self.device)
        self.critic_tar.load_state_dict(self.critic.state_dict())
        self.opt_crit = optim.Adam(self.critic.parameters(), lr=LR_CRIT)

        self.mem = ReplayBuffer(MEM_SIZE)

    def select_act(self, s):
        s = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        return self.actor(s).cpu().detach().numpy()[0]

    def train(self):
        if len(self.mem) < BATCH:
            return
        s,a,r,ns,d = self.mem.sample(BATCH)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # Update critic
        next_a = self.actor_tar(ns)
        tar_q = self.critic_tar(ns, next_a)
        tar_q = r + (1-d) * GAMMA * tar_q
        curr_q = self.critic(s, a)
        loss_crit = nn.MSELoss()(curr_q, tar_q.detach())
        self.opt_crit.zero_grad()
        loss_crit.backward()
        self.opt_crit.step()

        # Update actor
        loss_act = -self.critic(s, self.actor(s)).mean()
        self.opt_act.zero_grad()
        loss_act.backward()
        self.opt_act.step()

        # Soft update target networks
        for p, tp in zip(self.actor.parameters(), self.actor_tar.parameters()):
            tp.data.copy_(TAU * p.data + (1-TAU) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_tar.parameters()):
            tp.data.copy_(TAU * p.data + (1-TAU) * tp.data)

agent = DDPG(s_dim, a_dim, max_act)
reward_history = []

# Training
for ep in range(EPISODES):
    state, _ = env.reset()
    total_r = 0
    done = False
    while not done:
        act = agent.select_act(state)
        # Add exploration noise
        act = (act + np.random.normal(0, 0.1, size=a_dim)).clip(-max_act, max_act)
        next_s, r, done, _, _ = env.step(act)
        agent.mem.add(state, act, r, next_s, done)
        state = next_s
        total_r += r
        agent.train()

    reward_history.append(total_r)
    if (ep+1) % 20 == 0:
        print(f"Episode {ep+1:3d} | Reward: {total_r:.2f}")

# Plot
plt.figure(figsize=(10,4))
plt.plot(reward_history, alpha=0.6, label="Raw Reward")
win = 10
avg = np.convolve(reward_history, np.ones(win)/win, mode="valid")
plt.plot(avg, linewidth=2, label=f"Average (window={win})")
plt.title("DDPG Convergence (Pendulum)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

env.close()