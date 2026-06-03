import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Config
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

LR = 0.001
MEM_SIZE = 100000
BATCH = 64
GAMMA = 0.99
EPS = 1.0
EPS_MIN = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
EPISODES = 500

# DQN Network
class DQN(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, a_dim)
    def forward(self, x):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
opt = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEM_SIZE)

reward_history = []

def select_action(state):
    global EPS
    if random.random() < EPS:
        return env.action_space.sample()
    s = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return policy_net(s).argmax().item()

def train_step():
    if len(memory) < BATCH:
        return
    s,a,r,ns,d = memory.sample(BATCH)
    s = torch.FloatTensor(s).to(device)
    a = torch.LongTensor(a).to(device)
    r = torch.FloatTensor(r).to(device)
    ns = torch.FloatTensor(ns).to(device)
    d = torch.FloatTensor(d).to(device)

    current_q = policy_net(s).gather(1, a.unsqueeze(1))
    next_q = target_net(ns).max(1)[0].detach()
    target_q = r + (1 - d) * GAMMA * next_q

    loss = nn.MSELoss()(current_q.squeeze(), target_q)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Training loop
for ep in range(EPISODES):
    state, _ = env.reset()
    total_r = 0
    done = False
    while not done:
        act = select_action(state)
        next_s, r, done, _, _ = env.step(act)
        memory.add(state, act, r, next_s, done)
        state = next_s
        total_r += r
        train_step()

    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if EPS > EPS_MIN:
        EPS *= EPS_DECAY

    reward_history.append(total_r)
    if (ep+1) % 50 == 0:
        print(f"Episode {ep+1:3d} | Reward: {total_r:3.0f} | Eps: {EPS:.3f}")

# Plot
plt.figure(figsize=(10,4))
plt.plot(reward_history, alpha=0.6, label="Raw Reward")
win = 20
avg = np.convolve(reward_history, np.ones(win)/win, mode="valid")
plt.plot(avg, linewidth=2, label=f"Average (window={win})")
plt.title("DQN Convergence (CartPole)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

env.close()