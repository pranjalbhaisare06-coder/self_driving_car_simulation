"""
agent/dqn_agent.py
Deep Q-Network agent for autonomous driving.

Architecture: 3-layer MLP  |  Input → 128 → 64 → Actions
Optimizer   : Adam
Loss        : Huber (smooth L1) for stable training
"""

import os
import math
import random
import json
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .replay_buffer import ReplayBuffer


# ── Pure-Python fallback Q-network (numpy-free, no deps) ──────────────────────
class SimpleMLP:
    """Tiny hand-rolled MLP for environments without PyTorch."""

    def __init__(self, layer_sizes, lr=0.001):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            w = [[random.gauss(0, (2/fan_in)**0.5)
                  for _ in range(layer_sizes[i+1])]
                 for _ in range(fan_in)]
            b = [0.0] * layer_sizes[i+1]
            self.layers.append({"w": w, "b": b})
        self.lr = lr

    def _relu(self, x):
        return [max(0.0, v) for v in x]

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            out = []
            for j in range(len(layer["b"])):
                s = layer["b"][j] + sum(h[k] * layer["w"][k][j]
                                        for k in range(len(h)))
                out.append(s)
            # ReLU on all but last layer
            h = self._relu(out) if i < len(self.layers) - 1 else out
        return h

    def update(self, x, target_q, action):
        """Single-sample gradient step (simplified SGD)."""
        q = self.forward(x)
        err = q[action] - target_q
        # Back-prop through last layer only (simplified)
        h = x
        for i in range(len(self.layers) - 1):
            h = self._relu(self.layers[i]["w"][0])  # rough pass
        # Gradient update on output layer weights
        last = self.layers[-1]
        for k in range(len(last["w"])):
            last["w"][k][action] -= self.lr * err * (h[k] if k < len(h) else 0)
        last["b"][action] -= self.lr * err


# ── PyTorch network ────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )

        def forward(self, x):
            return self.net(x)


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Double-DQN agent with experience replay and target network.

    Parameters:
        state_size  : dimension of observation vector
        action_size : number of discrete actions
        lr          : learning rate (Adam)
        gamma       : discount factor
        epsilon     : initial exploration rate
        epsilon_min : floor on epsilon
        epsilon_decay: multiplicative decay per replay call
        batch_size  : mini-batch size for replay
        buffer_size : max transitions in replay buffer
        target_update: steps between target network syncs
    """

    def __init__(self, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.9995,
                 batch_size: int = 64, buffer_size: int = 50_000,
                 target_update: int = 200):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self._steps = 0

        self.memory = ReplayBuffer(capacity=buffer_size)

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = QNetwork(state_size, action_size).to(self.device)
            self.target_net = QNetwork(state_size, action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.loss_fn = nn.SmoothL1Loss()
            self._backend = "torch"
        else:
            self.policy_net = SimpleMLP([state_size, 128, 64, action_size], lr=lr)
            self._backend = "python"

    # ── Action selection ───────────────────────────────────────────────────────
    def act(self, state: list) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        if self._backend == "torch":
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.policy_net(t)
                return int(q.argmax().item())
        else:
            q = self.policy_net.forward(state)
            return q.index(max(q))

    # ── Memory ─────────────────────────────────────────────────────────────────
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # ── Learning ───────────────────────────────────────────────────────────────
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        if self._backend == "torch":
            s  = torch.FloatTensor(states).to(self.device)
            a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            r  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            ns = torch.FloatTensor(next_states).to(self.device)
            d  = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # Double-DQN target
            with torch.no_grad():
                best_actions = self.policy_net(ns).argmax(1, keepdim=True)
                target_q = r + self.gamma * (1 - d) * \
                           self.target_net(ns).gather(1, best_actions)

            current_q = self.policy_net(s).gather(1, a)
            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
        else:
            # Python fallback: single-sample updates
            for state, action, reward, next_state, done in batch:
                q_next = max(self.policy_net.forward(next_state))
                target = reward + (0 if done else self.gamma * q_next)
                self.policy_net.update(state, target, action)

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._steps += 1

        # Sync target network
        if self._backend == "torch" and self._steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if self._backend == "torch":
            torch.save({
                "policy": self.policy_net.state_dict(),
                "target": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "steps": self._steps
            }, path)
        else:
            with open(path + ".json", "w") as f:
                json.dump({"epsilon": self.epsilon, "steps": self._steps}, f)

    def load(self, path: str):
        if self._backend == "torch" and os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(ckpt["policy"])
            self.target_net.load_state_dict(ckpt["target"])
            self.epsilon = ckpt.get("epsilon", self.epsilon_min)
            self._steps = ckpt.get("steps", 0)

    def get_stats(self) -> dict:
        return {
            "backend": self._backend,
            "epsilon": round(self.epsilon, 4),
            "steps": self._steps,
            "memory_size": len(self.memory),
            "state_size": self.state_size,
            "action_size": self.action_size
        }