import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

class GRUQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.n_actions = int(n_actions)

        self.gru = nn.GRUCell(input_size=self.obs_dim, hidden_size=self.hidden_size)
        self.q_head = nn.Linear(self.hidden_size, self.n_actions)
        self.h0 = nn.Parameter(torch.zeros(self.hidden_size))
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        for n,p in self.gru.named_parameters():
            if 'weight' in n:
                nn.init.orthogonal_(p, gain=1.0)
            elif 'bias' in n:
                nn.init.constant_(p, 0.0)

    def init_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.h0.view(1, -1).expand(batch, -1).to(device)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_next = self.gru(obs, h)
        q = self.q_head(h_next)
        return q, h_next    

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, hidden_size: int) -> torch.Tensor:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size

        self.obs = np.zeros((capacity, self.obs_dim), dtype=np.float32)
        self.h = np.zeros((capacity, self.hidden_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, self.obs_dim), dtype=np.float32)
        self.next_h = np.zeros((capacity, self.hidden_size), dtype=np.float32)
        self.actions = np.zeros((capacity, ), dtype=np.int64)
        self.rewards = np.zeros((capacity, ), dtype=np.float32)
        self.dones = np.zeros((capacity, ), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, obs, h, actions, rewards, next_obs, next_h, dones):
        n = int(actions.shape[0])
        idxs = (np.arange(n) + self.pos) % self.capacity

        self.obs[idxs] = np.asarray(obs, dtype=np.float32)
        self.h[idxs] = np.asarray(h, dtype=np.float32)
        self.actions[idxs] = np.asarray(actions, dtype=np.int64)
        self.rewards[idxs] = np.asarray(rewards, dtype=np.float32)
        self.next_obs[idxs] = np.asarray(next_obs, dtype=np.float32)
        self.next_h[idxs] = np.asarray(next_h, dtype=np.float32)
        self.dones[idxs] = np.asarray(dones, dtype=np.float32)

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device:torch.device) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return (
            torch.from_numpy(self.obs[idx]).float().to(device),
            torch.from_numpy(self.h[idx]).float().to(device),
            torch.from_numpy(self.actions[idx]).long().to(device),
            torch.from_numpy(self.rewards[idx]).float().to(device),
            torch.from_numpy(self.next_obs[idx]).float().to(device),
            torch.from_numpy(self.next_h[idx]).float().to(device),
            torch.from_numpy(self.dones[idx]).float().to(device)
        )

class MLPQNetwork(nn.Module):
    def __init__(self, obs_dim: int, h1: int, h2: int, n_actions: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.net = nn.Sequential(
            nn.Linear(int(obs_dim), int(h1)),
            nn.ReLU(),
            nn.Linear(int(h1), int(h2)),
            nn.ReLU(),
            nn.Linear(int(h2), int(n_actions)),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)