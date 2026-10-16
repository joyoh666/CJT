from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


STATE_NAMES = ["RU1", "RU2", "RU3", "RU12", "RU13", "RU23"]
N_ACTIONS = 6

RU_ANTS: Dict[int, List[int]] = {
    1: [0, 1],
    2: [2, 3],
    3: [4, 5],
}

STATE_RUS: Dict[int, List[int]] = {
    0: [1],
    1: [2],
    2: [3],
    3: [1, 2],
    4: [1, 3],
    5: [2, 3],
}


class WindowStore:
    def __init__(self, window_dir: str):
        self.window_dir = Path(window_dir)
        x_path = self.window_dir / "X.npz"
        m_path = self.window_dir / "manifest.csv"
        if not x_path.is_file():
            raise FileNotFoundError(f"Missing: {x_path}")
        if not m_path.is_file():
            raise FileNotFoundError(f"Missing: {m_path}")

        self._npz = np.load(str(x_path), mmap_mode="r")
        if "X" not in self._npz:
            raise KeyError(f"'X' not found in {x_path}")
        self.X = self._npz["X"]

        if self.X.ndim != 3 or self.X.shape[-1] != 6:
            raise ValueError(f"Expected X shape (N,L,6), got {self.X.shape}")

        self.N = int(self.X.shape[0])
        self.L = int(self.X.shape[1])
        self.dt = float(self._npz["dt"]) if "dt" in self._npz else 0.005

        self.manifest = pd.read_csv(str(m_path))
        if "sample_id" not in self.manifest.columns:
            self.manifest["sample_id"] = np.arange(self.N, dtype=int)
        self.manifest = self.manifest.sort_values("sample_id").reset_index(drop=True)
        if len(self.manifest) != self.N:
            raise ValueError(f"manifest rows ({len(self.manifest)}) != X windows N ({self.N})")

    def __len__(self) -> int:
        return self.N

    def get_window_gains(self, idx: int) -> np.ndarray:
        H = np.array(self.X[int(idx)], copy=False)
        g = np.abs(H).astype(np.float32)
        bad = ~np.isfinite(g)
        if np.any(bad):
            g[bad] = 0.0
        return g


@dataclass
class EnvConfig:
    seg_len_ticks: int = 20
    obs_bin_ticks: int = 20
    obs_hist_bins: int = 3
    mask_value: float = 0.0
    handover_penalty: float = 5e-4
    inter_ru_phase_noise_std: float = math.pi / 8.0
    noise_power: float = 1e-3
    max_episode_steps: int = 0


class RUWindowEnv:
    def __init__(self, store: WindowStore, cfg: EnvConfig):
        self.store = store
        self.cfg = cfg

        self.obs_dim = 3 * self.cfg.obs_hist_bins + N_ACTIONS
        self.n_actions = N_ACTIONS

        self._g: Optional[np.ndarray] = None
        self._ep_idx: int = -1
        self._seg_k: int = 0
        self._n_segs: int = 0
        self._steps: int = 0
        self._max_steps_this_ep: int = 0
        self._conn_state: int = 0
        self._conn_hist: List[int] = []

    def reset(self, ep_idx: int) -> np.ndarray:
        g = self.store.get_window_gains(int(ep_idx))
        L = g.shape[0]
        segL = int(self.cfg.seg_len_ticks)
        if (L % segL) != 0:
            raise ValueError(f"Window length L={L} not divisible by seg_len_ticks={segL}")

        n_segs = L // segL
        if n_segs < 2:
            raise ValueError(f"Window too short: n_segs={n_segs} < 2")

        self._g = g
        self._ep_idx = int(ep_idx)
        self._seg_k = 0
        self._n_segs = int(n_segs)
        self._steps = 0
        self._conn_state = int(np.random.choice(list(range(N_ACTIONS))))
        self._conn_hist = [int(self._conn_state)]

        default_full = self._n_segs - 1
        if int(self.cfg.max_episode_steps) <= 0:
            self._max_steps_this_ep = default_full
        else:
            self._max_steps_this_ep = min(int(self.cfg.max_episode_steps), default_full)

        return self._make_obs()

    def _seg_slice(self, seg_idx: int) -> slice:
        segL = int(self.cfg.seg_len_ticks)
        a = seg_idx * segL
        b = a + segL
        return slice(a, b)

    def ru_gain_segment(self, seg_idx: int) -> np.ndarray:
        assert self._g is not None
        seg = self._g[self._seg_slice(seg_idx), :]
        g1 = np.sqrt(np.maximum(seg[:, 0] ** 2 + seg[:, 1] ** 2, 0.0)).mean()
        g2 = np.sqrt(np.maximum(seg[:, 2] ** 2 + seg[:, 3] ** 2, 0.0)).mean()
        g3 = np.sqrt(np.maximum(seg[:, 4] ** 2 + seg[:, 5] ** 2, 0.0)).mean()
        return np.array([g1, g2, g3], dtype=np.float32)

    def all_state_power_segment(self, seg_idx: int) -> np.ndarray:
        assert self._g is not None
        seg = self._g[self._seg_slice(seg_idx), :].astype(np.float64)

        p1_t = seg[:, 0] ** 2 + seg[:, 1] ** 2
        p2_t = seg[:, 2] ** 2 + seg[:, 3] ** 2
        p3_t = seg[:, 4] ** 2 + seg[:, 5] ** 2

        p1 = float(np.mean(p1_t))
        p2 = float(np.mean(p2_t))
        p3 = float(np.mean(p3_t))

        phase_std = float(self.cfg.inter_ru_phase_noise_std)

        def pair_power(pa_t: np.ndarray, pb_t: np.ndarray) -> float:
            a = np.sqrt(np.maximum(pa_t, 0.0) * 0.5)
            b = np.sqrt(np.maximum(pb_t, 0.0) * 0.5)
            dphi = np.random.normal(0.0, phase_std, size=a.shape)
            p = a * a + b * b + 2.0 * a * b * np.cos(dphi)
            return float(np.mean(p))

        p12 = pair_power(p1_t, p2_t)
        p13 = pair_power(p1_t, p3_t)
        p23 = pair_power(p2_t, p3_t)

        power = np.array([p1, p2, p3, p12, p13, p23], dtype=np.float32)
        return np.log2(1.0 + power / float(self.cfg.noise_power))

    def _make_obs(self) -> np.ndarray:
        B = int(self.cfg.obs_hist_bins)
        obs = np.full((3, B), float(self.cfg.mask_value), dtype=np.float32)

        for bi in range(B):
            seg_idx = int(self._seg_k) - (B - bi)
            seg_idx = max(0, min(seg_idx, self._n_segs - 1))
            full = self.ru_gain_segment(seg_idx).astype(np.float32)

            hist_idx = max(0, min(seg_idx, len(self._conn_hist) - 1))
            conn_state_seg = int(self._conn_hist[hist_idx])
            for ru in STATE_RUS[conn_state_seg]:
                obs[ru - 1, bi] = full[ru - 1]

        conn_oh = np.zeros((N_ACTIONS,), dtype=np.float32)
        conn_oh[int(self._conn_state)] = 1.0
        return np.concatenate([obs.reshape(-1).astype(np.float32), conn_oh], axis=0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self._g is not None
        action = int(np.clip(int(action), 0, N_ACTIONS - 1))

        obs_seg = int(self._seg_k)
        conn_before = int(self._conn_state)
        rew_seg = obs_seg + 1

        state_power = self.all_state_power_segment(rew_seg)
        handover_attempt = 1 if (int(action) != int(conn_before)) else 0
        reward = float(state_power[action]) - float(self.cfg.handover_penalty) * float(handover_attempt)

        self._conn_state = int(action)
        self._seg_k += 1
        self._steps += 1

        if len(self._conn_hist) <= int(self._seg_k):
            self._conn_hist.append(int(self._conn_state))

        done = self._steps >= self._max_steps_this_ep
        obs = self._make_obs()

        info = {
            "window_index": int(self._ep_idx),
            "seg_k": int(self._seg_k),
            "obs_seg": int(obs_seg),
            "reward_seg": int(rew_seg),
            "action": int(action),
            "reward": float(reward),
            "selected_state_power": float(state_power[action]),
            "best_state_power": float(np.max(state_power)),
            "is_best_action": int(action == int(np.argmax(state_power))),
            "handover_attempt": int(handover_attempt),
            "conn_state_before": int(conn_before),
            "conn_state_after": int(self._conn_state),
        }
        return obs, reward, done, info
