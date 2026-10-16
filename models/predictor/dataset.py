from torch.utils.data import Dataset
import os
import glob
import numpy as np
import torch
import math
from .feature_encoder import complex_to_gcs_tokens
from .train_config import AUGMENT_RANDOM_GLOBAL_PHASE, GAIN_SCALE, SAMPLES_PER_STEP

def find_x_npz(data_ready_root: str) -> str:
    cands = glob.glob(os.path.join(data_ready_root, "**", "X.npz"), recursive=True)
    if not cands:
        raise FileNotFoundError("No X.npz files found")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def find_latest_ckpt(ckpt_root: str) -> str:
    cands = glob.glob(os.path.join(ckpt_root, "**", "*.pt"), recursive=True)
    if not cands:
        raise FileNotFoundError(f"No .pt checkpoints found under: {os.path.abspath(ckpt_root)}")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def clamp_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, x)))

#Dataset
class XNPZWindows(Dataset):
    def __init__(self,
                 x_npz_path: str,
                 samples_per_step: int = SAMPLES_PER_STEP):
        self.path = x_npz_path
        self.samples_per_step = int(samples_per_step)
        if self.samples_per_step <= 0:
            raise ValueError(f"samples_per_step must be positive, got {self.samples_per_step}")

        self._npz = np.load(self.path, mmap_mode='r')
        if "X" not in self._npz:
            raise KeyError(f"'X' not found in {self.path}")
        self.X = self._npz["X"]  #(N,L,6) complex 64
        if self.X.ndim != 3 or self.X.shape[-1] != 6:
            raise ValueError(f"Expected X shape (N,L,6), got {self.X.shape}")

        L=self.X.shape[1]
        if (L % self.samples_per_step) != 0:
            raise ValueError(f"L={L} not divisible by samples_per_step={self.samples_per_step}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = np.array(self.X[idx], copy=False)
        return torch.from_numpy(x)

class PerRxSequenceDataset(Dataset):
    def __init__(self,
                 windows_ds: XNPZWindows,
                 window_indices: np.ndarray,
                 train: bool = False,
                 gain_scale: float = GAIN_SCALE):
        self.windows_ds = windows_ds
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.train = bool(train)
        self.gain_scale = float(gain_scale)

        if len(self.window_indices) == 0:
            raise ValueError("window_indices must not be empty")

        sample0 = self.windows_ds[int(self.window_indices[0])]
        L = int(sample0.shape[0])
        self.S = L // self.windows_ds.samples_per_step

    def __len__(self) -> int:
        return int(len(self.window_indices) * 6)

    def __getitem__(self, idx:int) -> torch.Tensor:
        w_local = idx // 6
        rx = idx % 6
        w_idx = int(self.window_indices[w_local])

        H = self.windows_ds[w_idx]  # (L,6) complex
        h = H[:, rx]                # (L,) complex
        
        if self.train and AUGMENT_RANDOM_GLOBAL_PHASE:
            phi = torch.rand((), dtype=torch.float32) * (2.0 * math.pi)
            phasor = torch.polar(torch.ones((), dtype=torch.float32), phi)  # unit phasor
            h = h * phasor.to(h.dtype)
        
        h_taps = h.reshape(self.S, self.windows_ds.samples_per_step)  # (S,TAPS) complex
        seq = complex_to_gcs_tokens(h_taps, gain_scale=self.gain_scale)  # (S,3*TAPS) float32
        return seq
        
        
