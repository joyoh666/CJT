import torch
from .gru_predictor import SimpleChannelRNN, build_model_from_ckpt
from .feature_encoder import postprocess_pred_tokens, tokens_to_complex, complex_to_gcs_tokens, phase_mae_deg_from_tokens
from .dataset import clamp_int, XNPZWindows, PerRxSequenceDataset, find_latest_ckpt, find_x_npz
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
import math

from .test_config import (KF_R_REL,
                          KF_Q_REL,
                            DEVICE, 
                            PRINT_MICRO, 
                            BATCH_SIZE, 
                            EXAMPLE_T0, 
                            EXAMPLE_TEST_INDEX, 
                            CKPT_EXPLICIT,
                            CKPT_ROOT, 
                            X_NPZ_EXPLICIT, 
                            MAX_TEST_BATCHES, 
                            MICRO_SHOW_CONTEXT,
                            NUM_WORKERS,
                            DATA_READY_ROOT)

@torch.no_grad()
def model_kstep_predict_batch(
    model: SimpleChannelRNN,
    seq: torch.Tensor,            # (B,S,D) tokens (GT)
    t0: int,
    K: int,
    taps: int,
    gain_scale: float,
    eps_norm: float,
    max_age_feature: int,
    detach_pred_input: bool,
) -> torch.Tensor:
    """
    Returns pred_tokens: (B,K,D) postprocessed.
    Warmup with GT[0..t0-1] using age=0, then:
      x = GT[t0], age=0 -> pred1
      x = pred1,  age=1 -> pred2
      ...
    """
    B, S, D = seq.shape
    assert t0 >= 0 and (t0 + K) < S

    # init input (age=0)
    init_tok = model.init_tok.view(1, 1, D).expand(B, 1, D)
    if model.use_age_feature:
        init_in = torch.cat([init_tok, torch.zeros((B, 1, 1), device=seq.device, dtype=seq.dtype)], dim=-1)
    else:
        init_in = init_tok

    # prefix warmup: GT[0..t0-1], age=0
    if t0 > 0:
        prefix_tok = seq[:, :t0, :]
        if model.use_age_feature:
            prefix_in = torch.cat([prefix_tok, torch.zeros((B, t0, 1), device=seq.device, dtype=seq.dtype)], dim=-1)
        else:
            prefix_in = prefix_tok
        prefix_full = torch.cat([init_in, prefix_in], dim=1)
    else:
        prefix_full = init_in

    _, h = model.rnn(prefix_full)

    x = seq[:, t0, :]
    preds = []
    for i in range(K):
        if model.use_age_feature:
            age = torch.full((B,), i, device=seq.device, dtype=torch.long).clamp(0, max_age_feature).to(seq.dtype)
            age_norm = (age / float(max_age_feature)).unsqueeze(-1)  # (B,1)
        else:
            age_norm = None

        out, h = model.rnn_step(x, age_norm, h)
        pred = postprocess_pred_tokens(model.out_proj(out), taps=taps, eps_norm=eps_norm)
        preds.append(pred)

        if i != (K - 1):
            x = pred.detach() if detach_pred_input else pred

    return torch.stack(preds, dim=1)  # (B,K,D)

# -------------------------
# Baseline: Linear regression on ONE GT step (taps samples) -> predict K*taps samples
# -------------------------
@torch.no_grad()
def lr_predict_future_complex_from_gt0(
    gt0_complex: torch.Tensor,    # (B,taps) complex
    future_len: int,              # K*taps
) -> torch.Tensor:
    """
    Fit y(n)=a*n+b on n=0..taps-1 using least squares, separately for real/imag.
    Predict y(n) for n=taps..taps+future_len-1
    Returns (B,future_len) complex
    """
    B, taps = gt0_complex.shape
    device = gt0_complex.device
    dtype = torch.float32

    n = torch.arange(taps, device=device, dtype=dtype)  # (taps,)
    n_mean = n.mean()
    n_var = ((n - n_mean) ** 2).mean().clamp_min(1e-12)

    y_r = gt0_complex.real.to(dtype)  # (B,taps)
    y_i = gt0_complex.imag.to(dtype)

    y_r_mean = y_r.mean(dim=1, keepdim=True)
    y_i_mean = y_i.mean(dim=1, keepdim=True)

    cov_r = ((n.view(1, -1) - n_mean) * (y_r - y_r_mean)).mean(dim=1, keepdim=True)  # (B,1)
    cov_i = ((n.view(1, -1) - n_mean) * (y_i - y_i_mean)).mean(dim=1, keepdim=True)

    a_r = cov_r / n_var
    a_i = cov_i / n_var
    b_r = y_r_mean - a_r * n_mean
    b_i = y_i_mean - a_i * n_mean

    n_f = torch.arange(taps, taps + future_len, device=device, dtype=dtype).view(1, -1)  # (1,future_len)
    pred_r = a_r * n_f + b_r  # (B,future_len)
    pred_i = a_i * n_f + b_i

    return torch.complex(pred_r, pred_i)

# -------------------------
# Baseline: Kalman filter constant-velocity on ONE GT step (taps samples)
# -------------------------
@torch.no_grad()
def kf_cv_predict_batch_1d(
    y_obs: torch.Tensor,   # (B,taps) float32/float64
    future_len: int,
    r_rel: float,
    q_rel: float,
    dt: float = 1.0,
) -> torch.Tensor:
    """
    Batched constant-velocity Kalman filter:
      state = [pos, vel]
      F = [[1,dt],[0,1]]
      H = [1,0]
      Q = q * [[dt^4/4, dt^3/2],[dt^3/2, dt^2]]   (white accel)
      R = r
    r and q are set per-sample from var(y_obs): r=r_rel*var, q=q_rel*var

    Returns predicted positions for the next future_len steps AFTER the last observation:
      (B,future_len)
    """
    B, taps = y_obs.shape
    device = y_obs.device
    dtype = y_obs.dtype

    var = y_obs.var(dim=1, unbiased=False).clamp_min(1e-12)  # (B,)
    r = (float(r_rel) * var).clamp_min(1e-12)                # (B,)
    q = (float(q_rel) * var).clamp_min(1e-12)                # (B,)

    # precompute Q elems (per-sample)
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    q11 = q * (dt4 / 4.0)
    q12 = q * (dt3 / 2.0)
    q22 = q * (dt2)

    # state x: (B,2)
    x = torch.zeros((B, 2), device=device, dtype=dtype)
    x[:, 0] = y_obs[:, 0]
    x[:, 1] = 0.0

    # covariance P: start large
    P = torch.zeros((B, 2, 2), device=device, dtype=dtype)
    P[:, 0, 0] = 1.0
    P[:, 1, 1] = 1.0

    # constants
    F11, F12, F21, F22 = 1.0, dt, 0.0, 1.0
    I11, I22 = 1.0, 1.0

    # filter over observations
    for k in range(taps):
        # predict: x = F x
        x0 = F11 * x[:, 0] + F12 * x[:, 1]
        x1 = F21 * x[:, 0] + F22 * x[:, 1]
        x = torch.stack([x0, x1], dim=1)

        # predict: P = F P F^T + Q
        # P elements
        p00 = P[:, 0, 0]
        p01 = P[:, 0, 1]
        p10 = P[:, 1, 0]
        p11 = P[:, 1, 1]

        # FP
        fp00 = F11 * p00 + F12 * p10
        fp01 = F11 * p01 + F12 * p11
        fp10 = F21 * p00 + F22 * p10
        fp11 = F21 * p01 + F22 * p11

        # FPF^T
        p00n = fp00 * F11 + fp01 * F12
        p01n = fp00 * F21 + fp01 * F22
        p10n = fp10 * F11 + fp11 * F12
        p11n = fp10 * F21 + fp11 * F22

        # +Q
        p00n = p00n + q11
        p01n = p01n + q12
        p10n = p10n + q12
        p11n = p11n + q22

        P = torch.stack([torch.stack([p00n, p01n], dim=1),
                         torch.stack([p10n, p11n], dim=1)], dim=1)

        # update with measurement y_obs[:,k]
        yk = y_obs[:, k]

        # innovation: z - Hx = y - x_pos
        innov = yk - x[:, 0]

        # S = H P H^T + R = P00 + R
        S = P[:, 0, 0] + r

        # Kalman gain K = P H^T / S = [P00, P10]^T / S
        K0 = P[:, 0, 0] / S
        K1 = P[:, 1, 0] / S

        # state update
        x[:, 0] = x[:, 0] + K0 * innov
        x[:, 1] = x[:, 1] + K1 * innov

        # covariance update: P = (I - K H) P
        # (I - K H) = [[1-K0, 0],[-K1,1]]
        p00 = (I11 - K0) * P[:, 0, 0]
        p01 = (I11 - K0) * P[:, 0, 1]
        p10 = (-K1) * P[:, 0, 0] + I22 * P[:, 1, 0]
        p11 = (-K1) * P[:, 0, 1] + I22 * P[:, 1, 1]
        P = torch.stack([torch.stack([p00, p01], dim=1),
                         torch.stack([p10, p11], dim=1)], dim=1)

    # now predict future_len steps without updates
    preds = []
    for _ in range(future_len):
        x0 = F11 * x[:, 0] + F12 * x[:, 1]
        x1 = F21 * x[:, 0] + F22 * x[:, 1]
        x = torch.stack([x0, x1], dim=1)
        preds.append(x[:, 0])

    return torch.stack(preds, dim=1)  # (B,future_len)

@torch.no_grad()
def kf_predict_future_complex_from_gt0(
    gt0_complex: torch.Tensor,    # (B,taps) complex
    future_len: int,
    r_rel: float,
    q_rel: float,
) -> torch.Tensor:
    """
    Run KF on real and imag separately, then combine to complex.
    Returns (B,future_len) complex.
    """
    y_r = gt0_complex.real.to(torch.float32)
    y_i = gt0_complex.imag.to(torch.float32)

    pr_r = kf_cv_predict_batch_1d(y_r, future_len=future_len, r_rel=r_rel, q_rel=q_rel, dt=1.0)
    pr_i = kf_cv_predict_batch_1d(y_i, future_len=future_len, r_rel=r_rel, q_rel=q_rel, dt=1.0)
    return torch.complex(pr_r, pr_i)

def make_step_weights(K: int, gamma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w = torch.tensor([float(gamma) ** i for i in range(K)], device=device, dtype=dtype)
    return w

@torch.no_grad()
def eval_all_methods(
    model: SimpleChannelRNN,
    loader: DataLoader,
    device: torch.device,
    taps: int,
    gain_scale: float,
    eps_norm: float,
    K: int,
    gamma: float,
    max_age_feature: int,
    detach_pred_input: bool,
    max_batches: int = 0
) -> Dict[str, Dict]:

    model.eval()

    w = make_step_weights(K, gamma, device, torch.float32)
    w_sum = w.sum().item()

    methods = ["MODEL", "LR", "KF"]
    acc = {m: {
            "tok_mse_w_sum": 0.0,
            "cmse_w_sum": 0.0,
            "den_w_sum": 0.0,
            "phase_mae_w_sum": 0.0,
            "per_step_nmse_sum": np.zeros((K,), dtype=np.float64),
            "n": 0
        } for m in methods}

    for bi, seq in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        seq = seq.to(device, non_blocking=True)
        B, S, D = seq.shape
        if S < (K + 1):
            continue

        t0 = int(torch.randint(low=0, high=(S - K), size=(1,), device=device).item())

        gt_future_tok = seq[:, (t0+1):(t0+1+K), :]
        gt_future_c = tokens_to_complex(gt_future_tok, taps=taps, gain_scale=gain_scale)
        gt_future_c_flat = gt_future_c.reshape(B, K*taps)

        gt0_tok = seq[:, t0, :]
        gt0_c = tokens_to_complex(gt0_tok, taps=taps, gain_scale=gain_scale)

        pred_model_tok = model_kstep_predict_batch(
            model=model,
            seq=seq,
            t0=t0,
            K=K,
            taps=taps,
            gain_scale=gain_scale,
            eps_norm=eps_norm,
            max_age_feature=max_age_feature,
            detach_pred_input=detach_pred_input,
        ) 

        pred_model_c = tokens_to_complex(pred_model_tok, taps=taps, gain_scale=gain_scale)  # (B,K,taps)
        pred_model_c_flat = pred_model_c.reshape(B, K * taps)

         # ---- LR baseline ----
        lr_c_flat = lr_predict_future_complex_from_gt0(gt0_c, future_len=K * taps)  # (B,K*taps)
        lr_c = lr_c_flat.reshape(B, K, taps)
        lr_tok = complex_to_gcs_tokens(lr_c.reshape(B * K, taps), gain_scale=gain_scale).reshape(B, K, 3 * taps)
        
        # ---- KF baseline ----
        kf_c_flat = kf_predict_future_complex_from_gt0(gt0_c, future_len=K * taps, r_rel=KF_R_REL, q_rel=KF_Q_REL)
        kf_c = kf_c_flat.reshape(B, K, taps)
        kf_tok = complex_to_gcs_tokens(kf_c.reshape(B * K, taps), gain_scale=gain_scale).reshape(B, K, 3 * taps)

        # common function: accumulate
        def accumulate(method: str, pred_tok: torch.Tensor, pred_c_flat: torch.Tensor):
            # token MSE per step (B,K)
            tok_mse_k = ((pred_tok - gt_future_tok) ** 2).mean(dim=2)  # (B,K)
            tok_mse_w = (tok_mse_k * w.view(1, K)).sum(dim=1).mean().item() / max(w_sum, 1e-12)

            # complex MSE per step (B,K) and denom
            pred_c = pred_c_flat.reshape(B, K, taps)
            err2_k = ((pred_c - gt_future_c).abs() ** 2).mean(dim=2)     # (B,K)
            den_k  = ((gt_future_c).abs() ** 2).mean(dim=2)              # (B,K)

            cmse_w = (err2_k * w.view(1, K)).sum(dim=1).mean().item() / max(w_sum, 1e-12)
            den_w  = (den_k  * w.view(1, K)).sum(dim=1).mean().item() / max(w_sum, 1e-12)

            # phase MAE (token-based circular)
            ph_mae_k = []
            for k in range(K):
                ph_mae_k.append(phase_mae_deg_from_tokens(pred_tok[:, k, :], gt_future_tok[:, k, :], taps=taps))
            ph_mae_k = torch.tensor(ph_mae_k, device=device, dtype=torch.float32)  # (K,)
            ph_w = (ph_mae_k * w).sum().item() / max(w_sum, 1e-12)

            # per-step NMSE (complex)
            nmse_k = (err2_k.mean(dim=0) / den_k.mean(dim=0).clamp_min(1e-12)).detach().cpu().numpy()  # (K,)

            acc[method]["tok_mse_w_sum"] += tok_mse_w
            acc[method]["cmse_w_sum"] += cmse_w
            acc[method]["den_w_sum"] += den_w
            acc[method]["phase_mae_w_sum"] += ph_w
            acc[method]["per_step_nmse_sum"] += nmse_k
            acc[method]["n"] += 1   

        accumulate("MODEL", pred_model_tok, pred_model_c_flat)
        accumulate("LR",    lr_tok,         lr_c_flat)
        accumulate("KF",    kf_tok,         kf_c_flat)

    # finalize
    out = {}
    for m in methods:
        n = acc[m]["n"]
        if n == 0:
            out[m] = {
                "tok_mse_w": float("nan"),
                "complex_mse_w": float("nan"),
                "nmse_w": float("nan"),
                "phase_mae_deg_w": float("nan"),
                "per_step_nmse": [float("nan")] * K,
                "n_batches": 0
            }
        else:
            tok = acc[m]["tok_mse_w_sum"] / n
            cmse = acc[m]["cmse_w_sum"] / n
            den = acc[m]["den_w_sum"] / n
            nmse = cmse / max(den, 1e-12)
            ph = acc[m]["phase_mae_w_sum"] / n
            per_step_nmse = (acc[m]["per_step_nmse_sum"] / n).tolist()
            out[m] = {
                "tok_mse_w": tok,
                "complex_mse_w": cmse,
                "nmse_w": nmse,
                "phase_mae_deg_w": ph,
                "per_step_nmse": per_step_nmse,
                "n_batches": n
            }
    return out

def _fmt(x: Optional[float], w: int, p: int) -> str:
    if x is None:
        return " " * w
    return f"{x:{w}.{p}f}"

@torch.no_grad()
def micro_predict_all(
    model: SimpleChannelRNN,
    seq1: torch.Tensor,   # (S,D) CPU tokens
    t0: int,
    device: torch.device,
    taps: int,
    gain_scale: float,
    eps_norm: float,
    K: int,
    gamma: float,
    max_age_feature: int,
    detach_pred_input: bool,
) -> Dict[str, torch.Tensor]:
    """
    Returns a pack with:
      gt0 (context), gt_future_tokens (K), plus MODEL/LR/KF predicted complex taps (K*taps).
    """
    model.eval()
    seq = seq1.to(device).unsqueeze(0)  # (1,S,D)
    _, S, D = seq.shape
    t0 = clamp_int(t0, 0, S - K - 1)

    gt0_tok = seq[0, t0, :].detach()
    gt_future_tok = seq[0, (t0 + 1):(t0 + 1 + K), :].detach()  # (K,D)

    gt0_c = tokens_to_complex(gt0_tok.unsqueeze(0), taps=taps, gain_scale=gain_scale)[0]  # (taps,)
    gt_future_c = tokens_to_complex(gt_future_tok.unsqueeze(0), taps=taps, gain_scale=gain_scale)[0]  # (K,taps)

    # MODEL preds
    pred_model_tok = model_kstep_predict_batch(
        model=model,
        seq=seq,
        t0=t0,
        K=K,
        taps=taps,
        gain_scale=gain_scale,
        eps_norm=eps_norm,
        max_age_feature=max_age_feature,
        detach_pred_input=detach_pred_input,
    )[0]  # (K,D)
    pred_model_c = tokens_to_complex(pred_model_tok.unsqueeze(0), taps=taps, gain_scale=gain_scale)[0]  # (K,taps)

    # LR/KF future taps
    lr_c_flat = lr_predict_future_complex_from_gt0(gt0_c.unsqueeze(0), future_len=K * taps)[0]  # (K*taps,)
    kf_c_flat = kf_predict_future_complex_from_gt0(gt0_c.unsqueeze(0), future_len=K * taps, r_rel=KF_R_REL, q_rel=KF_Q_REL)[0]

    lr_c = lr_c_flat.reshape(K, taps)
    kf_c = kf_c_flat.reshape(K, taps)

    return {
        "t0": torch.tensor(t0),
        "gt0_tok": gt0_tok.detach().cpu(),
        "gt_future_tok": gt_future_tok.detach().cpu(),     # (K,D)
        "gt0_c": gt0_c.detach().cpu(),                     # (taps,)
        "gt_future_c": gt_future_c.detach().cpu(),         # (K,taps)
        "model_c": pred_model_c.detach().cpu(),            # (K,taps)
        "lr_c": lr_c.detach().cpu(),                       # (K,taps)
        "kf_c": kf_c.detach().cpu(),                       # (K,taps)
        "K": torch.tensor(K),
        "taps": torch.tensor(taps),
        "gamma": torch.tensor(float(gamma)),
    }


def print_micro(pack: Dict[str, torch.Tensor], show_context: bool = True):
    t0 = int(pack["t0"].item())
    K = int(pack["K"].item())
    taps = int(pack["taps"].item())
    step_ms = 5.0 * taps

    gt0 = pack["gt0_c"]            # (taps,)
    gtF = pack["gt_future_c"]      # (K,taps)
    mF  = pack["model_c"]          # (K,taps)
    lrF = pack["lr_c"]
    kfF = pack["kf_c"]

    def gain_phase(hc: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        g = hc.abs().numpy()
        ph = (torch.angle(hc).numpy() * (180.0 / math.pi))
        return g, ph

    # header
    print("\n" + "=" * 132)
    print(f"[MICRO] t0={t0} | taps={taps} (each 5ms) | step={step_ms:.1f}ms | horizon={K} steps ({K*step_ms:.1f}ms)")
    print("        Columns show GT vs MODEL vs LR(one-step) vs KF(one-step) for each 5ms tap.")
    print("=" * 132)

    header = (
        " time_ms | blk | tap |   GT|h|  MODEL|h|    LR|h|    KF|h| | "
        " GTph(deg) MODELph      LRph      KFph |  |eM|  |eLR|  |eKF|"
    )
    print(header)
    print("-" * len(header))

    # context rows
    if show_context:
        g0, p0 = gain_phase(gt0)
        for tap in range(taps):
            time_ms = 0.0 + 5.0 * tap
            print(
                f"{time_ms:7.1f} | GT0 | {tap:3d} | "
                f"{_fmt(float(g0[tap]),8,4)} {_fmt(None,9,4)} {_fmt(None,9,4)} {_fmt(None,9,4)} | "
                f"{_fmt(float(p0[tap]),9,2)} {_fmt(None,9,2)} {_fmt(None,9,2)} {_fmt(None,9,2)} | "
                f"{_fmt(None,6,4)} {_fmt(None,6,4)} {_fmt(None,6,4)}"
            )

    # future rows
    for k in range(K):
        gt_g, gt_p = gain_phase(gtF[k])
        m_g,  m_p  = gain_phase(mF[k])
        lr_g, lr_p = gain_phase(lrF[k])
        kf_g, kf_p = gain_phase(kfF[k])

        for tap in range(taps):
            time_ms = (k + 1) * step_ms + 5.0 * tap
            eM  = float((mF[k, tap]  - gtF[k, tap]).abs().item())
            eLR = float((lrF[k, tap] - gtF[k, tap]).abs().item())
            eKF = float((kfF[k, tap] - gtF[k, tap]).abs().item())
            blk = f"+{k+1:1d}"
            print(
                f"{time_ms:7.1f} | {blk:3s} | {tap:3d} | "
                f"{_fmt(float(gt_g[tap]),8,4)} {_fmt(float(m_g[tap]),9,4)} {_fmt(float(lr_g[tap]),9,4)} {_fmt(float(kf_g[tap]),9,4)} | "
                f"{_fmt(float(gt_p[tap]),9,2)} {_fmt(float(m_p[tap]),9,2)} {_fmt(float(lr_p[tap]),9,2)} {_fmt(float(kf_p[tap]),9,2)} | "
                f"{_fmt(eM,6,4)} {_fmt(eLR,6,4)} {_fmt(eKF,6,4)}"
            )

    # quick summary over all future taps
    gt_flat = gtF.reshape(-1)
    m_flat  = mF.reshape(-1)
    lr_flat = lrF.reshape(-1)
    kf_flat = kfF.reshape(-1)

    def cmse_nmse(pred: torch.Tensor) -> Tuple[float, float]:
        cmse = float(((pred - gt_flat).abs() ** 2).mean().item())
        den = float((gt_flat.abs() ** 2).mean().item())
        return cmse, cmse / max(den, 1e-12)

    cm_m, nm_m   = cmse_nmse(m_flat)
    cm_lr, nm_lr = cmse_nmse(lr_flat)
    cm_kf, nm_kf = cmse_nmse(kf_flat)

    print("-" * len(header))
    print(f"[MICRO] Future taps={K*taps} | MODEL cmse={cm_m:.3e} nmse={nm_m:.3e} | LR cmse={cm_lr:.3e} nmse={nm_lr:.3e} | KF cmse={cm_kf:.3e} nmse={nm_kf:.3e}")
    print("=" * 132 + "\n")

def main():
    print(f"[DEVICE] {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[CUDA] {torch.cuda.get_device_name(DEVICE)}")

    # ckpt
    ckpt_path = CKPT_EXPLICIT.strip() if CKPT_EXPLICIT.strip() else find_latest_ckpt(CKPT_ROOT)
    print(f"[CKPT] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    samples_per_step = int(cfg.get("samples_per_step", 5))
    taps = samples_per_step
    tok_dim = int(cfg.get("tok_dim", 3 * taps))
    gain_scale = float(cfg.get("gain_scale", 20.0))
    eps_norm = float(cfg.get("eps_norm", 1e-6))
    split_seed = int(cfg.get("seed", 123))
    torch.manual_seed(split_seed)

    # these come from training config (fallbacks are safe)
    K = int(cfg.get("pred_steps", 5))
    gamma = float(cfg.get("loss_gamma", 0.5))
    use_age = bool(cfg.get("use_age_feature", True))
    max_age_feature = int(cfg.get("max_age_feature", 32))
    detach_pred_input = bool(cfg.get("detach_pred_input", True))

    step_ms = 5.0 * taps

    print(f"[CFG ] seed={split_seed} samples_per_step={samples_per_step} -> taps={taps} step={step_ms:.1f}ms tok_dim={tok_dim}")
    print(f"[CFG ] K={K} gamma={gamma} use_age={use_age} max_age_feature={max_age_feature} detach_pred_input={detach_pred_input}")
    print(f"[BASE] LR/KF context = ONE step ({taps} taps = {step_ms:.1f}ms), predict {K} steps ({K*step_ms:.1f}ms)")
    print(f"[KF  ] KF_R_REL={KF_R_REL} KF_Q_REL={KF_Q_REL}")

    # data
    checkpoint_x_path = ckpt.get("x_path", "") or cfg.get("x_path", "")
    x_path = X_NPZ_EXPLICIT.strip() if X_NPZ_EXPLICIT.strip() else checkpoint_x_path or find_x_npz(DATA_READY_ROOT)
    print(f"[DATA] Using: {x_path}")

    windows = XNPZWindows(x_path, samples_per_step=samples_per_step)
    N = len(windows)
    if N < 2:
        raise ValueError(f"At least 2 windows are required to recreate the held-out split, got {N}")

    # same split as training (perm + 90/10)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(N)
    n_train = min(max(int(round(0.9 * N)), 1), N - 1)
    test_indices = perm[n_train:]

    test_ds = PerRxSequenceDataset(windows, test_indices, gain_scale=gain_scale)

    pin = (DEVICE.type == "cuda")
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        drop_last=False
    )
    print(f"[SPLIT] windows N={N} => test_windows={len(test_indices)} => test_examples={len(test_ds)} (x6)")

    # model
    model = build_model_from_ckpt(ckpt, tok_dim_fallback=tok_dim).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # eval
    res = eval_all_methods(
        model=model,
        loader=test_loader,
        device=DEVICE,
        taps=taps,
        gain_scale=gain_scale,
        eps_norm=eps_norm,
        K=K,
        gamma=gamma,
        max_age_feature=max_age_feature,
        detach_pred_input=detach_pred_input,
        max_batches=MAX_TEST_BATCHES
    )

    def print_method(name: str):
        r = res[name]
        print(f"\n[{name}] weighted (w_k = gamma^(k-1), gamma={gamma}) over K={K} steps")
        print(f"  tok_mse_w     = {r['tok_mse_w']:.6e}")
        print(f"  complex_mse_w = {r['complex_mse_w']:.6e}")
        print(f"  nmse_w        = {r['nmse_w']:.6e}")
        print(f"  phase_mae_w   = {r['phase_mae_deg_w']:.3f} deg")
        print("  per-step NMSE = " + "  ".join([f"{x:.3e}" for x in r["per_step_nmse"]]))

    print_method("MODEL")
    print_method("LR")
    print_method("KF")

    # micro
    if PRINT_MICRO:
        if 0 <= EXAMPLE_TEST_INDEX < len(test_ds):
            seq1 = test_ds[EXAMPLE_TEST_INDEX]  # (S,D) CPU
            S = seq1.shape[0]
            t0 = clamp_int(EXAMPLE_T0, 0, S - K - 1)
            pack = micro_predict_all(
                model=model,
                seq1=seq1,
                t0=t0,
                device=DEVICE,
                taps=taps,
                gain_scale=gain_scale,
                eps_norm=eps_norm,
                K=K,
                gamma=gamma,
                max_age_feature=max_age_feature,
                detach_pred_input=detach_pred_input,
            )
            print_micro(pack, show_context=MICRO_SHOW_CONTEXT)
        else:
            print(f"[WARN] EXAMPLE_TEST_INDEX={EXAMPLE_TEST_INDEX} out of range (0..{len(test_ds)-1}). Skip micro.")

    print("Done.")


if __name__ == "__main__":
    main()
