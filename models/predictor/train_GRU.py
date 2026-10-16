import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from .dataset import PerRxSequenceDataset, XNPZWindows, find_x_npz
from .evaluate import eval_loader_kstep
from .gru_predictor import SimpleChannelRNN
from .feature_encoder import postprocess_pred_tokens
from .train_config import (
    AUGMENT_RANDOM_GLOBAL_PHASE,
    BATCH_SIZE,
    DATA_READY_ROOT,
    DETACH_PRED_INPUT,
    DEVICE,
    DROPOUT,
    EPS_NORM,
    EPOCHS,
    EVAL_BATCHES,
    EVAL_EVERY_STEPS,
    GAIN_SCALE,
    GRAD_CLIP,
    HIDDEN_SIZE,
    LOG_TRAIN_EVERY_STEPS,
    LOSS_GAMMA,
    LR,
    MAX_AGE_FEATURE,
    NUM_LAYERS,
    PRED_STEPS,
    RNN_TYPE,
    SAMPLES_PER_STEP,
    SAVE_DIR,
    SAVE_EVERY_EPOCHS,
    SEED,
    TAPS,
    TOK_DIM,
    USE_AGE_FEATURE,
    WEIGHT_DECAY,
    X_NPZ_EXPLICIT,
)

def k_step_loss(model: SimpleChannelRNN,
                seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, S, D = seq.shape
    assert D == TOK_DIM

    K = int(PRED_STEPS)
    if S < (K + 1):
        raise ValueError(f"Sequence has {S} steps, but at least {K + 1} are required for K={K}")

    t0 = int(torch.randint(low=0, high=(S-K), size=(1,), device=seq.device).item())

    # prefix to set hidden after consuming GT[0..t0-1], age=0
    init_tok = model.init_tok.view(1 ,1, D).expand(B, 1, D)
    if model.use_age_feature:
        age0_init = torch.zeros((B, 1, 1),device=seq.device, dtype=seq.dtype)
        init_in = torch.cat([init_tok, age0_init], dim=-1)
    else:
        init_in = init_tok

    if t0 > 0:
        prefix_tok = seq[:, :t0, :]
        if model.use_age_feature:
            prefix_age0 = torch.zeros((B, t0, 1), device=seq.device, dtype=seq.dtype)
            prefix_in = torch.cat([prefix_tok, prefix_age0], dim=-1)
        else:
            prefix_in = prefix_tok
        prefix_full = torch.cat([init_in, prefix_in], dim=1)
    else:
        prefix_full = init_in

    _, h = model.rnn(prefix_full)

    #weights
    w = torch.tensor([float(LOSS_GAMMA) ** i for i in range(K)], device=seq.device, dtype=seq.dtype)
    sum_w = float(w.sum().item())

    #rollout
    loss_acc = torch.zeros((), device=seq.device, dtype=seq.dtype)

    x = seq[:,t0, :]
    for i in range(K):
        age_val = i
        if model.use_age_feature:
            age_norm = (torch.full((B,), age_val, device=seq.device, dtype=torch.long)
                                    .clamp(0, MAX_AGE_FEATURE).to(seq.dtype) / float(MAX_AGE_FEATURE)).unsqueeze(-1)  
        else:
            age_norm = None

        out, h = model.rnn_step(x, age_norm, h)
        pred = postprocess_pred_tokens(model.out_proj(out))
        tgt = seq[:, t0 + i + 1, :]

        mse_i = F.mse_loss(pred, tgt, reduction="mean")
        loss_acc += w[i] * mse_i

        if i != (K - 1):
            x = pred.detach() if DETACH_PRED_INPUT else pred

    loss = loss_acc / max(sum_w, 1e-12)
    return loss, {"t0": float(t0), "sum_w": sum_w}

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[DEVICE] {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[CUDA] name: {torch.cuda.get_device_name(DEVICE)}")

    x_path = X_NPZ_EXPLICIT.strip() if X_NPZ_EXPLICIT.strip() else find_x_npz(DATA_READY_ROOT)
    print(f"[DATA] Using: {x_path}")

    windows = XNPZWindows(x_path)
    N = len(windows)
    if N < 2:
        raise ValueError(f"At least 2 windows are required for a train/validation split, got {N}")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(N)
    n_train = min(max(int(round(0.9 * N)), 1), N - 1)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    print(f"[SPLIT] windows N={N} => train={len(train_indices)} val={len(val_indices)} (expanded x6 per-Rx)")

    train_ds = PerRxSequenceDataset(windows, train_indices, train=True)
    val_ds  = PerRxSequenceDataset(windows, val_indices,  train=False)

    L = int(windows[0].shape[0])
    S = train_ds.S
    step_ms = 5.0 * SAMPLES_PER_STEP
    print(f"[SEQ] L={L} samples, step={SAMPLES_PER_STEP} -> {step_ms:.1f}ms, S={S} steps, TOK_DIM={TOK_DIM}")
    print(f"[TOK] [gain*{GAIN_SCALE:.1f}, cos, sin] taps={TAPS} | pred: softplus(gain) + normalize(cos,sin)")
    print(f"[KSTEP] K={PRED_STEPS}, LOSS_GAMMA={LOSS_GAMMA} (w: 1,0.5,0.25,...)")
    print(f"[AGE ] enabled={USE_AGE_FEATURE}, MAX_AGE_FEATURE={MAX_AGE_FEATURE} | age=0 on GT input, then 1..K-1 on AR inputs")
    print(f"[AR  ] DETACH_PRED_INPUT={DETACH_PRED_INPUT}")

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
        drop_last=(len(train_ds) >= BATCH_SIZE),
    )
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin, drop_last=False)

    model = SimpleChannelRNN(
        tok_dim=TOK_DIM,
        hidden=HIDDEN_SIZE,
        layers=NUM_LAYERS,
        dropout=DROPOUT,
        rnn_type=RNN_TYPE,
        use_age_feature=USE_AGE_FEATURE
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    os.makedirs(SAVE_DIR, exist_ok=True)

    global_step = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        run_n = 0

        for seq in train_loader:
            seq = seq.to(DEVICE, non_blocking=True)  # (B,S,TOK_DIM)

            loss, dbg = k_step_loss(model, seq)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

            global_step += 1
            run_loss += float(loss.item())
            run_n += 1

            if LOG_TRAIN_EVERY_STEPS > 0 and (global_step % LOG_TRAIN_EVERY_STEPS == 0):
                elapsed = time.time() - t_start
                print(
                    f"[train step {global_step:7d} | epoch {epoch:4d}] "
                    f"lossK={run_loss/run_n:.6e} (t0={int(dbg['t0'])}) | t={elapsed/60:.1f} min"
                )
                run_loss = 0.0
                run_n = 0

            if EVAL_EVERY_STEPS > 0 and (global_step % EVAL_EVERY_STEPS == 0):
                m = eval_loader_kstep(model, val_loader, DEVICE, eval_batches=EVAL_BATCHES)
                elapsed = time.time() - t_start
                print(
                    f"[VAL step {global_step:7d} | epoch {epoch:4d}] "
                    f"Kstep tok_mse={m['tok_mse']:.3e} cmse={m['complex_mse']:.3e} nmse={m['nmse']:.3e} "
                    f"| t={elapsed/60:.1f} min"
                )

        # end-of-epoch eval
        m = eval_loader_kstep(model, val_loader, DEVICE, eval_batches=EVAL_BATCHES)
        msg = (f"[EPOCH {epoch:4d}] Kstep tok_mse={m['tok_mse']:.3e} "
               f"cmse={m['complex_mse']:.3e} nmse={m['nmse']:.3e}")

        # save
        saved_path = None
        if (SAVE_EVERY_EPOCHS is not None) and (SAVE_EVERY_EPOCHS > 0) and (epoch % SAVE_EVERY_EPOCHS == 0):
            saved_path = os.path.join(SAVE_DIR, f"rnn_gcs_kstep_age_gamma{LOSS_GAMMA:.2f}_K{PRED_STEPS}_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "x_path": x_path,
                "config": {
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "grad_clip": GRAD_CLIP,
                    "rnn_type": RNN_TYPE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "samples_per_step": SAMPLES_PER_STEP,
                    "tok_dim": TOK_DIM,
                    "gain_scale": GAIN_SCALE,
                    "eps_norm": EPS_NORM,
                    "augment_random_global_phase": AUGMENT_RANDOM_GLOBAL_PHASE,
                    "detach_pred_input": DETACH_PRED_INPUT,
                    "pred_steps": PRED_STEPS,
                    "loss_gamma": LOSS_GAMMA,
                    "use_age_feature": USE_AGE_FEATURE,
                    "max_age_feature": MAX_AGE_FEATURE,
                    "data_ready_root": DATA_READY_ROOT,
                }
            }, saved_path)

        print(msg + (f" | saved: {saved_path}" if saved_path else " | (no checkpoint)"))


if __name__ == "__main__":
    main()
