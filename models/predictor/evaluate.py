import torch
import torch.nn.functional as F
from .gru_predictor import SimpleChannelRNN
from .feature_encoder import postprocess_pred_tokens, tokens_to_complex
from torch.utils.data import DataLoader
from typing import Dict
from .train_config import (
    DETACH_PRED_INPUT,
    LOSS_GAMMA,
    MAX_AGE_FEATURE,
    PRED_STEPS,
)

@torch.no_grad()
def eval_loader_kstep(
    model: SimpleChannelRNN,
    loader: DataLoader,
    device: torch.device,
    eval_batches: int = 0
) -> Dict[str, float]:

    was_training = model.training
    model.eval()

    tok_sum = 0.0
    cmse_sum = 0.0
    nmse_sum = 0.0
    n = 0

    K = int(PRED_STEPS)
    w = torch.tensor([float(LOSS_GAMMA) ** i for i in range(K)], device=device, dtype=torch.float32)
    w_sum = float(w.sum().item())

    for bi, seq in enumerate(loader):
        if eval_batches > 0 and bi >= eval_batches:
            break

        seq = seq.to(device, non_blocking=True)
        B, S, D = seq.shape
        if S < (K + 1):
            continue

        t0 = int(torch.randint(low=0, high=(S - K), size=(1,), device=device).item())

        # prefix (age=0)
        init_tok = model.init_tok.view(1, 1, D).expand(B, 1, D)
        if model.use_age_feature:
                    init_in = torch.cat([init_tok, torch.zeros((B, 1, 1), device=device, dtype=seq.dtype)], dim=-1)
        else:
            init_in = init_tok

        if t0 > 0:
            prefix_tok = seq[:, :t0, :]
            if model.use_age_feature:
                prefix_in = torch.cat([prefix_tok, torch.zeros((B, t0, 1), device=device, dtype=seq.dtype)], dim=-1)
            else:
                prefix_in = prefix_tok
            prefix_full = torch.cat([init_in, prefix_in], dim=1)
        else:
            prefix_full = init_in
        
        _, h = model.rnn(prefix_full)

        # rollout and accumulate weighted metrics
        x = seq[:, t0, :]
        tok_acc = 0.0
        cmse_acc = 0.0
        denom_acc = 0.0

        for i in range(K):
            age_val = i
            if model.use_age_feature:
                age_norm = (torch.full((B,), age_val, device=device, dtype=torch.long)
                            .clamp(0, MAX_AGE_FEATURE).to(seq.dtype) / float(MAX_AGE_FEATURE)).unsqueeze(-1)
            else:
                age_norm = None

            out, h = model.rnn_step(x, age_norm, h)
            pred = postprocess_pred_tokens(model.out_proj(out))
            tgt = seq[:, t0 + i + 1, :]

            wi = float(w[i].item())

            tok_acc += wi * F.mse_loss(pred, tgt, reduction="mean").item()     

            pred_h = tokens_to_complex(pred)
            tgt_h = tokens_to_complex(tgt)
            cmse_acc += wi * ((pred_h - tgt_h).abs() ** 2).mean().item()
            denom_acc += wi * (tgt_h.abs() ** 2).mean().item()

            if i != (K - 1):
                 x = pred.detach() if DETACH_PRED_INPUT else pred

        tok_mse = tok_acc / max(w_sum, 1e-12)
        cmse = cmse_acc / max(w_sum,1e-12)
        nmse = cmse / max(denom_acc / max(w_sum, 1e-12), 1e-12)

        tok_sum += tok_mse
        cmse_sum += cmse
        nmse_sum += nmse
        n += 1

    if was_training:
        model.train()

    if n == 0:
        return {"tok_mse": float("nan"), "complex_mse": float("nan"), "nmse": float("nan")}
    return {"tok_mse": tok_sum / n, "complex_mse": cmse_sum / n, "nmse": nmse_sum / n}
