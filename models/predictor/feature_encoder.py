import torch
import torch.nn.functional as F
from .train_config import EPS_NORM, GAIN_DIM, GAIN_SCALE, PHASE_DIM
import math

def complex_to_gcs_tokens(
    h_taps: torch.Tensor,
    gain_scale: float = GAIN_SCALE,
) -> torch.Tensor:

    mag = h_taps.abs()*float(gain_scale)
    ang = torch.angle(h_taps)
    cos = torch.cos(ang)
    sin = torch.sin(ang)

    features = torch.cat([mag,cos,sin],dim=-1)

    return features.to(torch.float32)

def postprocess_pred_tokens(
    token: torch.Tensor,
    taps: int = GAIN_DIM,
    eps_norm: float = EPS_NORM,
) -> torch.Tensor:
    gain_raw = token[..., :taps]
    cos_raw = token[..., taps:2*taps]
    sin_raw = token[..., 2*taps:3*taps]

    gain = F.softplus(gain_raw)

    denom = torch.sqrt(cos_raw**2 + sin_raw**2 + float(eps_norm))
    cos_n = cos_raw / denom
    sin_n = sin_raw / denom

    return torch.cat([gain, cos_n, sin_n], dim=-1)

def tokens_to_complex(token: torch.Tensor, taps=GAIN_DIM, gain_scale=GAIN_SCALE) -> torch.Tensor:
    gain_s = token[..., :taps]
    cos = token[..., taps:2*taps]
    sin = token[..., 2*taps:3*taps]

    gain = gain_s / float(gain_scale)

    return torch.complex(gain * cos, gain * sin)

def tok_phase_radians(tok: torch.Tensor, taps: int) -> torch.Tensor:
    cos = tok[..., taps:2*taps]
    sin = tok[..., 2*taps:3*taps]
    return torch.atan2(sin, cos)


def phase_mae_deg_from_tokens(pred_tok: torch.Tensor, tgt_tok: torch.Tensor, taps: int) -> float:
    """
    Circular MAE in degrees over taps, averaged over batch and time.
    pred_tok/tgt_tok should be postprocessed tokens.
    """
    pred_ang = tok_phase_radians(pred_tok, taps=taps)
    tgt_ang  = tok_phase_radians(tgt_tok,  taps=taps)
    d = pred_ang - tgt_ang
    d_wrap = torch.atan2(torch.sin(d), torch.cos(d))
    return (d_wrap.abs().mean().item() * (180.0 / math.pi))
