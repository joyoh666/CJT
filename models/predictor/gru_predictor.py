import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class SimpleChannelRNN(nn.Module):
    def __init__(self, 
                 tok_dim: int,
                 hidden: int,
                 layers: int,
                 dropout: float,
                 rnn_type: str="GRU",
                 use_age_feature: bool=True):
        super().__init__()
        self.tok_dim = int(tok_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.rnn_type = str(rnn_type).upper()
        self.use_age_feature = bool(use_age_feature)

        self.input_dim = self.tok_dim + (1 if self.use_age_feature else 0)

        self.init_tok = nn.Parameter(torch.zeros(self.tok_dim))
        nn.init.normal_(self.init_tok, std=0.02)

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden,
                num_layers=self.layers,
                dropout=self.dropout if self.layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=self.input_dim,
                hidden_size=self.hidden,
                num_layers=self.layers,
                dropout=self.dropout if self.layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")

        self.out_proj = nn.Linear(self.hidden, self.tok_dim)

    def _pack_in(self,
                tok: torch.Tensor,
                age_norm: torch.Tensor | None):

        if not self.use_age_feature:
            return tok
        assert age_norm is not None
        return torch.cat([tok, age_norm], dim=-1)

    def rnn_step(self,
                 tok_in: torch.Tensor,
                 age_norm: torch.Tensor | None,
                 h: torch.Tensor | None):

        x = self._pack_in(tok_in, age_norm).unsqueeze(1)
        y, h_next = self.rnn(x, h)
        return y[:,0,:], h_next

def build_model_from_ckpt(ckpt: Dict, tok_dim_fallback: int) -> SimpleChannelRNN:
    cfg = ckpt.get("config", {})
    rnn_type = cfg.get("rnn_type", "GRU")
    hidden = int(cfg.get("hidden_size", 64))
    layers = int(cfg.get("num_layers", 1))
    dropout = float(cfg.get("dropout", 0.0))
    tok_dim = int(cfg.get("tok_dim", tok_dim_fallback))
    use_age = bool(cfg.get("use_age_feature", True))
    return SimpleChannelRNN(tok_dim=tok_dim, hidden=hidden, layers=layers, dropout=dropout, rnn_type=rnn_type, use_age_feature=use_age)
