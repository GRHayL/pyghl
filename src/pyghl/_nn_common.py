from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class FeatureTransformStats:
    """Feature transform configuration.

    kind codes:
      0 = identity
      1 = log10(max(x, eps))
    """

    kind: torch.Tensor
    eps: float = 1e-30
    q_idx: int = 0
    s_idx: int = 2


@torch.no_grad()
def compute_x_bounds_from_inputs(
    X_raw: torch.Tensor,
    *,
    q_idx: int,
    s_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if X_raw.shape[1] <= max(q_idx, s_idx):
        raise ValueError(
            f"Input has d_in={X_raw.shape[1]} but requires q_idx={q_idx}, s_idx={s_idx}."
        )
    q = X_raw[:, q_idx : q_idx + 1]
    s = X_raw[:, s_idx : s_idx + 1]

    x_lo = 1.0 + q - s
    x_hi = 2.0 + 2.0 * q - s
    width = x_hi - x_lo
    return x_lo, x_hi, width


def y01_clamp(y01: torch.Tensor, y_eps: float) -> torch.Tensor:
    return torch.clamp(y01, min=float(y_eps), max=1.0 - float(y_eps))


def y01_to_x(
    X_raw: torch.Tensor,
    y01: torch.Tensor,
    *,
    q_idx: int,
    s_idx: int,
    y_eps: float,
) -> torch.Tensor:
    x_lo, _, width = compute_x_bounds_from_inputs(X_raw, q_idx=q_idx, s_idx=s_idx)
    w = torch.clamp(width.to(torch.float32), min=1.0e-12)
    y01c = y01_clamp(y01, y_eps)
    return (x_lo.to(y01c.dtype) + y01c * w.to(y01c.dtype)).to(torch.float32)


@torch.no_grad()
def apply_feature_transform(X: torch.Tensor, ft: FeatureTransformStats) -> torch.Tensor:
    Xf = X.to(torch.float32).clone()
    kind = ft.kind.to(device=Xf.device)
    mask = kind == 1
    if bool(mask.any().item()):
        eps = torch.tensor(ft.eps, device=Xf.device, dtype=torch.float32)
        Xf[:, mask] = torch.log10(torch.clamp(Xf[:, mask], min=eps))
    return Xf


@torch.no_grad()
def apply_robust_minmax(
    X_t: torch.Tensor,
    stats: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    lo = stats["lo"].to(device=X_t.device, dtype=torch.float32)
    hi = stats["hi"].to(device=X_t.device, dtype=torch.float32)
    invrng = stats["invrng"].to(device=X_t.device, dtype=torch.float32)

    below = X_t < lo
    above = X_t > hi
    frac_clipped = float((below | above).float().mean().item())

    Xc = torch.clamp(X_t, min=lo, max=hi)
    X01 = (Xc - lo) * invrng
    return X01, frac_clipped


class TinyMLP_Logit(nn.Module):
    """input -> (Linear + HardTanh) x n_hidden -> Linear -> logit"""

    def __init__(
        self, in_dim: int, hidden_dim: int = 8, n_hidden: int = 3, out_dim: int = 1
    ):
        super().__init__()
        if n_hidden < 1:
            raise ValueError("n_hidden must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_hidden = int(n_hidden)
        self.out_dim = int(out_dim)

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(self.in_dim, self.hidden_dim, bias=True))
        for _ in range(self.n_hidden - 1):
            self.fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
        self.out = nn.Linear(self.hidden_dim, self.out_dim, bias=True)

        self.act_h = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lyr in self.fcs:
            nn.init.xavier_uniform_(lyr.weight, gain=1.0)
            nn.init.zeros_(lyr.bias)
        nn.init.xavier_uniform_(self.out.weight, gain=0.25)
        nn.init.zeros_(self.out.bias)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        h = x01
        for lyr in self.fcs:
            h = self.act_h(lyr(h))
        return self.out(h)
