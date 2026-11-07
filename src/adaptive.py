import math

import torch
import torch.nn as nn

from model import DeepTCN


# Adaptive receptive field. The encoder is not touched; a gate decides how
# much of the input window each series gets to see.
#
#   tau_i  = tau_max * sigmoid(f(ctx_i))         memory length of series i
#   m_i(l) = sigmoid((tau_i - l) / s)            l = 0 newest .. L-1 oldest
#   input  = [y * m_i, m_i]                       mask also as its own channel
#
# The mask goes in as a second channel because y * m = 0 would otherwise
# look exactly like an observed zero.
#
# tau_max = L + 4 s0 rather than L: with tau = L the oldest step still gets
# sigmoid(1/s) ~ 0.5. With the extra 4 s0 every step starts at >= 0.98, so
# the gated model starts as the plain DeepTCN (the mask channel weights also
# start at zero, see model.py) and moves away only if the loss says so.
#
# s is annealed from L/10 towards s_end during training. A sharp mask from
# the start has zero gradient almost everywhere and tau would never move.
#
# Modes, picked by config:
#   none    plain DeepTCN, full window
#   fixed   hard cutoff at fixed_len steps
#   global  one learned tau for every series
#   series  tau per window from log1p(n_observed) and zero ratio
#
# A per-series embedding as extra context was tried and dropped: with a few
# windows per series it barely trained and never separated the series types.

MODES = ["none", "fixed", "global", "series"]

INIT_LOGIT = 4.0   # sigmoid(4) = 0.98, so tau starts near tau_max


def context(y_hist):
    """Per-window features from the history alone, (B, L) -> (B, 2).

    n_observed counts steps since the first non-zero value in the window,
    so a series that started 3 steps ago gives 3 whatever came before.
    """
    L = y_hist.shape[1]
    nz = (y_hist != 0).float()
    seen = torch.cumsum(nz, dim=1) > 0           # True from the first non-zero
    n_obs = seen.float().sum(dim=1)
    zero_ratio = 1.0 - nz.mean(dim=1)
    return torch.stack([torch.log1p(n_obs) / math.log1p(L), zero_ratio], dim=1)


class MemoryGate(nn.Module):
    def __init__(self, mode, in_len, fixed_len=None, hidden=16):
        super().__init__()
        if mode not in MODES:
            raise ValueError("gate mode must be one of " + ", ".join(MODES))
        self.mode = mode
        self.L = in_len
        self.fixed_len = fixed_len
        s0 = in_len / 10.0
        self.tau_max = in_len + 4.0 * s0
        # temperature, set from outside by anneal()
        self.register_buffer("s", torch.tensor(s0))

        # l for every input position: the tensor runs oldest -> newest
        self.register_buffer(
            "age", torch.arange(in_len - 1, -1, -1, dtype=torch.float32)
        )

        if mode == "global":
            self.logit = nn.Parameter(torch.tensor(INIT_LOGIT))
        elif mode == "series":
            self.mlp = nn.Sequential(
                nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )
            with torch.no_grad():
                self.mlp[2].weight.mul_(0.01)
                self.mlp[2].bias.fill_(INIT_LOGIT)

    @property
    def learned(self):
        return self.mode in ("global", "series")

    def anneal(self, frac, s_end):
        """frac 0 -> s0, frac 1 -> s_end, geometric in between."""
        s0 = self.L / 10.0
        self.s.fill_(s0 * (s_end / s0) ** min(max(frac, 0.0), 1.0))

    def tau(self, y_hist):
        """(B,) memory length in steps, in (0, tau_max)."""
        if self.mode == "global":
            z = self.logit.expand(y_hist.shape[0])
        else:
            z = self.mlp(context(y_hist)).squeeze(1)
        return self.tau_max * torch.sigmoid(z)

    def forward(self, y_hist):
        """Returns mask (B, L) and tau (B,), tau is None if nothing is learned."""
        b = y_hist.shape[0]
        if self.mode == "none":
            return torch.ones_like(y_hist), None
        if self.mode == "fixed":
            m = (self.age < self.fixed_len).float()
            return m.expand(b, -1), None
        tau = self.tau(y_hist)
        m = torch.sigmoid((tau[:, None] - self.age[None, :]) / self.s)
        return m, tau


class GatedTCN(nn.Module):
    """DeepTCN that sees y * mask and the mask. With mode none it is the
    plain model, no extra channel."""

    def __init__(self, gate_mode, n_series, dilations, channels, kernel, n_out,
                 in_len, positive=True, fixed_len=None):
        super().__init__()
        self.gate = MemoryGate(gate_mode, in_len, fixed_len)
        n_extra = 0 if gate_mode == "none" else 1
        self.tcn = DeepTCN(
            n_series, dilations, channels, kernel, n_out, positive, n_extra
        )
        self.last_tau = None

    def forward(self, y_hist, series_idx, cal_hist, cal_fut):
        m, tau = self.gate(y_hist)
        self.last_tau = tau
        if self.gate.mode == "none":
            return self.tcn(y_hist, series_idx, cal_hist, cal_fut)
        return self.tcn(
            y_hist * m, series_idx, cal_hist, cal_fut, extra=m.unsqueeze(2)
        )

    def penalty(self, lam):
        """lam * mean(tau / L) for the last batch, 0 when tau is not learned."""
        if self.last_tau is None or lam == 0:
            return 0.0
        return lam * (self.last_tau / self.gate.L).mean()
