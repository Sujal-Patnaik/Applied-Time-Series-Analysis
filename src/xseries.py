import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EncBlock, bn_last


# DeepTCN for traffic with two optional extensions, both starting at zero
# effect so the model begins as the plain baseline:
#
#   attn  cross-series attention on the encoder summary h. All series go
#         through in one forward pass, so A covers the whole panel.
#             A = softmax(E E^T / sqrt(d)), diagonal masked out
#             h = h + alpha * (A @ h),  alpha one scalar, init 0
#         The diagonal is masked because with E ~ N(0,1) the self-score
#         |e_i|^2 dominates every row and A would start as the identity.
#
#   tau   learned per-series memory length from the series embedding
#             tau_i = tau_max * sigmoid(MLP(E_i))
#             m_i(l) = sigmoid((tau_i - l) / s),  l = 0 newest .. L-1 oldest
#             input = [y * m, m]
#         tau_max = L + 4 s0 rather than L, so every step starts at mask
#         >= 0.98 (with tau_max = L the oldest step would start at 0.5).
#         The mask channel's input weights start at zero.

COV = 4 + 3   # hour-of-day and day-of-week embeddings


class XTCN(nn.Module):
    def __init__(self, n_series, in_len=168, dilations=(1, 2, 4, 8, 16, 20, 32, 32, 32),
                 channels=16, emb_dim=32, n_q=3, attn=False, tau=False, s0=16.0):
        super().__init__()
        self.attn, self.use_tau, self.L = attn, tau, in_len
        self.E = nn.Embedding(n_series, emb_dim)
        self.emb_h = nn.Embedding(24, 4)
        self.emb_d = nn.Embedding(7, 3)
        cov = emb_dim + COV

        n_in = 1 + (1 if tau else 0) + cov
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.blocks = nn.ModuleList([EncBlock(channels, 2, d) for d in dilations])

        # decoder: delta = R(X_fut) + h
        self.r1 = nn.Linear(cov, channels)
        self.rb1 = nn.BatchNorm1d(channels)
        self.r2 = nn.Linear(channels, channels)
        self.rb2 = nn.BatchNorm1d(channels)
        self.head = nn.Linear(channels, n_q)

        if attn:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        if tau:
            self.tau_max = in_len + 4.0 * s0
            self.register_buffer("s", torch.tensor(s0))
            self.register_buffer("age", torch.arange(in_len - 1, -1, -1).float())
            self.tau_mlp = nn.Sequential(
                nn.Linear(emb_dim, 16), nn.ReLU(), nn.Linear(16, 1)
            )
            with torch.no_grad():
                self.inp.weight[:, 1].zero_()
                self.tau_mlp[2].weight.mul_(0.01)
                self.tau_mlp[2].bias.fill_(4.0)   # sigmoid(4) = 0.98

    def cov(self, sidx, cal):
        """sidx (N,), cal (N, T, 2) -> (N, T, emb_dim + 7)"""
        e = self.E(sidx).unsqueeze(1).expand(-1, cal.shape[1], -1)
        return torch.cat([e, self.emb_h(cal[..., 0]), self.emb_d(cal[..., 1])], 2)

    def tau(self, sidx):
        return self.tau_max * torch.sigmoid(self.tau_mlp(self.E(sidx)).squeeze(1))

    def mask(self, sidx):
        return torch.sigmoid((self.tau(sidx)[:, None] - self.age[None]) / self.s)

    def attention(self, sidx):
        e = self.E(sidx)
        sc = e @ e.T / e.shape[1] ** 0.5
        sc = sc.masked_fill(torch.eye(len(sidx), dtype=torch.bool, device=sc.device), float("-inf"))
        return torch.softmax(sc, dim=-1)

    def encode(self, y, sidx, cal_hist):
        """Encoder output for every step, (N, C, L)."""
        parts = [y.unsqueeze(2)]
        if self.use_tau:
            m = self.mask(sidx)
            parts = [(y * m).unsqueeze(2), m.unsqueeze(2)]
        x = torch.cat(parts + [self.cov(sidx, cal_hist)], 2).permute(0, 2, 1)
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return x

    def forward(self, y, sidx, cal_hist, cal_fut):
        """y (N, L), sidx (N,), cal_hist (N, L, 2), cal_fut (N, H, 2) -> (N, H, n_q)"""
        h = self.encode(y, sidx, cal_hist)[:, :, -1]
        if self.attn:
            h = h + self.alpha * (self.attention(sidx) @ h)
        r = F.relu(bn_last(self.rb1, self.r1(self.cov(sidx, cal_fut))))
        r = bn_last(self.rb2, self.r2(r))
        return self.head(r + h.unsqueeze(1))
