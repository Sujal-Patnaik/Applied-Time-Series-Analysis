import torch
import torch.nn as nn
import torch.nn.functional as F


def bn_last(bn, x):
    """BatchNorm1d over the last dim of a (B, T, H) tensor."""
    return bn(x.permute(0, 2, 1)).permute(0, 2, 1)


class EncBlock(nn.Module):
    """One residual block of the encoder: two dilated causal convolutions."""

    def __init__(self, ch, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv1 = nn.Conv1d(ch, ch, k, dilation=d)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, k, dilation=d)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        # left padding only, so step t never sees anything after t
        out = F.pad(x, (self.pad, 0))
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.pad(out, (self.pad, 0))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class DeepTCN(nn.Module):
    def __init__(self, n_series, dilations, channels, kernel, n_out,
                 positive=True, n_extra=0):
        """n_out is the number of quantiles for the quantile model, or the
        number of distribution parameters for a parametric one. positive
        keeps the final ReLU; the parametric heads constrain their own
        parameters in distributions.py and need the raw output.

        n_extra adds input channels next to the target (adaptive.py uses
        one for the memory mask). Their weights start at zero, so at
        initialisation the model computes exactly what it would without
        them."""
        super().__init__()
        self.positive = positive
        self.n_extra = n_extra

        self.emb_series = nn.Embedding(n_series, 8)
        self.emb_hour = nn.Embedding(24, 4)
        self.emb_dow = nn.Embedding(7, 3)
        self.emb_month = nn.Embedding(12, 2)

        cov_dim = 8 + 4 + 3 + 2

        self.inp = nn.Conv1d(1 + n_extra + cov_dim, channels, 1)
        with torch.no_grad():
            self.inp.weight[:, 1 : 1 + n_extra].zero_()
        self.blocks = nn.ModuleList(
            [EncBlock(channels, kernel, d) for d in dilations]
        )

        # resnet-v on the future covariates
        self.d1 = nn.Linear(cov_dim, 64)
        self.b1 = nn.BatchNorm1d(64)
        self.d2 = nn.Linear(64, 64)
        self.b2 = nn.BatchNorm1d(64)

        self.merge = nn.Linear(channels + 64, 64)
        self.bm = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(64, n_out)

    def covariates(self, series_idx, cal):
        """series_idx (B,), cal (B, T, 3) -> (B, T, cov_dim)"""
        t = cal.shape[1]
        s = self.emb_series(series_idx)
        s = s.unsqueeze(1).expand(-1, t, -1)
        h = self.emb_hour(cal[:, :, 0])
        d = self.emb_dow(cal[:, :, 1])
        m = self.emb_month(cal[:, :, 2])
        return torch.cat([s, h, d, m], dim=2)

    def resnet_v(self, x):
        r = F.relu(bn_last(self.b1, self.d1(x)))
        return bn_last(self.b2, self.d2(r))

    def forward(self, y_hist, series_idx, cal_hist, cal_fut, extra=None):
        """y_hist (B, L), series_idx (B,), cal_hist (B, L, 3), cal_fut (B, tau, 3)
        extra (B, L, n_extra) when n_extra > 0

        Returns (B, tau, n_out).
        """
        cov_h = self.covariates(series_idx, cal_hist)
        parts = [y_hist.unsqueeze(2)]
        if self.n_extra > 0:
            parts.append(extra)
        x = torch.cat(parts + [cov_h], dim=2)

        x = x.permute(0, 2, 1)
        x = self.inp(x)
        for blk in self.blocks:
            x = blk(x)

        # summary of the history: the last step of the encoder output
        h = x[:, :, -1]

        cov_f = self.covariates(series_idx, cal_fut)
        r = self.resnet_v(cov_f)

        tau = cov_f.shape[1]
        h_exp = h.unsqueeze(1).expand(-1, tau, -1)

        m = torch.cat([h_exp, r], dim=2)
        m = F.relu(bn_last(self.bm, self.merge(m)))
        m = self.drop(m)

        out = self.head(m)
        if self.positive:
            return F.relu(out)
        return out
