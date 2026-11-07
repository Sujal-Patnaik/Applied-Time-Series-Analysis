import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


# Parametric heads for DeepTCN (paper section 3.2.2). The network outputs two
# raw numbers per future hour, and these turn them into a distribution:
#
#   gaussian  mu, sigma    the paper's example; on log1p(y) if log_target
#   negbin    mu, alpha    Var = mu + alpha mu^2, over-dispersed counts,
#                          suggested by the paper for long-tailed series
#   zip       lambda, pi   Poisson with an extra point mass at zero
#
# negbin and zip are count distributions and neither dataset is counts
# (traffic is occupancy, electricity is kW), so they are fitted on
# c = round(y * count_unit) from config. The unit is a resolution, and for
# zip it is more than that: Poisson has Var = mean, so in the original units
# Var(y) = mean(y) / count_unit and the unit fixes the noise level. negbin
# has alpha to absorb this and is barely affected.
#
# All three are scaled the DeepAR way (Salinas et al. 2020): the network
# sees the history divided by nu = 1 + mean|history|, location parameters
# are multiplied back by nu, and the likelihood is on the unscaled target.
# The 1 + keeps nu sane when the history is all zeros, which happens on
# electricity (series that start late) and which the quantile model's
# 1e-6 floor turns into a target of 4.3e6 (log1p scale).
#
# Every head ends in the same three quantiles as the quantile model, so the
# calibration half of the project takes them unchanged.

EPS = 1e-6

NAMES = ["gaussian", "negbin", "zip"]


def is_count(dist):
    return dist in ("negbin", "zip")


def check(dist):
    if dist != "quantile" and dist not in NAMES:
        raise ValueError(
            "unknown dist " + dist + ", use quantile, " + ", ".join(NAMES)
        )


# ---------------------------------------------------------------- parameters


def params(dist, raw, scale):
    """raw (B, tau, 2), scale nu (B,) -> two (B, tau) tensors on the
    unscaled fitting scale (log1p(y) or y for gaussian, counts otherwise).
    """
    nu = scale[:, None]
    if dist == "gaussian":
        mu = nu * raw[:, :, 0]
        sigma = nu * F.softplus(raw[:, :, 1]) + EPS
        return mu, sigma
    if dist == "negbin":
        mu = nu * F.softplus(raw[:, :, 0]) + EPS
        alpha = F.softplus(raw[:, :, 1]) / torch.sqrt(nu) + EPS
        return mu, alpha
    if dist == "zip":
        lam = nu * F.softplus(raw[:, :, 0]) + EPS
        # pi stays as a logit, the loss needs log(pi) and log(1-pi)
        return lam, raw[:, :, 1]
    raise ValueError("unknown dist " + dist)


# ---------------------------------------------------------------- likelihood


def nll(dist, y, raw, scale):
    """Negative log-likelihood averaged over batch and horizon.

    y     (B, tau)    unscaled target, integer counts for negbin and zip
    raw   (B, tau, 2)
    scale (B,)

    Done in float64: electricity counts reach 2e5, and lgamma of that in
    float32 is only good to about 0.5.
    """
    a, b = params(dist, raw.double(), scale.double())
    y = y.double()

    if dist == "gaussian":
        mu, sigma = a, b
        z = (y - mu) / sigma
        ll = -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * z * z

    elif dist == "negbin":
        mu, alpha = a, b
        r = 1.0 / alpha
        ll = (
            torch.lgamma(y + r)
            - torch.lgamma(y + 1.0)
            - torch.lgamma(r)
            - r * torch.log1p(alpha * mu)
            + y * (torch.log(alpha * mu) - torch.log1p(alpha * mu))
        )

    elif dist == "zip":
        lam, logit = a, b
        log_pi = F.logsigmoid(logit)
        log_1m = F.logsigmoid(-logit)
        pois = y * torch.log(lam) - lam - torch.lgamma(y + 1.0)
        zero = torch.logaddexp(log_pi, log_1m - lam)
        ll = torch.where(y == 0, zero, log_1m + pois)

    else:
        raise ValueError("unknown dist " + dist)

    return -ll.mean()


# ---------------------------------------------------------------- quantiles


def quantiles(dist, raw, scale, qs):
    """raw (B, tau, 2) and scale (B,) as numpy -> (B, tau, n_q) numpy.

    Same scale as nll: log1p(y) or y for gaussian, counts otherwise. The
    caller maps them back to y.
    """
    with torch.no_grad():
        a, b = params(
            dist,
            torch.as_tensor(raw, dtype=torch.float64),
            torch.as_tensor(scale, dtype=torch.float64),
        )
    a = a.numpy().astype(np.float64)
    b = b.numpy().astype(np.float64)

    out = np.zeros(a.shape + (len(qs),), dtype=np.float64)
    for j in range(len(qs)):
        q = qs[j]
        if dist == "gaussian":
            out[:, :, j] = a + b * stats.norm.ppf(q)

        elif dist == "negbin":
            # scipy: n = 1/alpha, p = 1/(1 + alpha mu) gives mean mu
            n = 1.0 / b
            p = 1.0 / (1.0 + b * a)
            out[:, :, j] = stats.nbinom.ppf(q, n, p)

        elif dist == "zip":
            # P(Y <= k) = pi + (1-pi) F_pois(k), so below pi the answer is 0
            pi = 1.0 / (1.0 + np.exp(-b))
            qa = (q - pi) / (1.0 - pi)
            v = stats.poisson.ppf(np.clip(qa, 0.0, 1.0 - 1e-12), a)
            out[:, :, j] = np.where(qa <= 0, 0.0, v)

        else:
            raise ValueError("unknown dist " + dist)

    return out.astype(np.float32)
