import torch


def pinball_loss(y, pred, quantiles):
    """y    (B, tau)
    pred (B, tau, n_q)

    Sum of the pinball loss over quantiles, averaged over batch and horizon.
    """
    total = 0.0
    for j in range(len(quantiles)):
        q = quantiles[j]
        d = y - pred[:, :, j]
        total = total + torch.maximum(q * d, (q - 1.0) * d).mean()
    return total
