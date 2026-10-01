"""Inference on prepared cases."""

from __future__ import annotations

import numpy as np
import torch

from .data import select_inputs

__all__ = ["predict_bodysize"]


@torch.no_grad()
def predict_bodysize(model, standardizers, sigma_scales, cases, targets,
                     device="cpu", batch_size=8):
    """
    {target: (mean, sigma, lo95, hi95)} in the target's units for prepared
    cases, with the calibrated sigma.
    """
    model.eval().to(device)
    mus, sds = [], []
    for s in range(0, len(cases), batch_size):
        batch = [select_inputs(c) for c in cases[s:s + batch_size]]
        proj = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        axial = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        mu, log_var = model(proj, axial)
        mus.append(mu.float().cpu().numpy())
        sds.append(torch.exp(0.5 * log_var).float().cpu().numpy())
    mu, sd = np.concatenate(mus), np.concatenate(sds)
    out = {}
    for k, t in enumerate(targets):
        std, scale = standardizers[t], sigma_scales[t]
        mean, sigma = std.inverse_transform(mu[:, k], sd[:, k] * scale)
        lo, hi = std.interval(mu[:, k], sd[:, k] * scale, 1.96)
        out[t] = (mean, sigma, lo, hi)
    return out
