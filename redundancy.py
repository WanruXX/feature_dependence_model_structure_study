from __future__ import annotations

import math
import torch
from torch import Tensor


def _matrix(features: Tensor) -> Tensor:
    values = torch.as_tensor(features).detach().cpu().double()
    if values.ndim != 2 or values.shape[0] < 8 or values.shape[1] < 2:
        raise ValueError("features must have shape [N,F] with N >= 8 and F >= 2")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("features must be finite")
    return values


@torch.no_grad()
def compute_redundancy_scores(features: Tensor, *, seed: int = 42,
                              ridge: float = 0.01) -> dict[str, Tensor]:
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    values = _matrix(features)
    centered = values - values.mean(0)
    standardized = centered / centered.square().mean(0).sqrt().clamp_min(1e-8)
    correlation = (standardized.T @ standardized / values.shape[0]).clamp(-1, 1)
    correlation.fill_diagonal_(0)
    corr_score = correlation.abs().sum(1) / (values.shape[1] - 1)
    order = torch.randperm(len(values), generator=torch.Generator().manual_seed(seed))
    boundary = max(2, int(0.7 * len(values)))
    fit, held_out = values[order[:boundary]], values[order[boundary:]]
    fit_mean, fit_scale = fit.mean(0), fit.std(0, unbiased=False).clamp_min(1e-8)
    fit = (fit - fit_mean) / fit_scale
    held_out = (held_out - fit_mean) / fit_scale
    recon_score = torch.zeros(values.shape[1], dtype=values.dtype)
    for feature in range(values.shape[1]):
        predictors = torch.arange(values.shape[1]) != feature
        design, target = fit[:, predictors], fit[:, feature]
        gram = design.T @ design / len(fit)
        coefficients = torch.linalg.solve(
            gram + ridge * torch.eye(design.shape[1], dtype=values.dtype),
            design.T @ target / len(fit),
        )
        observed = held_out[:, feature]
        variance = (observed - observed.mean()).square().sum()
        if float(variance) > 1e-8:
            error = (observed - held_out[:, predictors] @ coefficients).square().sum()
            recon_score[feature] = (1 - error / variance).clamp(0, 1)
    return {
        "correlation": corr_score.float(),
        "reconstructability": recon_score.float(),
        "combined": ((corr_score + recon_score) / 2).float(),
        "fit_positions": order[:boundary],
        "evaluation_positions": order[boundary:],
    }