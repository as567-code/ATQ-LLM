"""Adaptive Ternary Quantization -- core quantization functions.

Implements two thresholding strategies:
1. Magnitude-based: threshold = alpha * std(W)
2. Sparsity-target: threshold = percentile(|W|, sparsity_target)

Both produce ternary weights in {-1, 0, +1} with a learned scale factor.
"""

import torch
from torch import Tensor


def adaptive_threshold_magnitude(weight: Tensor, alpha: float = 0.7) -> float:
    """Compute threshold as alpha * std(weight)."""
    return alpha * weight.std().item()


def adaptive_threshold_sparsity(weight: Tensor, sparsity_target: float = 0.5) -> float:
    """Compute threshold to achieve a target sparsity level."""
    abs_weight = weight.abs().flatten()
    k = int(sparsity_target * abs_weight.numel())
    k = max(1, min(k, abs_weight.numel() - 1))
    threshold = abs_weight.kthvalue(k).values.item()
    return threshold


def compute_scale_factor(weight: Tensor, ternary_weight: Tensor) -> float:
    """Compute scale factor as mean of |W| where ternary weight is non-zero."""
    mask = ternary_weight != 0
    if mask.sum() == 0:
        return 1.0
    return weight.abs()[mask].mean().item()


def ternary_quantize(weight: Tensor, threshold: float) -> tuple[Tensor, float]:
    """Quantize weights to ternary {-1, 0, +1} using the given threshold."""
    ternary = torch.zeros_like(weight)
    pos_mask = weight > threshold
    neg_mask = weight < -threshold
    ternary[pos_mask] = 1.0
    ternary[neg_mask] = -1.0
    scale = compute_scale_factor(weight, ternary)
    return ternary, scale
