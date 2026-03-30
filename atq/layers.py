"""TernaryLinear -- drop-in replacement for nn.Linear with ATQ.

Uses Straight-Through Estimator (STE) so gradients flow through
the ternary quantization during backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from atq.quantizers import (
    adaptive_threshold_magnitude,
    adaptive_threshold_sparsity,
    ternary_quantize,
)


class STEQuantize(torch.autograd.Function):
    """Straight-Through Estimator for ternary quantization.

    The scale parameter is intentionally kept outside this function so that
    its gradient flows through the standard autograd path (not STE).
    """

    @staticmethod
    def forward(ctx, weight: Tensor, threshold: float) -> Tensor:
        ternary = torch.zeros_like(weight)
        ternary[weight > threshold] = 1.0
        ternary[weight < -threshold] = -1.0
        return ternary

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, mode="magnitude", alpha=0.7, sparsity_target=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.alpha = alpha
        self.sparsity_target = sparsity_target

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def _compute_threshold(self):
        if self.mode == "magnitude":
            return adaptive_threshold_magnitude(self.weight.data, self.alpha)
        else:
            return adaptive_threshold_sparsity(self.weight.data, self.sparsity_target)

    def get_quantized_weight(self):
        threshold = self._compute_threshold()
        w_q, _ = ternary_quantize(self.weight.data, threshold)
        return w_q

    def forward(self, x):
        threshold = self._compute_threshold()
        # Apply STE for the hard ternary step; multiply scale outside so its
        # gradient flows through the standard autograd path.
        w_ternary = STEQuantize.apply(self.weight, threshold)
        w_q = w_ternary * self.scale
        return F.linear(x, w_q, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, mode={self.mode}"
