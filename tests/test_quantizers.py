import torch
import pytest
from atq.quantizers import (
    ternary_quantize,
    adaptive_threshold_magnitude,
    adaptive_threshold_sparsity,
    compute_scale_factor,
)


class TestAdaptiveThresholdMagnitude:
    def test_returns_positive_threshold(self):
        w = torch.randn(128, 128)
        threshold = adaptive_threshold_magnitude(w, alpha=0.7)
        assert threshold > 0

    def test_scales_with_alpha(self):
        w = torch.randn(128, 128)
        t1 = adaptive_threshold_magnitude(w, alpha=0.5)
        t2 = adaptive_threshold_magnitude(w, alpha=1.0)
        assert t2 > t1

    def test_equals_alpha_times_std(self):
        w = torch.randn(256, 256)
        alpha = 0.7
        threshold = adaptive_threshold_magnitude(w, alpha=alpha)
        expected = alpha * w.std().item()
        assert abs(threshold - expected) < 1e-5


class TestAdaptiveThresholdSparsity:
    def test_returns_positive_threshold(self):
        w = torch.randn(128, 128)
        threshold = adaptive_threshold_sparsity(w, sparsity_target=0.5)
        assert threshold > 0

    def test_higher_sparsity_higher_threshold(self):
        w = torch.randn(256, 256)
        t1 = adaptive_threshold_sparsity(w, sparsity_target=0.3)
        t2 = adaptive_threshold_sparsity(w, sparsity_target=0.7)
        assert t2 > t1

    def test_achieves_target_sparsity(self):
        w = torch.randn(512, 512)
        target = 0.5
        threshold = adaptive_threshold_sparsity(w, sparsity_target=target)
        zeros = (w.abs() < threshold).float().mean().item()
        assert abs(zeros - target) < 0.05


class TestTernaryQuantize:
    def test_output_values_in_ternary_set(self):
        w = torch.randn(128, 128)
        w_q, scale = ternary_quantize(w, threshold=0.5)
        unique = set(w_q.unique().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_scale_factor_positive(self):
        w = torch.randn(128, 128)
        w_q, scale = ternary_quantize(w, threshold=0.5)
        assert scale > 0

    def test_zero_threshold_no_zeros(self):
        w = torch.randn(128, 128)
        w_q, scale = ternary_quantize(w, threshold=0.0)
        nonzero_frac = (w_q != 0).float().mean().item()
        assert nonzero_frac > 0.99

    def test_high_threshold_mostly_zeros(self):
        w = torch.randn(128, 128)
        w_q, scale = ternary_quantize(w, threshold=10.0)
        zero_frac = (w_q == 0).float().mean().item()
        assert zero_frac > 0.99

    def test_signs_preserved(self):
        w = torch.tensor([[-2.0, 0.1, 1.5], [0.05, -0.3, 3.0]])
        w_q, scale = ternary_quantize(w, threshold=0.2)
        assert w_q[0, 0].item() == -1.0
        assert w_q[0, 2].item() == 1.0


class TestComputeScaleFactor:
    def test_scale_is_mean_of_abs_nonzero(self):
        w = torch.tensor([[-2.0, 0.0, 1.0, 3.0]])
        w_q = torch.tensor([[-1.0, 0.0, 1.0, 1.0]])
        scale = compute_scale_factor(w, w_q)
        expected = (2.0 + 1.0 + 3.0) / 3.0
        assert abs(scale - expected) < 1e-5

    def test_all_zeros_returns_one(self):
        w = torch.zeros(4, 4)
        w_q = torch.zeros(4, 4)
        scale = compute_scale_factor(w, w_q)
        assert scale == 1.0
