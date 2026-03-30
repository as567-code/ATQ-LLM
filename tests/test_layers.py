import torch
import pytest
from atq.layers import TernaryLinear


class TestTernaryLinearForward:
    def test_output_shape(self):
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64)
        y = layer(x)
        assert y.shape == (8, 32)

    def test_output_shape_with_bias(self):
        layer = TernaryLinear(64, 32, bias=True)
        x = torch.randn(8, 64)
        y = layer(x)
        assert y.shape == (8, 32)

    def test_output_shape_no_bias(self):
        layer = TernaryLinear(64, 32, bias=False)
        x = torch.randn(8, 64)
        y = layer(x)
        assert y.shape == (8, 32)

    def test_quantized_weights_are_ternary(self):
        layer = TernaryLinear(64, 32)
        _ = layer(torch.randn(1, 64))
        w_q = layer.get_quantized_weight()
        unique = set(w_q.unique().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_sparsity_mode(self):
        layer = TernaryLinear(128, 64, mode="sparsity", sparsity_target=0.5)
        _ = layer(torch.randn(1, 128))
        w_q = layer.get_quantized_weight()
        zero_frac = (w_q == 0).float().mean().item()
        assert abs(zero_frac - 0.5) < 0.1


class TestTernaryLinearBackward:
    def test_gradients_flow(self):
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (32, 64)
        assert x.grad is not None

    def test_gradients_not_all_zero(self):
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.weight.grad.abs().sum() > 0

    def test_scale_has_gradient(self):
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.scale.grad is not None


class TestTernaryLinearModes:
    def test_magnitude_mode_default(self):
        layer = TernaryLinear(64, 32)
        assert layer.mode == "magnitude"

    def test_sparsity_mode(self):
        layer = TernaryLinear(64, 32, mode="sparsity", sparsity_target=0.3)
        assert layer.mode == "sparsity"
        assert layer.sparsity_target == 0.3
