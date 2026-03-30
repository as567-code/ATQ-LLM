import torch
import pytest
from atq.bit_packing import pack_ternary, unpack_ternary


class TestPackUnpackRoundTrip:
    def test_small_tensor(self):
        w = torch.tensor([[-1.0, 0.0, 1.0, 0.0]])
        scale = 1.5
        packed, shape = pack_ternary(w, scale)
        recovered, recovered_scale = unpack_ternary(packed, shape)
        assert torch.equal(recovered, w)
        assert abs(recovered_scale - scale) < 1e-6

    def test_random_ternary(self):
        w = torch.randint(-1, 2, (64, 64)).float()
        scale = 2.3
        packed, shape = pack_ternary(w, scale)
        recovered, recovered_scale = unpack_ternary(packed, shape)
        assert torch.equal(recovered, w)
        assert abs(recovered_scale - scale) < 1e-6

    def test_all_zeros(self):
        w = torch.zeros(16, 16)
        packed, shape = pack_ternary(w, 1.0)
        recovered, recovered_scale = unpack_ternary(packed, shape)
        assert torch.equal(recovered, w)

    def test_all_ones(self):
        w = torch.ones(16, 16)
        packed, shape = pack_ternary(w, 1.0)
        recovered, recovered_scale = unpack_ternary(packed, shape)
        assert torch.equal(recovered, w)

    def test_all_neg_ones(self):
        w = -torch.ones(16, 16)
        packed, shape = pack_ternary(w, 1.0)
        recovered, recovered_scale = unpack_ternary(packed, shape)
        assert torch.equal(recovered, w)

    def test_non_multiple_of_4_length(self):
        w = torch.tensor([[1.0, -1.0, 0.0, 1.0, -1.0]])
        packed, shape = pack_ternary(w, 1.0)
        recovered, _ = unpack_ternary(packed, shape)
        assert torch.equal(recovered, w)


class TestCompressionRatio:
    def test_compression_ratio(self):
        w = torch.randint(-1, 2, (256, 256)).float()
        packed, shape = pack_ternary(w, 1.0)
        original_bytes = w.numel() * 4
        packed_bytes = packed["packed_data"].numel()
        ratio = original_bytes / packed_bytes
        assert ratio >= 15.0


class TestEdgeCases:
    def test_1d_tensor(self):
        w = torch.tensor([-1.0, 0.0, 1.0])
        packed, shape = pack_ternary(w, 1.0)
        recovered, _ = unpack_ternary(packed, shape)
        assert torch.equal(recovered, w)

    def test_large_tensor(self):
        w = torch.randint(-1, 2, (512, 512)).float()
        packed, shape = pack_ternary(w, 1.0)
        recovered, _ = unpack_ternary(packed, shape)
        assert torch.equal(recovered, w)
