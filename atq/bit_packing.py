"""2-bit packing for ternary weights.

Encodes {-1, 0, +1} as {0, 1, 2} and packs 4 values per uint8 byte,
achieving 16x compression vs FP32.
"""

import torch
from torch import Tensor


def pack_ternary(weight: Tensor, scale: float) -> tuple[dict, tuple]:
    """Pack a ternary weight tensor into 2-bit representation.

    Maps: -1 to 0, 0 to 1, +1 to 2. Packs 4 values per uint8 byte.

    Returns:
        packed: Dict with 'packed_data' (uint8 tensor) and 'scale' (float).
        original_shape: Original tensor shape for unpacking.
    """
    original_shape = weight.shape
    flat = weight.flatten().to(torch.int8)
    encoded = (flat + 1).to(torch.uint8)

    remainder = encoded.numel() % 4
    if remainder != 0:
        padding = 4 - remainder
        encoded = torch.cat([encoded, torch.ones(padding, dtype=torch.uint8)])
    else:
        padding = 0

    encoded = encoded.reshape(-1, 4)
    packed = (
        encoded[:, 0]
        | (encoded[:, 1] << 2)
        | (encoded[:, 2] << 4)
        | (encoded[:, 3] << 6)
    )

    packed_dict = {
        "packed_data": packed,
        "scale": scale,
        "padding": padding,
    }
    return packed_dict, original_shape


def unpack_ternary(packed: dict, original_shape: tuple) -> tuple[Tensor, float]:
    """Unpack 2-bit packed data back to ternary tensor."""
    packed_data = packed["packed_data"]
    scale = packed["scale"]

    v0 = packed_data & 0x03
    v1 = (packed_data >> 2) & 0x03
    v2 = (packed_data >> 4) & 0x03
    v3 = (packed_data >> 6) & 0x03

    flat = torch.stack([v0, v1, v2, v3], dim=1).flatten().to(torch.float32)

    total = 1
    for s in original_shape:
        total *= s
    flat = flat[:total]

    flat = flat - 1.0
    return flat.reshape(original_shape), scale
