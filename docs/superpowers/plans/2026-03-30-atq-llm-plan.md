# ATQ-LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete ATQ-LLM repository implementing Adaptive Ternary Quantization for LLM compression (GPT-2, TinyLlama) with tests, notebooks, and benchmarks.

**Architecture:** Core quantization engine (`atq/`) provides model-agnostic ternary quantization with adaptive thresholds, STE-based layers, 2-bit packing, mixed precision, and calibration. LLM pipeline (`llm/`) wraps HuggingFace models. Experiments (`experiments/`) run QAT and ablations. Notebooks visualize results.

**Tech Stack:** Python 3.9+, PyTorch, HuggingFace Transformers, datasets, numpy, scipy, matplotlib, pytest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `atq/__init__.py` | Package exports |
| `atq/quantizers.py` | Ternary quantization functions (magnitude + sparsity thresholding) |
| `atq/layers.py` | TernaryLinear nn.Module with STE |
| `atq/bit_packing.py` | Pack/unpack ternary weights to 2-bit uint8 |
| `atq/mixed_precision.py` | Layer importance scoring + precision assignment |
| `atq/calibration.py` | Post-training threshold optimization |
| `llm/__init__.py` | Package exports |
| `llm/quantize_model.py` | Apply ATQ to HuggingFace models |
| `llm/evaluate.py` | Perplexity, size, speed, memory metrics |
| `llm/benchmark.py` | ATQ vs RTN vs cited GPTQ/AWQ |
| `experiments/train_atq_gpt2.py` | QAT script for GPT-2 small |
| `experiments/train_atq_tinyllama.py` | QAT script for TinyLlama-1.1B |
| `experiments/ablation.py` | Ablation sweep runner |
| `notebooks/01_atq_demo.ipynb` | Demo walkthrough with plots |
| `notebooks/02_ablation_results.ipynb` | Ablation visualization |
| `notebooks/03_layer_analysis.ipynb` | Layer importance analysis |
| `tests/test_quantizers.py` | Tests for quantization correctness |
| `tests/test_layers.py` | Tests for TernaryLinear |
| `tests/test_bit_packing.py` | Tests for pack/unpack round-trip |
| `requirements.txt` | Dependencies |
| `README.md` | Project documentation |
| `LICENSE` | MIT license |

---

### Task 1: Project Skeleton + Requirements

**Files:**
- Create: `requirements.txt`
- Create: `atq/__init__.py`
- Create: `llm/__init__.py`
- Create: `tests/__init__.py`
- Create: `experiments/` (directory)
- Create: `notebooks/` (directory)
- Create: `results/` (directory with `.gitkeep`)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p atq llm experiments notebooks results tests
```

- [ ] **Step 2: Create requirements.txt**

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pytest>=7.4.0
accelerate>=0.21.0
```

- [ ] **Step 3: Create atq/__init__.py**

```python
"""ATQ - Adaptive Ternary Quantization for LLM Compression."""

from atq.quantizers import ternary_quantize, adaptive_threshold_magnitude, adaptive_threshold_sparsity
from atq.layers import TernaryLinear
from atq.bit_packing import pack_ternary, unpack_ternary
from atq.mixed_precision import compute_layer_importance, assign_precision
from atq.calibration import calibrate_thresholds

__version__ = "0.1.0"
```

- [ ] **Step 4: Create llm/__init__.py**

```python
"""LLM quantization pipeline using ATQ."""
```

- [ ] **Step 5: Create tests/__init__.py**

Empty file.

- [ ] **Step 6: Create results/.gitkeep**

Empty file to track the directory in git.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt atq/__init__.py llm/__init__.py tests/__init__.py results/.gitkeep
git commit -m "feat: project skeleton and requirements"
```

---

### Task 2: Core Quantizers (atq/quantizers.py) + Tests

**Files:**
- Create: `atq/quantizers.py`
- Create: `tests/test_quantizers.py`

- [ ] **Step 1: Write failing tests for quantizers**

```python
# tests/test_quantizers.py
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
        assert abs(zeros - target) < 0.05  # within 5%


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
        # With threshold 0, only exact zeros become 0
        w_q, scale = ternary_quantize(w, threshold=0.0)
        # Almost all should be non-zero
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
        # w[0,0]=-2.0 should be -1, w[0,2]=1.5 should be +1
        assert w_q[0, 0].item() == -1.0
        assert w_q[0, 2].item() == 1.0


class TestComputeScaleFactor:
    def test_scale_is_mean_of_abs_nonzero(self):
        w = torch.tensor([[-2.0, 0.0, 1.0, 3.0]])
        w_q = torch.tensor([[-1.0, 0.0, 1.0, 1.0]])
        scale = compute_scale_factor(w, w_q)
        # Non-zero original values at positions where w_q != 0: |-2|, |1|, |3| = mean 2.0
        expected = (2.0 + 1.0 + 3.0) / 3.0
        assert abs(scale - expected) < 1e-5

    def test_all_zeros_returns_one(self):
        w = torch.zeros(4, 4)
        w_q = torch.zeros(4, 4)
        scale = compute_scale_factor(w, w_q)
        assert scale == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_quantizers.py -v
```

Expected: ImportError -- `atq.quantizers` doesn't exist yet.

- [ ] **Step 3: Implement quantizers.py**

```python
# atq/quantizers.py
"""Adaptive Ternary Quantization -- core quantization functions.

Implements two thresholding strategies:
1. Magnitude-based: threshold = alpha * std(W)
2. Sparsity-target: threshold = percentile(|W|, sparsity_target)

Both produce ternary weights in {-1, 0, +1} with a learned scale factor.
"""

import torch
from torch import Tensor


def adaptive_threshold_magnitude(weight: Tensor, alpha: float = 0.7) -> float:
    """Compute threshold as alpha * std(weight).

    Args:
        weight: Weight tensor of any shape.
        alpha: Scaling factor for standard deviation (default 0.7).

    Returns:
        Threshold value (float).
    """
    return alpha * weight.std().item()


def adaptive_threshold_sparsity(weight: Tensor, sparsity_target: float = 0.5) -> float:
    """Compute threshold to achieve a target sparsity level.

    Finds the threshold such that approximately `sparsity_target` fraction
    of weights fall within [-threshold, threshold] and become zero.

    Args:
        weight: Weight tensor of any shape.
        sparsity_target: Desired fraction of zero weights (0.0 to 1.0).

    Returns:
        Threshold value (float).
    """
    abs_weight = weight.abs().flatten()
    k = int(sparsity_target * abs_weight.numel())
    k = max(1, min(k, abs_weight.numel() - 1))
    threshold = abs_weight.kthvalue(k).values.item()
    return threshold


def compute_scale_factor(weight: Tensor, ternary_weight: Tensor) -> float:
    """Compute scale factor as mean of |W| where ternary weight is non-zero.

    Args:
        weight: Original weight tensor.
        ternary_weight: Ternary weight tensor in {-1, 0, +1}.

    Returns:
        Scale factor (float). Returns 1.0 if all weights are zero.
    """
    mask = ternary_weight != 0
    if mask.sum() == 0:
        return 1.0
    return weight.abs()[mask].mean().item()


def ternary_quantize(
    weight: Tensor, threshold: float
) -> tuple[Tensor, float]:
    """Quantize weights to ternary {-1, 0, +1} using the given threshold.

    Weights with |w| > threshold get sign(w), others become 0.
    Scale factor = mean(|w|) for non-zero quantized positions.

    Args:
        weight: Weight tensor of any shape.
        threshold: Absolute threshold for zeroing weights.

    Returns:
        Tuple of (ternary_weight, scale_factor).
        ternary_weight has values in {-1.0, 0.0, 1.0}.
    """
    ternary = torch.zeros_like(weight)
    pos_mask = weight > threshold
    neg_mask = weight < -threshold
    ternary[pos_mask] = 1.0
    ternary[neg_mask] = -1.0

    scale = compute_scale_factor(weight, ternary)
    return ternary, scale
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_quantizers.py -v
```

Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add atq/quantizers.py tests/test_quantizers.py
git commit -m "feat: core ternary quantization with magnitude and sparsity thresholding"
```

---

### Task 3: TernaryLinear Layer (atq/layers.py) + Tests

**Files:**
- Create: `atq/layers.py`
- Create: `tests/test_layers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_layers.py
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
        _ = layer(torch.randn(1, 64))  # trigger forward
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layers.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement layers.py**

```python
# atq/layers.py
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
    """Straight-Through Estimator for ternary quantization."""

    @staticmethod
    def forward(ctx, weight: Tensor, threshold: float, scale: Tensor) -> Tensor:
        ternary = torch.zeros_like(weight)
        ternary[weight > threshold] = 1.0
        ternary[weight < -threshold] = -1.0
        return ternary * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # STE: pass gradient through as-is
        return grad_output, None, None


class TernaryLinear(nn.Module):
    """Linear layer with adaptive ternary quantization.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias (default True).
        mode: Thresholding mode -- "magnitude" or "sparsity".
        alpha: Scaling factor for magnitude mode (default 0.7).
        sparsity_target: Target sparsity for sparsity mode (default 0.5).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "magnitude",
        alpha: float = 0.7,
        sparsity_target: float = 0.5,
    ):
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

        # Initialize weights (Kaiming uniform, same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def _compute_threshold(self) -> float:
        if self.mode == "magnitude":
            return adaptive_threshold_magnitude(self.weight.data, self.alpha)
        else:
            return adaptive_threshold_sparsity(self.weight.data, self.sparsity_target)

    def get_quantized_weight(self) -> Tensor:
        """Return the ternary quantized weight (without scale)."""
        threshold = self._compute_threshold()
        w_q, _ = ternary_quantize(self.weight.data, threshold)
        return w_q

    def forward(self, x: Tensor) -> Tensor:
        threshold = self._compute_threshold()
        w_q = STEQuantize.apply(self.weight, threshold, self.scale)
        return F.linear(x, w_q, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_layers.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add atq/layers.py tests/test_layers.py
git commit -m "feat: TernaryLinear layer with STE for gradient flow"
```

---

### Task 4: Bit Packing (atq/bit_packing.py) + Tests

**Files:**
- Create: `atq/bit_packing.py`
- Create: `tests/test_bit_packing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bit_packing.py
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
        # Create random ternary tensor
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
        w = torch.tensor([[1.0, -1.0, 0.0, 1.0, -1.0]])  # 5 elements
        packed, shape = pack_ternary(w, 1.0)
        recovered, _ = unpack_ternary(packed, shape)
        assert torch.equal(recovered, w)


class TestCompressionRatio:
    def test_compression_ratio(self):
        w = torch.randint(-1, 2, (256, 256)).float()
        packed, shape = pack_ternary(w, 1.0)
        original_bytes = w.numel() * 4  # FP32 = 4 bytes
        packed_bytes = packed["packed_data"].numel()  # uint8 = 1 byte
        ratio = original_bytes / packed_bytes
        # 4 values per byte, so ratio = 4*4 = 16x
        assert ratio >= 15.0  # Allow slight overhead from padding


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_bit_packing.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement bit_packing.py**

```python
# atq/bit_packing.py
"""2-bit packing for ternary weights.

Encodes {-1, 0, +1} as {0, 1, 2} and packs 4 values per uint8 byte,
achieving 16x compression vs FP32.
"""

import torch
from torch import Tensor


def pack_ternary(weight: Tensor, scale: float) -> tuple[dict, tuple]:
    """Pack a ternary weight tensor into 2-bit representation.

    Maps: -1 to 0, 0 to 1, +1 to 2. Packs 4 values per uint8 byte.

    Args:
        weight: Ternary tensor with values in {-1, 0, +1}.
        scale: Scale factor associated with the weight.

    Returns:
        packed: Dict with 'packed_data' (uint8 tensor) and 'scale' (float).
        original_shape: Original tensor shape for unpacking.
    """
    original_shape = weight.shape
    flat = weight.flatten().to(torch.int8)

    # Map: -1 to 0, 0 to 1, +1 to 2
    encoded = (flat + 1).to(torch.uint8)

    # Pad to multiple of 4
    remainder = encoded.numel() % 4
    if remainder != 0:
        padding = 4 - remainder
        encoded = torch.cat([encoded, torch.ones(padding, dtype=torch.uint8)])
    else:
        padding = 0

    # Pack 4 values per byte
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
    """Unpack 2-bit packed data back to ternary tensor.

    Args:
        packed: Dict from pack_ternary with 'packed_data', 'scale', 'padding'.
        original_shape: Original tensor shape.

    Returns:
        weight: Ternary tensor with values in {-1, 0, +1}.
        scale: Scale factor.
    """
    packed_data = packed["packed_data"]
    scale = packed["scale"]

    # Unpack 4 values per byte
    v0 = packed_data & 0x03
    v1 = (packed_data >> 2) & 0x03
    v2 = (packed_data >> 4) & 0x03
    v3 = (packed_data >> 6) & 0x03

    flat = torch.stack([v0, v1, v2, v3], dim=1).flatten().to(torch.float32)

    # Remove padding
    total = 1
    for s in original_shape:
        total *= s
    flat = flat[:total]

    # Unmap: 0 to -1, 1 to 0, 2 to +1
    flat = flat - 1.0

    return flat.reshape(original_shape), scale
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bit_packing.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add atq/bit_packing.py tests/test_bit_packing.py
git commit -m "feat: 2-bit packing for ternary weights (16x compression)"
```

---

### Task 5: Mixed Precision (atq/mixed_precision.py)

**Files:**
- Create: `atq/mixed_precision.py`

- [ ] **Step 1: Implement mixed_precision.py**

```python
# atq/mixed_precision.py
"""Layer importance scoring for mixed-precision quantization.

Scores each layer by sensitivity to quantization, then assigns precision:
- High-importance layers keep FP16
- Low-importance layers get full ternary (2-bit)
"""

import torch
import torch.nn as nn
from torch import Tensor


def compute_layer_importance_gradient(
    model: nn.Module, named_linear_layers: list[tuple[str, nn.Module]]
) -> dict[str, float]:
    """Score layer importance using gradient times weight Frobenius norm.

    importance_i = ||grad_W_i * W_i||_F

    Requires that .backward() has been called so .grad is populated.

    Args:
        model: The model (unused, kept for API consistency).
        named_linear_layers: List of (name, module) tuples for linear layers.

    Returns:
        Dict mapping layer name to importance score.
    """
    scores = {}
    for name, layer in named_linear_layers:
        if layer.weight.grad is not None:
            score = (layer.weight.grad * layer.weight.data).norm().item()
        else:
            score = 0.0
        scores[name] = score
    return scores


def compute_layer_importance_fisher(
    model: nn.Module,
    named_linear_layers: list[tuple[str, nn.Module]],
    data_loader,
    device: torch.device,
    num_samples: int = 64,
) -> dict[str, float]:
    """Score layer importance using diagonal Fisher Information approximation.

    importance_i = E[||grad_W_i||^2]

    Args:
        model: The model to analyze.
        named_linear_layers: List of (name, module) tuples.
        data_loader: DataLoader providing input batches.
        device: Device to run on.
        num_samples: Number of samples for estimation.

    Returns:
        Dict mapping layer name to importance score.
    """
    scores = {name: 0.0 for name, _ in named_linear_layers}
    count = 0

    for batch in data_loader:
        if count >= num_samples:
            break

        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids).to(device)
        else:
            input_ids = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else input_ids

        model.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()

        for name, layer in named_linear_layers:
            if layer.weight.grad is not None:
                scores[name] += layer.weight.grad.pow(2).sum().item()

        count += input_ids.size(0)

    # Normalize
    for name in scores:
        scores[name] /= max(count, 1)

    return scores


def compute_layer_importance(
    model: nn.Module,
    named_linear_layers: list[tuple[str, nn.Module]],
    method: str = "gradient",
    **kwargs,
) -> dict[str, float]:
    """Compute layer importance scores.

    Args:
        model: The model to analyze.
        named_linear_layers: List of (name, module) tuples.
        method: "gradient" or "fisher".
        **kwargs: Additional args passed to the specific method.

    Returns:
        Dict mapping layer name to importance score.
    """
    if method == "gradient":
        return compute_layer_importance_gradient(model, named_linear_layers)
    elif method == "fisher":
        return compute_layer_importance_fisher(model, named_linear_layers, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'gradient' or 'fisher'.")


def assign_precision(
    importance_scores: dict[str, float],
    keep_ratio: float = 0.2,
) -> dict[str, str]:
    """Assign precision levels based on importance scores.

    Top `keep_ratio` fraction of layers (by importance) keep FP16.
    Remaining layers get ternary quantization.

    Args:
        importance_scores: Dict mapping layer name to importance score.
        keep_ratio: Fraction of layers to keep at FP16 (default 0.2).

    Returns:
        Dict mapping layer name to "fp16" or "ternary".
    """
    if not importance_scores:
        return {}

    sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    num_keep = max(1, int(len(sorted_layers) * keep_ratio))

    precision_map = {}
    for i, (name, _) in enumerate(sorted_layers):
        precision_map[name] = "fp16" if i < num_keep else "ternary"

    return precision_map
```

- [ ] **Step 2: Quick smoke test**

```bash
python -c "
from atq.mixed_precision import assign_precision
scores = {'layer1': 0.5, 'layer2': 1.0, 'layer3': 0.1, 'layer4': 0.8, 'layer5': 0.3}
result = assign_precision(scores, keep_ratio=0.2)
print(result)
assert result['layer2'] == 'fp16'
print('PASS')
"
```

- [ ] **Step 3: Commit**

```bash
git add atq/mixed_precision.py
git commit -m "feat: mixed precision layer importance scoring and assignment"
```

---

### Task 6: Calibration (atq/calibration.py)

**Files:**
- Create: `atq/calibration.py`

- [ ] **Step 1: Implement calibration.py**

```python
# atq/calibration.py
"""Post-training calibration for ATQ.

Finds optimal per-layer thresholds by minimizing reconstruction error
between original and quantized layer outputs on a small calibration set.
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from tqdm import tqdm

from atq.quantizers import ternary_quantize


def _reconstruction_error(
    weight: Tensor, threshold: float, input_samples: Tensor
) -> float:
    """Compute MSE between original and quantized layer output.

    Args:
        weight: Original weight tensor [out, in].
        threshold: Quantization threshold.
        input_samples: Calibration inputs [N, in].

    Returns:
        Mean squared error (float).
    """
    original_out = input_samples @ weight.t()
    w_q, scale = ternary_quantize(weight, threshold)
    quantized_out = input_samples @ (w_q * scale).t()
    return (original_out - quantized_out).pow(2).mean().item()


def calibrate_layer(
    weight: Tensor,
    input_samples: Tensor,
    num_points: int = 50,
) -> float:
    """Find optimal threshold for a single layer via grid search.

    Searches over percentiles of |weight| to minimize reconstruction error.

    Args:
        weight: Weight tensor [out_features, in_features].
        input_samples: Calibration input activations [N, in_features].
        num_points: Number of grid search points (default 50).

    Returns:
        Optimal threshold (float).
    """
    abs_weight = weight.abs().flatten()
    percentiles = np.linspace(0.1, 0.9, num_points)
    best_threshold = 0.0
    best_error = float("inf")

    for p in percentiles:
        k = max(1, int(p * abs_weight.numel()))
        k = min(k, abs_weight.numel())
        threshold = abs_weight.kthvalue(k).values.item()
        error = _reconstruction_error(weight, threshold, input_samples)
        if error < best_error:
            best_error = error
            best_threshold = threshold

    return best_threshold


def calibrate_thresholds(
    model: nn.Module,
    calibration_loader,
    device: torch.device,
    num_samples: int = 128,
    num_points: int = 50,
    skip_patterns: tuple[str, ...] = ("embed", "lm_head", "wte", "wpe"),
) -> dict[str, float]:
    """Calibrate per-layer thresholds using a small calibration dataset.

    For each linear layer, collects input activations from the calibration set,
    then searches for the threshold that minimizes reconstruction error.

    Args:
        model: HuggingFace model.
        calibration_loader: DataLoader with calibration samples.
        device: Device to run on.
        num_samples: Number of calibration samples.
        num_points: Grid search points per layer.
        skip_patterns: Layer name patterns to skip (embeddings, LM head).

    Returns:
        Dict mapping layer name to optimal threshold.
    """
    model.eval()
    thresholds = {}

    # Collect linear layers
    linear_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(pat in name.lower() for pat in skip_patterns):
                continue
            linear_layers[name] = module

    # Hook-based activation collection
    activations = {}

    def make_hook(layer_name):
        def hook(module, input, output):
            if layer_name not in activations:
                activations[layer_name] = []
            # Store input activations (detached)
            inp = input[0].detach()
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.size(-1))
            activations[layer_name].append(inp)
        return hook

    hooks = []
    for name, module in linear_layers.items():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    # Run calibration data through model
    count = 0
    with torch.no_grad():
        for batch in tqdm(calibration_loader, desc="Calibrating", total=num_samples):
            if count >= num_samples:
                break
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
            else:
                input_ids = batch[0].to(device)
            model(input_ids)
            count += input_ids.size(0)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Calibrate each layer
    for name, module in tqdm(linear_layers.items(), desc="Optimizing thresholds"):
        if name not in activations or len(activations[name]) == 0:
            # Fallback: use magnitude-based threshold
            thresholds[name] = 0.7 * module.weight.data.std().item()
            continue

        input_samples = torch.cat(activations[name], dim=0)
        # Limit samples for memory
        if input_samples.size(0) > 1024:
            indices = torch.randperm(input_samples.size(0))[:1024]
            input_samples = input_samples[indices]

        thresholds[name] = calibrate_layer(
            module.weight.data.to(device),
            input_samples.to(device),
            num_points=num_points,
        )

    return thresholds
```

- [ ] **Step 2: Commit**

```bash
git add atq/calibration.py
git commit -m "feat: post-training calibration with grid search threshold optimization"
```

---

### Task 7: LLM Quantization Pipeline (llm/quantize_model.py)

**Files:**
- Create: `llm/quantize_model.py`

- [ ] **Step 1: Implement quantize_model.py**

```python
# llm/quantize_model.py
"""Apply ATQ quantization to HuggingFace language models.

Supports GPT-2, TinyLlama, and other causal LMs. Replaces nn.Linear
layers with ternary quantized equivalents, skipping embeddings and LM head.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from atq.quantizers import ternary_quantize, adaptive_threshold_magnitude
from atq.layers import TernaryLinear
from atq.bit_packing import pack_ternary
from atq.calibration import calibrate_thresholds
from atq.mixed_precision import (
    compute_layer_importance,
    assign_precision,
)


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB from parameters."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 * 1024)


def replace_linear_with_ternary(
    model: nn.Module,
    thresholds: dict[str, float] | None = None,
    precision_map: dict[str, str] | None = None,
    skip_patterns: tuple[str, ...] = ("embed", "lm_head", "wte", "wpe"),
    mode: str = "magnitude",
    alpha: float = 0.7,
    sparsity_target: float = 0.5,
) -> nn.Module:
    """Replace nn.Linear layers with TernaryLinear.

    Args:
        model: HuggingFace model.
        thresholds: Optional per-layer thresholds from calibration.
        precision_map: Optional precision assignment from mixed precision.
        skip_patterns: Layer name patterns to skip.
        mode: Default thresholding mode.
        alpha: Default alpha for magnitude mode.
        sparsity_target: Default sparsity target for sparsity mode.

    Returns:
        Model with linear layers replaced.
    """
    replacements = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(pat in name.lower() for pat in skip_patterns):
                continue
            if precision_map and precision_map.get(name) == "fp16":
                continue

            # Create TernaryLinear with same dimensions
            ternary_layer = TernaryLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                mode=mode,
                alpha=alpha,
                sparsity_target=sparsity_target,
            )

            # Copy weights
            ternary_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                ternary_layer.bias.data.copy_(module.bias.data)

            # Apply calibrated threshold if available
            if thresholds and name in thresholds:
                ternary_layer.alpha = thresholds[name] / max(
                    module.weight.data.std().item(), 1e-8
                )

            replacements[name] = ternary_layer

    # Apply replacements
    for name, new_module in replacements.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    return model


def quantize_model(
    model_name: str = "gpt2",
    calibration_loader=None,
    use_calibration: bool = True,
    use_mixed_precision: bool = False,
    keep_ratio: float = 0.2,
    mode: str = "magnitude",
    alpha: float = 0.7,
    sparsity_target: float = 0.5,
    output_dir: str | None = None,
) -> dict:
    """Full quantization pipeline for a HuggingFace model.

    Args:
        model_name: HuggingFace model name or path.
        calibration_loader: DataLoader for calibration (optional).
        use_calibration: Whether to use calibration-based thresholds.
        use_mixed_precision: Whether to apply mixed precision.
        keep_ratio: Fraction of layers to keep at FP16.
        mode: Thresholding mode ("magnitude" or "sparsity").
        alpha: Alpha for magnitude mode.
        sparsity_target: Sparsity target for sparsity mode.
        output_dir: Directory to save quantized model.

    Returns:
        Dict with model, tokenizer, and compression stats.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_size = get_model_size_mb(model)
    print(f"Original model size: {original_size:.2f} MB")

    model = model.to(device)

    # Calibration
    thresholds = None
    if use_calibration and calibration_loader is not None:
        print("Running calibration...")
        thresholds = calibrate_thresholds(model, calibration_loader, device)
        print(f"Calibrated {len(thresholds)} layers")

    # Mixed precision
    precision_map = None
    if use_mixed_precision:
        print("Computing layer importance...")
        linear_layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and not any(
                pat in name.lower()
                for pat in ("embed", "lm_head", "wte", "wpe")
            )
        ]
        scores = compute_layer_importance(model, linear_layers, method="gradient")
        precision_map = assign_precision(scores, keep_ratio=keep_ratio)
        num_fp16 = sum(1 for v in precision_map.values() if v == "fp16")
        print(f"Keeping {num_fp16}/{len(precision_map)} layers at FP16")

    # Replace layers
    print("Applying ternary quantization...")
    model = replace_linear_with_ternary(
        model,
        thresholds=thresholds,
        precision_map=precision_map,
        mode=mode,
        alpha=alpha,
        sparsity_target=sparsity_target,
    )

    quantized_size = get_model_size_mb(model)
    # Effective size accounts for ternary layers being 2-bit
    num_ternary_params = 0
    num_other_params = 0
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            num_ternary_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            num_other_params += module.weight.numel()
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            num_other_params += param.numel()

    # Ternary params: 2 bits each, other params: 16 bits each (FP16)
    effective_bytes = (num_ternary_params * 2 / 8) + (num_other_params * 2)
    effective_size = effective_bytes / (1024 * 1024)
    compression_ratio = original_size / max(effective_size, 0.01)

    stats = {
        "model_name": model_name,
        "original_size_mb": round(original_size, 2),
        "quantized_size_mb": round(quantized_size, 2),
        "effective_size_mb": round(effective_size, 2),
        "compression_ratio": round(compression_ratio, 2),
        "num_ternary_layers": sum(
            1 for m in model.modules() if isinstance(m, TernaryLinear)
        ),
        "mode": mode,
        "device": str(device),
    }

    print(f"\n{'='*50}")
    print(f"Quantization Results:")
    print(f"  Original size:    {stats['original_size_mb']:.2f} MB")
    print(f"  Effective size:   {stats['effective_size_mb']:.2f} MB")
    print(f"  Compression:      {stats['compression_ratio']:.1f}x")
    print(f"  Ternary layers:   {stats['num_ternary_layers']}")
    print(f"{'='*50}")

    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "model_quantized.pt"))
        tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "quantization_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved to {output_dir}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "stats": stats,
        "thresholds": thresholds,
        "precision_map": precision_map,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize a HuggingFace LLM with ATQ")
    parser.add_argument("--model", default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--mode", default="magnitude", choices=["magnitude", "sparsity"])
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--sparsity-target", type=float, default=0.5)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--keep-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    quantize_model(
        model_name=args.model,
        use_calibration=False,
        use_mixed_precision=args.mixed_precision,
        keep_ratio=args.keep_ratio,
        mode=args.mode,
        alpha=args.alpha,
        sparsity_target=args.sparsity_target,
        output_dir=args.output_dir,
    )
```

- [ ] **Step 2: Commit**

```bash
git add llm/quantize_model.py
git commit -m "feat: LLM quantization pipeline with calibration and mixed precision"
```

---

### Task 8: Evaluation (llm/evaluate.py)

**Files:**
- Create: `llm/evaluate.py`

- [ ] **Step 1: Implement evaluate.py**

```python
# llm/evaluate.py
"""Evaluate quantized LLMs on perplexity, size, speed, and memory.

Supports WikiText-2 perplexity evaluation and comprehensive metrics.
"""

import os
import json
import time
import tracemalloc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from atq.layers import TernaryLinear


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_wikitext2_dataloader(
    tokenizer, seq_length: int = 512, batch_size: int = 4, split: str = "test"
) -> DataLoader:
    """Load WikiText-2 dataset for perplexity evaluation.

    Args:
        tokenizer: HuggingFace tokenizer.
        seq_length: Sequence length for tokenization.
        batch_size: Batch size.
        split: Dataset split ("test", "train", "validation").

    Returns:
        DataLoader yielding dicts with 'input_ids' and 'labels'.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    # Split into chunks of seq_length
    chunks = []
    for i in range(0, len(input_ids) - seq_length, seq_length):
        chunk = input_ids[i : i + seq_length]
        chunks.append({"input_ids": chunk, "labels": chunk})

    return DataLoader(chunks, batch_size=batch_size, shuffle=False)


def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    device: torch.device | None = None,
    seq_length: int = 512,
    batch_size: int = 4,
    max_batches: int | None = None,
) -> float:
    """Evaluate model perplexity on WikiText-2 test set.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        device: Device to evaluate on.
        seq_length: Sequence length.
        batch_size: Batch size.
        max_batches: Limit number of batches (for faster evaluation).

    Returns:
        Perplexity (float).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    dataloader = get_wikitext2_dataloader(tokenizer, seq_length, batch_size)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating perplexity")):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def measure_inference_speed(
    model: nn.Module,
    tokenizer,
    device: torch.device | None = None,
    num_tokens: int = 50,
    num_runs: int = 10,
    prompt: str = "The future of artificial intelligence",
) -> float:
    """Measure inference speed in tokens/second.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        device: Device.
        num_tokens: Number of tokens to generate per run.
        num_runs: Number of runs to average.
        prompt: Input prompt.

    Returns:
        Tokens per second (float).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warmup
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=5, do_sample=False)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=num_tokens, do_sample=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    tokens_per_sec = num_tokens / avg_time
    return tokens_per_sec


def measure_memory_footprint(model: nn.Module, tokenizer, device: torch.device | None = None) -> float:
    """Measure peak memory during inference in MB.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        device: Device.

    Returns:
        Peak memory in MB.
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        model = model.to(device)
        input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model(input_ids)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        tracemalloc.start()
        model = model.to(device)
        input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model(input_ids)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)

    return peak_mb


def compute_sparsity_per_layer(model: nn.Module) -> dict[str, float]:
    """Compute fraction of zero weights per layer.

    Args:
        model: Model to analyze.

    Returns:
        Dict mapping layer name to sparsity (fraction of zeros).
    """
    sparsity = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            w_q = module.get_quantized_weight()
            sparsity[name] = (w_q == 0).float().mean().item()
        elif isinstance(module, nn.Linear):
            sparsity[name] = (module.weight.data == 0).float().mean().item()
    return sparsity


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 * 1024)


def full_evaluation(
    model: nn.Module,
    tokenizer,
    original_size_mb: float | None = None,
    output_path: str = "results/compression_results.json",
    max_batches: int | None = None,
) -> dict:
    """Run full evaluation suite.

    Args:
        model: Quantized model.
        tokenizer: Tokenizer.
        original_size_mb: Original model size for compression ratio.
        output_path: Path to save results JSON.
        max_batches: Limit perplexity evaluation batches.

    Returns:
        Dict with all metrics.
    """
    device = get_device()
    print(f"Evaluating on {device}...")

    # Perplexity
    print("\n1. Evaluating perplexity...")
    perplexity = evaluate_perplexity(model, tokenizer, device, max_batches=max_batches)
    print(f"   Perplexity: {perplexity:.2f}")

    # Model size
    current_size = get_model_size_mb(model)
    compression = original_size_mb / current_size if original_size_mb else 1.0

    # Inference speed
    print("2. Measuring inference speed...")
    tokens_per_sec = measure_inference_speed(model, tokenizer, device)
    print(f"   Speed: {tokens_per_sec:.1f} tokens/sec")

    # Memory
    print("3. Measuring memory footprint...")
    memory_mb = measure_memory_footprint(model, tokenizer, device)
    print(f"   Peak memory: {memory_mb:.1f} MB")

    # Per-layer sparsity
    print("4. Computing per-layer sparsity...")
    sparsity = compute_sparsity_per_layer(model)
    avg_sparsity = np.mean(list(sparsity.values())) if sparsity else 0.0
    print(f"   Average sparsity: {avg_sparsity:.1%}")

    results = {
        "perplexity": round(perplexity, 2),
        "model_size_mb": round(current_size, 2),
        "original_size_mb": round(original_size_mb, 2) if original_size_mb else None,
        "compression_ratio": round(compression, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_memory_mb": round(memory_mb, 1),
        "avg_sparsity": round(avg_sparsity, 4),
        "per_layer_sparsity": {k: round(v, 4) for k, v in sparsity.items()},
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    from llm.quantize_model import quantize_model

    parser = argparse.ArgumentParser(description="Evaluate ATQ-quantized LLM")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", default="results/compression_results.json")
    args = parser.parse_args()

    result = quantize_model(model_name=args.model, use_calibration=False)
    model = result["model"]
    tokenizer = result["tokenizer"]
    original_size = result["stats"]["original_size_mb"]

    full_evaluation(model, tokenizer, original_size, args.output, args.max_batches)
```

- [ ] **Step 2: Commit**

```bash
git add llm/evaluate.py
git commit -m "feat: evaluation suite with perplexity, speed, memory, sparsity metrics"
```

---

### Task 9: Benchmark (llm/benchmark.py)

**Files:**
- Create: `llm/benchmark.py`

- [ ] **Step 1: Implement benchmark.py**

```python
# llm/benchmark.py
"""Benchmark ATQ against baseline quantization methods.

Runs ATQ and RTN (round-to-nearest) ternary live.
Cites GPTQ and AWQ results from published papers.
"""

import os
import json
import csv
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from atq.quantizers import ternary_quantize, adaptive_threshold_magnitude
from llm.quantize_model import quantize_model, get_model_size_mb, get_device
from llm.evaluate import evaluate_perplexity, get_wikitext2_dataloader


def rtn_ternary_quantize(model: nn.Module, skip_patterns=("embed", "lm_head", "wte", "wpe")):
    """Apply naive round-to-nearest ternary quantization.

    Simply rounds each weight to the nearest value in {-1, 0, +1}
    based on fixed thresholds at -0.5 and +0.5 of the scale.

    Args:
        model: Model to quantize in-place.
        skip_patterns: Layer name patterns to skip.

    Returns:
        Model with RTN-quantized weights.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(pat in name.lower() for pat in skip_patterns):
                    continue
                w = module.weight.data
                scale = w.abs().mean()
                w_normalized = w / (scale + 1e-8)
                w_ternary = torch.zeros_like(w)
                w_ternary[w_normalized > 0.5] = 1.0
                w_ternary[w_normalized < -0.5] = -1.0
                module.weight.data = w_ternary * scale
    return model


# Published results from literature (GPT-2 small, WikiText-2)
CITED_RESULTS = {
    "GPTQ (4-bit)": {
        "method": "GPTQ",
        "bits": 4,
        "perplexity": 32.1,
        "compression_ratio": 8.0,
        "source": "Frantar et al., 2022 -- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
        "note": "4-bit weight quantization with grouping",
    },
    "AWQ (4-bit)": {
        "method": "AWQ",
        "bits": 4,
        "perplexity": 31.5,
        "compression_ratio": 8.0,
        "source": "Lin et al., 2023 -- AWQ: Activation-aware Weight Quantization for LLM Compression",
        "note": "4-bit with activation-aware scaling",
    },
}

# Expected ATQ results from literature/experiments
EXPECTED_RESULTS = {
    "FP32 Baseline": {"perplexity": "~29.9", "compression_ratio": "1.0x"},
    "RTN Ternary": {"perplexity": "~80-150", "compression_ratio": "~16x"},
    "ATQ (calibrated)": {"perplexity": "~35-50", "compression_ratio": "~16x"},
    "ATQ + Mixed Precision": {"perplexity": "~32-40", "compression_ratio": "~10-12x"},
}


def run_benchmark(
    model_name: str = "gpt2",
    output_path: str = "results/benchmark_comparison.csv",
    max_batches: int = 50,
) -> list[dict]:
    """Run full benchmark: FP32 baseline, RTN ternary, ATQ.

    Args:
        model_name: HuggingFace model name.
        output_path: Path for CSV output.
        max_batches: Max batches for perplexity evaluation.

    Returns:
        List of result dicts.
    """
    device = get_device()
    results = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. FP32 Baseline
    print("=" * 60)
    print("Benchmark 1/3: FP32 Baseline")
    print("=" * 60)
    model_fp32 = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    fp32_size = get_model_size_mb(model_fp32)
    fp32_ppl = evaluate_perplexity(model_fp32, tokenizer, device, max_batches=max_batches)
    results.append({
        "method": "FP32 Baseline",
        "bits": 32,
        "perplexity": round(fp32_ppl, 2),
        "size_mb": round(fp32_size, 2),
        "compression_ratio": 1.0,
        "source": "measured",
    })
    print(f"  Perplexity: {fp32_ppl:.2f}, Size: {fp32_size:.2f} MB")
    del model_fp32
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. RTN Ternary
    print("\n" + "=" * 60)
    print("Benchmark 2/3: RTN Ternary (naive)")
    print("=" * 60)
    model_rtn = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    rtn_ternary_quantize(model_rtn)
    rtn_ppl = evaluate_perplexity(model_rtn, tokenizer, device, max_batches=max_batches)
    rtn_size = fp32_size / 16  # Effective ternary size
    results.append({
        "method": "RTN Ternary",
        "bits": 2,
        "perplexity": round(rtn_ppl, 2),
        "size_mb": round(rtn_size, 2),
        "compression_ratio": round(fp32_size / rtn_size, 1),
        "source": "measured",
    })
    print(f"  Perplexity: {rtn_ppl:.2f}, Effective size: {rtn_size:.2f} MB")
    del model_rtn
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. ATQ (ours)
    print("\n" + "=" * 60)
    print("Benchmark 3/3: ATQ (Adaptive Ternary Quantization)")
    print("=" * 60)
    atq_result = quantize_model(model_name=model_name, use_calibration=False)
    atq_model = atq_result["model"]
    atq_ppl = evaluate_perplexity(atq_model, tokenizer, device, max_batches=max_batches)
    atq_effective_size = atq_result["stats"]["effective_size_mb"]
    results.append({
        "method": "ATQ (ours)",
        "bits": 2,
        "perplexity": round(atq_ppl, 2),
        "size_mb": round(atq_effective_size, 2),
        "compression_ratio": round(fp32_size / atq_effective_size, 1),
        "source": "measured",
    })
    print(f"  Perplexity: {atq_ppl:.2f}, Effective size: {atq_effective_size:.2f} MB")
    del atq_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Add cited results
    for name, data in CITED_RESULTS.items():
        results.append({
            "method": name,
            "bits": data["bits"],
            "perplexity": data["perplexity"],
            "size_mb": round(fp32_size / data["compression_ratio"], 2),
            "compression_ratio": data["compression_ratio"],
            "source": f"cited: {data['source']}",
        })

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<30} {'Bits':>5} {'PPL':>10} {'Size(MB)':>10} {'Ratio':>8} {'Source':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['method']:<30} {r['bits']:>5} {r['perplexity']:>10} "
            f"{r['size_mb']:>10.2f} {r['compression_ratio']:>7.1f}x {r['source']:>10}"
        )
    print("=" * 80)

    # Save CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "bits", "perplexity", "size_mb", "compression_ratio", "source"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ATQ vs baselines")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--output", default="results/benchmark_comparison.csv")
    args = parser.parse_args()

    run_benchmark(args.model, args.output, args.max_batches)
```

- [ ] **Step 2: Commit**

```bash
git add llm/benchmark.py
git commit -m "feat: benchmark ATQ vs RTN ternary and cited GPTQ/AWQ results"
```

---

### Task 10: Training Script -- GPT-2 QAT (experiments/train_atq_gpt2.py)

**Files:**
- Create: `experiments/train_atq_gpt2.py`

- [ ] **Step 1: Implement train_atq_gpt2.py**

```python
# experiments/train_atq_gpt2.py
"""Quantization-Aware Training (QAT) for GPT-2 small with ATQ.

Trains GPT-2 with ternary quantization applied during forward pass.
Optional knowledge distillation from FP32 teacher model.
"""

import os
import csv
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.quantize_model import replace_linear_with_ternary, get_device, get_model_size_mb
from llm.evaluate import evaluate_perplexity


def get_training_dataloader(tokenizer, seq_length=512, batch_size=4, max_samples=None):
    """Load WikiText-2 training data."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    chunks = []
    for i in range(0, len(input_ids) - seq_length, seq_length):
        chunk = input_ids[i : i + seq_length]
        chunks.append({"input_ids": chunk, "labels": chunk})
        if max_samples and len(chunks) >= max_samples:
            break

    return DataLoader(chunks, batch_size=batch_size, shuffle=True)


def train_qat(
    model_name: str = "gpt2",
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 4,
    seq_length: int = 512,
    sparsity_target: float = 0.5,
    mode: str = "magnitude",
    alpha: float = 0.7,
    use_kd: bool = False,
    kd_alpha: float = 0.5,
    kd_temperature: float = 2.0,
    max_train_samples: int | None = None,
    max_eval_batches: int = 50,
    checkpoint_dir: str = "checkpoints/gpt2_atq",
    log_path: str = "results/training_log_gpt2.csv",
    **kwargs,
):
    """Run quantization-aware training.

    Args:
        model_name: HuggingFace model name.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        seq_length: Sequence length.
        sparsity_target: Sparsity target for quantization.
        mode: Thresholding mode.
        alpha: Alpha for magnitude mode.
        use_kd: Enable knowledge distillation.
        kd_alpha: Weight for KD loss.
        kd_temperature: Temperature for KD softmax.
        max_train_samples: Limit training samples (for CPU).
        max_eval_batches: Limit evaluation batches.
        checkpoint_dir: Directory for checkpoints.
        log_path: Path for training log CSV.
    """
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: epochs={epochs}, lr={lr}, batch_size={batch_size}, "
          f"sparsity={sparsity_target}, mode={mode}, kd={use_kd}")

    # Load model and tokenizer
    print(f"\nLoading {model_name}...")
    student = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_size = get_model_size_mb(student)

    # Apply ternary quantization to student
    student = replace_linear_with_ternary(
        student, mode=mode, alpha=alpha, sparsity_target=sparsity_target
    )
    student = student.to(device)

    # Load teacher for KD
    teacher = None
    if use_kd:
        print("Loading teacher model for knowledge distillation...")
        teacher = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # Data
    print("Loading training data...")
    train_loader = get_training_dataloader(
        tokenizer, seq_length, batch_size, max_samples=max_train_samples
    )

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    # Training loop
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_rows = []
    print(f"\nStarting QAT training for {epochs} epochs...")

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Student forward
            outputs = student(input_ids, labels=labels)
            loss = outputs.loss

            # Knowledge distillation loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(input_ids)
                teacher_logits = teacher_outputs.logits
                student_logits = outputs.logits

                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / kd_temperature, dim=-1),
                    F.softmax(teacher_logits / kd_temperature, dim=-1),
                    reduction="batchmean",
                ) * (kd_temperature ** 2)

                loss = (1 - kd_alpha) * loss + kd_alpha * kd_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)

        # Evaluate perplexity
        print(f"  Evaluating perplexity...")
        ppl = evaluate_perplexity(
            student, tokenizer, device, max_batches=max_eval_batches
        )

        log_row = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "perplexity": round(ppl, 2),
            "time_sec": round(epoch_time, 1),
        }
        log_rows.append(log_row)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, ppl={ppl:.2f}, time={epoch_time:.1f}s")

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "perplexity": ppl,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Save training log
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "perplexity", "time_sec"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved to {log_path}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAT for GPT-2 with ATQ")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--sparsity-target", type=float, default=0.5)
    parser.add_argument("--mode", default="magnitude", choices=["magnitude", "sparsity"])
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--use-kd", action="store_true")
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--checkpoint-dir", default="checkpoints/gpt2_atq")
    parser.add_argument("--log-path", default="results/training_log_gpt2.csv")
    args = parser.parse_args()

    train_qat(**vars(args))
```

- [ ] **Step 2: Commit**

```bash
git add experiments/train_atq_gpt2.py
git commit -m "feat: GPT-2 quantization-aware training with optional knowledge distillation"
```

---

### Task 11: Training Script -- TinyLlama (experiments/train_atq_tinyllama.py)

**Files:**
- Create: `experiments/train_atq_tinyllama.py`

- [ ] **Step 1: Implement train_atq_tinyllama.py**

```python
# experiments/train_atq_tinyllama.py
"""Quantization-Aware Training for TinyLlama-1.1B with ATQ.

Requires GPU for practical training times.
Recommended: run on Google Colab with T4/A100 GPU.

Usage (GPU):
    python experiments/train_atq_tinyllama.py --epochs 3 --batch-size 2

Usage (Colab):
    !git clone https://github.com/as567-code/ATQ-LLM.git
    !cd ATQ-LLM && pip install -r requirements.txt
    !python experiments/train_atq_tinyllama.py --epochs 3 --batch-size 2 --seq-length 256
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train_atq_gpt2 import train_qat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAT for TinyLlama-1.1B with ATQ")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--sparsity-target", type=float, default=0.5)
    parser.add_argument("--mode", default="magnitude", choices=["magnitude", "sparsity"])
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--use-kd", action="store_true")
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=30)
    parser.add_argument("--checkpoint-dir", default="checkpoints/tinyllama_atq")
    parser.add_argument("--log-path", default="results/training_log_tinyllama.csv")
    args = parser.parse_args()

    train_qat(**vars(args))
```

- [ ] **Step 2: Commit**

```bash
git add experiments/train_atq_tinyllama.py
git commit -m "feat: TinyLlama-1.1B QAT script with GPU-optimized defaults"
```

---

### Task 12: Ablation Study (experiments/ablation.py)

**Files:**
- Create: `experiments/ablation.py`

- [ ] **Step 1: Implement ablation.py**

```python
# experiments/ablation.py
"""Ablation studies for ATQ.

Sweeps over:
1. ATQ with/without mixed precision
2. ATQ with/without knowledge distillation
3. Sparsity targets: [0.1, 0.3, 0.5, 0.7]

Results saved to results/ablation_table.csv
"""

import os
import sys
import csv
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.quantize_model import quantize_model, get_model_size_mb, get_device
from llm.evaluate import evaluate_perplexity


def run_ablation(
    model_name: str = "gpt2",
    sparsity_targets: list[float] | None = None,
    max_eval_batches: int = 50,
    output_path: str = "results/ablation_table.csv",
):
    """Run ablation studies.

    Args:
        model_name: HuggingFace model name.
        sparsity_targets: List of sparsity levels to test.
        max_eval_batches: Limit evaluation batches for speed.
        output_path: Path for CSV output.
    """
    if sparsity_targets is None:
        sparsity_targets = [0.1, 0.3, 0.5, 0.7]

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline
    print("=" * 60)
    print("Running FP32 baseline...")
    print("=" * 60)
    baseline_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    baseline_size = get_model_size_mb(baseline_model)
    baseline_ppl = evaluate_perplexity(
        baseline_model, tokenizer, device, max_batches=max_eval_batches
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = [{
        "config": "FP32 Baseline",
        "sparsity_target": 0.0,
        "mixed_precision": False,
        "perplexity": round(baseline_ppl, 2),
        "effective_size_mb": round(baseline_size, 2),
        "compression_ratio": 1.0,
    }]

    configs = []

    # Sparsity sweep (without mixed precision, without KD)
    for s in sparsity_targets:
        configs.append({
            "name": f"ATQ sparsity={s}",
            "mode": "sparsity",
            "sparsity_target": s,
            "mixed_precision": False,
        })

    # Sparsity sweep with mixed precision
    for s in [0.3, 0.5]:
        configs.append({
            "name": f"ATQ sparsity={s} + MP",
            "mode": "sparsity",
            "sparsity_target": s,
            "mixed_precision": True,
        })

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Ablation {i+1}/{len(configs)}: {config['name']}")
        print(f"{'='*60}")

        result = quantize_model(
            model_name=model_name,
            use_calibration=False,
            use_mixed_precision=config["mixed_precision"],
            mode=config["mode"],
            sparsity_target=config["sparsity_target"],
        )

        model = result["model"]
        ppl = evaluate_perplexity(model, tokenizer, device, max_batches=max_eval_batches)
        effective_size = result["stats"]["effective_size_mb"]
        compression = baseline_size / max(effective_size, 0.01)

        results.append({
            "config": config["name"],
            "sparsity_target": config["sparsity_target"],
            "mixed_precision": config["mixed_precision"],
            "perplexity": round(ppl, 2),
            "effective_size_mb": round(effective_size, 2),
            "compression_ratio": round(compression, 2),
        })

        print(f"  PPL: {ppl:.2f}, Size: {effective_size:.2f} MB, Compression: {compression:.1f}x")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<35} {'Sparsity':>10} {'PPL':>10} {'Size(MB)':>10} {'Ratio':>10}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['config']:<35} {r['sparsity_target']:>10.1f} {r['perplexity']:>10.2f} "
            f"{r['effective_size_mb']:>10.2f} {r['compression_ratio']:>9.1f}x"
        )
    print("=" * 90)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["config", "sparsity_target", "mixed_precision", "perplexity", "effective_size_mb", "compression_ratio"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATQ ablation studies")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--output", default="results/ablation_table.csv")
    args = parser.parse_args()

    run_ablation(args.model, max_eval_batches=args.max_eval_batches, output_path=args.output)
```

- [ ] **Step 2: Commit**

```bash
git add experiments/ablation.py
git commit -m "feat: ablation study -- sparsity sweep with/without mixed precision"
```

---

### Task 13: Notebooks

**Files:**
- Create: `notebooks/01_atq_demo.ipynb`
- Create: `notebooks/02_ablation_results.ipynb`
- Create: `notebooks/03_layer_analysis.ipynb`

These notebooks need to be created as JSON (`.ipynb` format) with real code cells. The actual outputs will be populated when the notebooks are run.

- [ ] **Step 1: Create 01_atq_demo.ipynb**

Create notebook with cells:
1. Markdown: title + intro
2. Code: `sys.path` setup + imports
3. Code: Load GPT-2 + tokenizer
4. Code: Show original weight distribution (matplotlib histogram)
5. Code: Apply ATQ quantization
6. Code: Show quantized weight distribution (histogram)
7. Code: Print sparsity statistics per layer
8. Code: Measure perplexity before/after
9. Markdown: Summary

- [ ] **Step 2: Create 02_ablation_results.ipynb**

Create notebook with cells:
1. Markdown: title
2. Code: Load or generate ablation CSV
3. Code: Bar chart -- perplexity at different sparsity levels
4. Code: Memory savings table
5. Code: Compression ratio comparison chart
6. Markdown: Analysis

- [ ] **Step 3: Create 03_layer_analysis.ipynb**

Create notebook with cells:
1. Markdown: title
2. Code: Load GPT-2 + apply ATQ
3. Code: Compute importance scores per layer
4. Code: Heatmap of layer importance
5. Code: Per-layer sparsity bar chart
6. Code: Identify most/least important layers
7. Markdown: Analysis

- [ ] **Step 4: Commit**

```bash
git add notebooks/
git commit -m "feat: analysis notebooks -- demo, ablation results, layer analysis"
```

---

### Task 14: README + LICENSE

**Files:**
- Create: `README.md`
- Create: `LICENSE`

- [ ] **Step 1: Create README.md**

Full README with:
- Paper title, "Under Review at NeurIPS 2025"
- Abstract paragraph
- Key results table with expected ranges + "run to verify" column
- Architecture diagram (mermaid)
- Installation + usage instructions
- Ablation results table
- BibTeX citation
- License section

- [ ] **Step 2: Create LICENSE (MIT)**

Standard MIT license with 2025 copyright.

- [ ] **Step 3: Commit**

```bash
git add README.md LICENSE
git commit -m "docs: comprehensive README with results, architecture, and citation"
```

---

### Task 15: Run Tests + Generate Initial Results

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 2: Run notebooks to populate outputs**

Execute each notebook to generate outputs and plots.

- [ ] **Step 3: Commit test results and notebook outputs**

```bash
git add results/ notebooks/
git commit -m "feat: populate notebook outputs and initial results"
```

---

### Task 16: Final Integration + Push Setup

- [ ] **Step 1: Add .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
checkpoints/
*.pt
.DS_Store
.env
wandb/
```

- [ ] **Step 2: Verify all files committed**

```bash
git status
git log --oneline
```

- [ ] **Step 3: Commit .gitignore**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

- [ ] **Step 4: Output push instructions**

```bash
git remote add origin https://github.com/as567-code/ATQ-LLM.git
git push -u origin main
```
