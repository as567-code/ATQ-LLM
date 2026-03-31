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

try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None


def _is_linear_layer(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    if HFConv1D is not None and isinstance(module, HFConv1D):
        return True
    return False


def _reconstruction_error(
    weight: Tensor, threshold: float, input_samples: Tensor
) -> float:
    """Compute MSE between original and quantized layer output."""
    original_out = input_samples @ weight.t()
    w_q, scale = ternary_quantize(weight, threshold)
    quantized_out = input_samples @ (w_q * scale).t()
    return (original_out - quantized_out).pow(2).mean().item()


def calibrate_layer(
    weight: Tensor,
    input_samples: Tensor,
    num_points: int = 50,
) -> float:
    """Find optimal threshold for a single layer via grid search."""
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
    """Calibrate per-layer thresholds using a small calibration dataset."""
    model.eval()
    thresholds = {}

    linear_layers = {}
    for name, module in model.named_modules():
        if _is_linear_layer(module):
            if any(pat in name.lower() for pat in skip_patterns):
                continue
            linear_layers[name] = module

    activations = {}

    def make_hook(layer_name):
        def hook(module, input, output):
            if layer_name not in activations:
                activations[layer_name] = []
            inp = input[0].detach()
            if inp.dim() == 3:
                inp = inp.reshape(-1, inp.size(-1))
            activations[layer_name].append(inp)
        return hook

    hooks = []
    for name, module in linear_layers.items():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

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

    for h in hooks:
        h.remove()

    for name, module in tqdm(linear_layers.items(), desc="Optimizing thresholds"):
        if name not in activations or len(activations[name]) == 0:
            thresholds[name] = 0.7 * module.weight.data.std().item()
            continue

        input_samples = torch.cat(activations[name], dim=0)
        if input_samples.size(0) > 1024:
            indices = torch.randperm(input_samples.size(0))[:1024]
            input_samples = input_samples[indices]

        thresholds[name] = calibrate_layer(
            module.weight.data.to(device),
            input_samples.to(device),
            num_points=num_points,
        )

    return thresholds
