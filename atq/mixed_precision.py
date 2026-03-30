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

    for name in scores:
        scores[name] /= max(count, 1)

    return scores


def compute_layer_importance(
    model: nn.Module,
    named_linear_layers: list[tuple[str, nn.Module]],
    method: str = "gradient",
    **kwargs,
) -> dict[str, float]:
    """Compute layer importance scores. method: 'gradient' or 'fisher'."""
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

    Top keep_ratio fraction of layers keep FP16, rest get ternary.
    """
    if not importance_scores:
        return {}

    sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    num_keep = max(1, int(len(sorted_layers) * keep_ratio))

    precision_map = {}
    for i, (name, _) in enumerate(sorted_layers):
        precision_map[name] = "fp16" if i < num_keep else "ternary"

    return precision_map
