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

# HuggingFace GPT-2 uses Conv1D instead of nn.Linear
try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None


def _is_linear_layer(module: nn.Module) -> bool:
    """Check if module is a linear layer (nn.Linear or HF Conv1D)."""
    if isinstance(module, nn.Linear):
        return True
    if HFConv1D is not None and isinstance(module, HFConv1D):
        return True
    return False


def _get_linear_dims(module: nn.Module) -> tuple[int, int]:
    """Get (in_features, out_features) from a linear layer.

    HF Conv1D stores weight as [in, out], nn.Linear as [out, in].
    """
    if HFConv1D is not None and isinstance(module, HFConv1D):
        return module.weight.shape[0], module.weight.shape[1]
    return module.in_features, module.out_features


def _get_weight_for_linear(module: nn.Module) -> torch.Tensor:
    """Get weight in [out, in] format regardless of layer type."""
    if HFConv1D is not None and isinstance(module, HFConv1D):
        return module.weight.data.t()  # Conv1D stores [in, out]
    return module.weight.data


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
        if _is_linear_layer(module):
            if any(pat in name.lower() for pat in skip_patterns):
                continue
            if precision_map and precision_map.get(name) == "fp16":
                continue

            in_feat, out_feat = _get_linear_dims(module)
            ternary_layer = TernaryLinear(
                in_feat,
                out_feat,
                bias=module.bias is not None,
                mode=mode,
                alpha=alpha,
                sparsity_target=sparsity_target,
            )

            # Copy weights (Conv1D stores [in, out], we need [out, in])
            w = _get_weight_for_linear(module)
            ternary_layer.weight.data.copy_(w)
            if module.bias is not None:
                ternary_layer.bias.data.copy_(module.bias.data)

            # Initialize scale to correct magnitude for post-training quantization
            # Scale = mean(|W|) for weights that will be non-zero after quantization
            from atq.quantizers import compute_scale_factor, ternary_quantize as _tq
            threshold = ternary_layer._compute_threshold()
            w_q, scale = _tq(w, threshold)
            ternary_layer.scale.data.fill_(scale)

            if thresholds and name in thresholds:
                ternary_layer.alpha = thresholds[name] / max(
                    w.std().item(), 1e-8
                )
                # Recompute scale with calibrated threshold
                w_q, scale = _tq(w, thresholds[name])
                ternary_layer.scale.data.fill_(scale)

            replacements[name] = ternary_layer

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
    """Full quantization pipeline for a HuggingFace model."""
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_size = get_model_size_mb(model)
    print(f"Original model size: {original_size:.2f} MB")

    model = model.to(device)

    thresholds = None
    if use_calibration and calibration_loader is not None:
        print("Running calibration...")
        thresholds = calibrate_thresholds(model, calibration_loader, device)
        print(f"Calibrated {len(thresholds)} layers")

    precision_map = None
    if use_mixed_precision:
        print("Computing layer importance...")
        linear_layers = [
            (name, module)
            for name, module in model.named_modules()
            if _is_linear_layer(module)
            and not any(
                pat in name.lower()
                for pat in ("embed", "lm_head", "wte", "wpe")
            )
        ]
        scores = compute_layer_importance(model, linear_layers, method="gradient")
        precision_map = assign_precision(scores, keep_ratio=keep_ratio)
        num_fp16 = sum(1 for v in precision_map.values() if v == "fp16")
        print(f"Keeping {num_fp16}/{len(precision_map)} layers at FP16")

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
    num_ternary_params = 0
    num_other_params = 0
    counted_params = set()
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            num_ternary_params += module.weight.numel()
            counted_params.add(id(module.weight))
        elif _is_linear_layer(module):
            num_other_params += module.weight.numel()
            counted_params.add(id(module.weight))
    for name, param in model.named_parameters():
        if id(param) not in counted_params:
            num_other_params += param.numel()

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
