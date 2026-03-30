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

            ternary_layer = TernaryLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                mode=mode,
                alpha=alpha,
                sparsity_target=sparsity_target,
            )

            ternary_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                ternary_layer.bias.data.copy_(module.bias.data)

            if thresholds and name in thresholds:
                ternary_layer.alpha = thresholds[name] / max(
                    module.weight.data.std().item(), 1e-8
                )

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
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            num_ternary_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            num_other_params += module.weight.numel()
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
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
