# experiments/ablation.py
"""Ablation studies for ATQ.

Sweeps over:
1. ATQ with/without mixed precision
2. Sparsity targets: [0.1, 0.3, 0.5, 0.7]

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
    """Run ablation studies."""
    if sparsity_targets is None:
        sparsity_targets = [0.1, 0.3, 0.5, 0.7]

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    for s in sparsity_targets:
        configs.append({
            "name": f"ATQ sparsity={s}",
            "mode": "sparsity",
            "sparsity_target": s,
            "mixed_precision": False,
        })

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

    print("\n" + "=" * 90)
    print(f"{'Config':<35} {'Sparsity':>10} {'PPL':>10} {'Size(MB)':>10} {'Ratio':>10}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['config']:<35} {r['sparsity_target']:>10.1f} {r['perplexity']:>10.2f} "
            f"{r['effective_size_mb']:>10.2f} {r['compression_ratio']:>9.1f}x"
        )
    print("=" * 90)

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
