# llm/evaluate.py
"""Evaluate quantized LLMs on perplexity, size, speed, and memory."""

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
    """Load WikiText-2 dataset for perplexity evaluation."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

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
    """Evaluate model perplexity on WikiText-2 test set."""
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
    """Measure inference speed in tokens/second."""
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

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
    """Measure peak memory during inference in MB."""
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
    """Compute fraction of zero weights per layer."""
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
    """Run full evaluation suite."""
    device = get_device()
    print(f"Evaluating on {device}...")

    print("\n1. Evaluating perplexity...")
    perplexity = evaluate_perplexity(model, tokenizer, device, max_batches=max_batches)
    print(f"   Perplexity: {perplexity:.2f}")

    current_size = get_model_size_mb(model)
    compression = original_size_mb / current_size if original_size_mb else 1.0

    print("2. Measuring inference speed...")
    tokens_per_sec = measure_inference_speed(model, tokenizer, device)
    print(f"   Speed: {tokens_per_sec:.1f} tokens/sec")

    print("3. Measuring memory footprint...")
    memory_mb = measure_memory_footprint(model, tokenizer, device)
    print(f"   Peak memory: {memory_mb:.1f} MB")

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
