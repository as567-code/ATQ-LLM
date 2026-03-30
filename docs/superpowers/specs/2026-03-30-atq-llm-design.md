# ATQ-LLM Design Spec

**Date:** 2026-03-30
**Author:** Aditya Swaroop
**Status:** Approved
**Context:** NeurIPS 2025 submission — "Adaptive Ternary Quantization for On-Device LLM Compression"

## Overview

A research repository implementing Adaptive Ternary Quantization (ATQ) specifically for LLM compression. Differentiates from co-author's ATQ-Multimodal (Fashion-MNIST + Flickr8k) by focusing on language models (GPT-2, TinyLlama) with perplexity-based evaluation, matching the paper title.

## Architecture

```
ATQ-LLM/
├── atq/                   # Core engine (model-agnostic)
│   ├── __init__.py
│   ├── quantizers.py       # Ternary quantization with adaptive thresholds
│   ├── layers.py           # TernaryLinear (drop-in nn.Linear replacement)
│   ├── bit_packing.py      # 2-bit packing (16x compression vs FP32)
│   ├── mixed_precision.py  # Layer importance scoring for precision assignment
│   └── calibration.py      # Post-training threshold calibration
├── llm/                   # HuggingFace model pipeline
│   ├── __init__.py
│   ├── quantize_model.py   # Apply ATQ to GPT-2 / TinyLlama layer-by-layer
│   ├── evaluate.py         # Perplexity, size, speed, memory, sparsity metrics
│   └── benchmark.py        # ATQ vs RTN ternary vs GPTQ/AWQ (cited)
├── experiments/           # Training & ablation scripts
│   ├── train_atq_gpt2.py   # QAT for GPT-2 small with optional KD
│   ├── train_atq_tinyllama.py  # QAT for TinyLlama-1.1B
│   └── ablation.py         # Sweep: sparsity × mixed precision × KD
├── notebooks/             # 3 Jupyter notebooks with outputs
│   ├── 01_atq_demo.ipynb
│   ├── 02_ablation_results.ipynb
│   └── 03_layer_analysis.ipynb
├── results/               # Generated plots, CSVs, JSON
├── tests/                 # pytest suite
│   ├── test_quantizers.py
│   ├── test_layers.py
│   └── test_bit_packing.py
├── requirements.txt
├── README.md
└── LICENSE (MIT)
```

## Core ATQ Engine (`atq/`)

### quantizers.py — Adaptive Ternary Quantization

Two thresholding modes:

1. **Magnitude-based:** threshold = `α * std(W)` where α is a learnable per-layer parameter (default α=0.7). Weights above threshold → +1, below −threshold → −1, between → 0.

2. **Sparsity-target-based:** Given target sparsity `s`, find threshold as the `s`-th percentile of `|W|`. This directly controls the fraction of zero weights.

Both modes compute a scale factor: `Δ = mean(|W|[W≠0])` to preserve the magnitude distribution of non-zero weights.

Quantized output: `W_q = Δ * ternary_sign(W, threshold)`

### layers.py — TernaryLinear

Drop-in replacement for `nn.Linear`:
- Forward pass: quantize weights using ATQ, compute `F.linear(x, W_q, bias)`
- Backward pass: Straight-Through Estimator (STE) — gradients flow through the quantization as identity
- Scale factor Δ is also a learned parameter
- Supports both thresholding modes via constructor arg

### bit_packing.py — 2-bit Representation

Mapping: {-1, 0, +1} → {0, 1, 2}
Pack 4 ternary values per uint8 byte → 2 bits per weight → 16× compression vs FP32.

Functions:
- `pack_ternary(tensor) → packed_bytes, shape, scale`
- `unpack_ternary(packed_bytes, shape, scale) → tensor`
- Round-trip must be lossless (verified by tests)

### mixed_precision.py — Layer Importance Scoring

Two scoring methods:
1. **Gradient-based:** `importance_i = ||∇W_i ⊙ W_i||_F` (gradient × weight Frobenius norm)
2. **Fisher Information:** `importance_i = E[||∇W_i log p(y|x)||²]` (diagonal Fisher approximation)

Assignment:
- Layers below median importance → full ternary (2-bit)
- Top-k layers (configurable ratio, default 20%) → keep FP16
- Returns a precision map: `{layer_name: "ternary" | "fp16"}`

### calibration.py — Post-Training Calibration

- Take a small calibration set (~128 samples from WikiText-2)
- Run forward pass through each layer, collecting input/output activations
- For each layer, optimize threshold to minimize `||output_original - output_quantized||²`
- Uses scipy.optimize or grid search over percentile range [0.1, 0.9]
- Returns optimized per-layer thresholds

## LLM Pipeline (`llm/`)

### quantize_model.py

- Load HuggingFace model (GPT-2 small, TinyLlama-1.1B)
- Replace `nn.Linear` layers with `TernaryLinear` (skip embedding and LM head)
- Apply calibration to find optimal thresholds
- Optionally apply mixed precision (keep critical layers at higher precision)
- Save quantized model (both PyTorch state dict and bit-packed format)
- Report: original size, quantized size, compression ratio

### evaluate.py

Metrics:
- **Perplexity** on WikiText-2 (standard LLM benchmark)
- **Model size** (MB) before and after
- **Compression ratio**
- **Inference speed** (tokens/sec, averaged over 100 forward passes)
- **Memory footprint** during inference (torch.cuda.max_memory_allocated or tracemalloc)
- **Per-layer sparsity distribution** (fraction of zeros per layer)

Output: `results/compression_results.json` + console summary

### benchmark.py

Live comparisons:
- **ATQ** (our method) — run live
- **RTN Ternary** (naive round-to-nearest) — implemented and run live

Cited comparisons (from published papers):
- **GPTQ** — cite Frantar et al. 2022 numbers
- **AWQ** — cite Lin et al. 2023 numbers

Output: comparison table in console + `results/benchmark_comparison.csv`

## Experiments

### train_atq_gpt2.py — QAT for GPT-2 Small

- Quantization-aware training with STE
- Optional KD: `--use-kd` flag adds `α * KL(teacher_logits, student_logits)` to loss (teacher = frozen FP32 GPT-2)
- Configurable: `--sparsity-target`, `--lr`, `--epochs`, `--batch-size`
- Logging: CSV file (W&B optional if installed)
- Perplexity tracked per epoch
- Checkpoints saved per epoch
- CPU-friendly defaults (small batch, few epochs); GPU configs in README

### train_atq_tinyllama.py — QAT for TinyLlama-1.1B

Same structure as GPT-2 script. Requires GPU for practical training times.
Includes clear instructions for running on Colab with recommended configs.

### ablation.py — Ablation Studies

Sweeps:
1. ATQ with vs without mixed precision
2. ATQ with vs without knowledge distillation
3. Sparsity targets: [0.1, 0.3, 0.5, 0.7]

Each combination: quantize, evaluate perplexity, log to `results/ablation_table.csv`.
Generate comparison table to stdout.

## Notebooks

### 01_atq_demo.ipynb
Walkthrough: load GPT-2 → apply ATQ → show weight distribution before/after (matplotlib histograms) → show sparsity stats → measure perplexity. Interactive and visual.

### 02_ablation_results.ipynb
Load `results/ablation_table.csv` → bar charts comparing perplexity at different sparsity levels → memory savings table → inference speed comparison. If CSV doesn't exist, run a minimal ablation inline.

### 03_layer_analysis.ipynb
Per-layer analysis: compute importance scores → heatmap of layer importance → which layers tolerate aggressive quantization → per-layer sparsity distribution bar chart.

## Results & Benchmark Numbers Strategy

**Dual approach:**
- Include expected ranges from published ternary quantization literature in README tables
- Code populates `results/` with actual numbers when run
- README tables show: `Expected: X.X | Actual: [run experiments/ablation.py]`

**Expected baselines (from literature):**
- GPT-2 small FP32 WikiText-2 perplexity: ~29-30
- Ternary RTN: ~80-150 (significant degradation)
- ATQ (calibrated): ~35-50 depending on sparsity target
- ATQ + mixed precision: ~32-40
- ATQ + KD + mixed precision: ~31-36

## Device Handling

All code auto-detects: CUDA → MPS → CPU
- Default configs: CPU-friendly (128 calibration samples, batch_size=4, 3 epochs)
- GPU configs documented: larger batches, more epochs, TinyLlama feasible

## Differentiation from ak736/ATQ-Multimodal

| Aspect | ak736/ATQ-Multimodal | ATQ-LLM (this repo) |
|--------|---------------------|----------------------|
| Domain | Fashion-MNIST + Flickr8k | GPT-2, TinyLlama LLMs |
| Task | Image classification + multimodal | Language modeling |
| Metric | Classification accuracy | Perplexity (WikiText-2) |
| Focus | Multimodal fusion | On-device LLM compression |
| Techniques | Basic ternary quantization | Adaptive thresholds + mixed precision + KD + calibration |
| Paper alignment | Multimodal extension | Matches paper title directly |

## Testing

pytest suite covering:
- `test_quantizers.py` — quantized weights ∈ {-1, 0, +1}, sparsity targets approximately met, scale factors correct
- `test_layers.py` — TernaryLinear forward/backward shapes correct, gradients flow (STE works)
- `test_bit_packing.py` — pack/unpack round-trip is lossless, compression ratio is 16×

## Git Strategy

Meaningful incremental commits:
1. Project skeleton + requirements
2. Core ATQ engine (quantizers, layers, bit_packing)
3. Mixed precision + calibration
4. LLM pipeline (quantize, evaluate, benchmark)
5. Training scripts + ablation
6. Tests
7. Notebooks with outputs
8. README + LICENSE
9. Results and plots
