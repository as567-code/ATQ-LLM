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
