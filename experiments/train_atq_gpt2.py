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
    """Run quantization-aware training."""
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: epochs={epochs}, lr={lr}, batch_size={batch_size}, "
          f"sparsity={sparsity_target}, mode={mode}, kd={use_kd}")

    print(f"\nLoading {model_name}...")
    student = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_size = get_model_size_mb(student)

    student = replace_linear_with_ternary(
        student, mode=mode, alpha=alpha, sparsity_target=sparsity_target
    )
    student = student.to(device)

    teacher = None
    if use_kd:
        print("Loading teacher model for knowledge distillation...")
        teacher = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    print("Loading training data...")
    train_loader = get_training_dataloader(
        tokenizer, seq_length, batch_size, max_samples=max_train_samples
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

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

            outputs = student(input_ids, labels=labels)
            loss = outputs.loss

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

        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "perplexity": ppl,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

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
