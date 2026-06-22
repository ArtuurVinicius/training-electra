"""
Mede a acuracia do DistilBERT ao longo do treino por STEP (mesmo formato do
experimento do ELECTRA), para comparacao direta no artigo.

Avalia os checkpoints intermediarios bert/bert_output_final/model-step-N e o
model-final, sobre o dataset rotulado, com a mesma metodologia da avaliacao
final (score por loss de MLM + melhor threshold no proprio dataset).

A epoch aproximada de cada step e calculada como step / steps_per_epoch, onde
steps_per_epoch = ceil(n_amostras / batch_size) -- exatamente como o train_bert
monta o dataloader (grad_accum=1, batch_size=8 por padrao).

Saidas (na pasta confusion_matrix_distilbert):
    - distilbert_accuracy_vs_steps.csv
    - distilbert_accuracy_vs_steps.png

Execucao a partir da raiz do projeto:
    venv\\Scripts\\python.exe confusion_matrix_distilbert\\distilbert_accuracy_vs_steps.py
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from distilbert_confusion_matrix_analysis import (
    compute_metrics,
    find_best_threshold,
    load_labeled_csv,
    parse_csv_values,
    score_texts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "confusion_matrix_distilbert"
DATASET_PATH = PROJECT_ROOT / "dataset" / "dataset_labeled.csv"
BERT_OUTPUT = PROJECT_ROOT / "bert" / "bert_output_final"

# Mesmos parametros do treino (train_bert.py) que definem steps_per_epoch.
TRAIN_CORPUS = PROJECT_ROOT / "dataset" / "final_corpus.csv"
TRAIN_TEXT_COLUMN = "content"
TRAIN_BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 3

# Mesmos parametros de avaliacao do confusion matrix do DistilBERT.
BATCH_SIZE = 16
MAX_LENGTH = 128
MASK_STRIDE = 7


def compute_steps_per_epoch() -> int:
    ds = load_dataset("csv", data_files=str(TRAIN_CORPUS), split="train")
    column = TRAIN_TEXT_COLUMN if TRAIN_TEXT_COLUMN in ds.column_names else ds.column_names[0]
    ds = ds.filter(lambda x: isinstance(x.get(column), str) and x.get(column).strip() != "")
    n_samples = len(ds)
    return math.ceil(n_samples / TRAIN_BATCH_SIZE)


def discover_step_checkpoints() -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in BERT_OUTPUT.glob("model-step-*"):
        match = re.search(r"model-step-(\d+)", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise RuntimeError(f"Nenhum checkpoint model-step-* encontrado em {BERT_OUTPUT}.")
    return found


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    steps_per_epoch = compute_steps_per_epoch()
    total_steps = steps_per_epoch * NUM_TRAIN_EPOCHS
    print(f"steps_per_epoch={steps_per_epoch} | total_steps(~)={total_steps}")

    fake_values = parse_csv_values("1,fake,falso,false")
    real_values = parse_csv_values("0,true,real,verdadeiro")

    texts, labels, skipped = load_labeled_csv(
        dataset_path=DATASET_PATH,
        text_column="content",
        label_column="label",
        fake_values=fake_values,
        real_values=real_values,
        max_samples=-1,
    )
    print(f"Samples: {len(texts)} | Skipped: {skipped}")

    checkpoints = discover_step_checkpoints()
    # Ultimo ponto = model-final, representando o fim do treino (epoch ~3.0).
    final_dir = BERT_OUTPUT / "model-final"
    if final_dir.exists():
        checkpoints.append((total_steps, final_dir))

    # Tokenizer e identico entre checkpoints; carrega uma vez.
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0][1], use_fast=True)
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer sem [MASK]; necessario para scoring MLM.")

    rows = []
    for step, ckpt_dir in checkpoints:
        epoch = step / steps_per_epoch
        print(f"\n--- step={step} (~epoch {epoch:.2f}) | {ckpt_dir.name} ---")
        model = AutoModelForMaskedLM.from_pretrained(ckpt_dir).to(device)
        model.eval()
        scores = score_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            mask_stride=MASK_STRIDE,
        )
        best = find_best_threshold(scores, labels)
        threshold = float(best["threshold"])
        predictions = [1 if s >= threshold else 0 for s in scores]
        metrics = compute_metrics(labels, predictions)
        row = {
            "step": step,
            "epoch": round(epoch, 3),
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_macro": metrics["f1_macro"],
        }
        rows.append(row)
        print(
            f"accuracy={row['accuracy']:.4f} f1={row['f1_score']:.4f} "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = OUTPUT_DIR / "distilbert_accuracy_vs_steps.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    png_path = OUTPUT_DIR / "distilbert_accuracy_vs_steps.png"
    epochs = [r["epoch"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    f1s = [r["f1_score"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accs, marker="o", label="Accuracy")
    plt.plot(epochs, f1s, marker="s", label="F1 (FAKE)")
    for x, y in zip(epochs, accs):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
    plt.xlabel("Epoch (aproximado a partir do step)")
    plt.ylabel("Metric")
    plt.title("DistilBERT - Acuracia vs Epochs (por step)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    print(f"\nCSV:  {csv_path}")
    print(f"PNG:  {png_path}")


if __name__ == "__main__":
    main()
