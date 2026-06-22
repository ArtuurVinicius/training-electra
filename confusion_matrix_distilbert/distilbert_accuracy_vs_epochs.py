"""
Mede como a acuracia do DistilBERT varia por epoch.

Avalia cada checkpoint salvo ao fim de uma epoch
(bert/bert_output_final/model-epoch-N) sobre o dataset rotulado, usando a mesma
metodologia da avaliacao final (score por loss de MLM + melhor threshold
selecionado no proprio dataset).

Pre-requisito: treinar com train_bert.py ja ajustado para salvar model-epoch-N.

Saidas (na pasta confusion_matrix_distilbert):
    - distilbert_accuracy_vs_epochs.csv
    - distilbert_accuracy_vs_epochs.png

Execucao a partir da raiz do projeto:
    venv\\Scripts\\python.exe confusion_matrix_distilbert\\distilbert_accuracy_vs_epochs.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
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

BATCH_SIZE = 16
MAX_LENGTH = 128
MASK_STRIDE = 7


def discover_epoch_checkpoints() -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in BERT_OUTPUT.glob("model-epoch-*"):
        match = re.search(r"model-epoch-(\d+)", path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise RuntimeError(
            "Nenhum checkpoint model-epoch-* encontrado em "
            f"{BERT_OUTPUT}. Treine antes com train_bert.py (ja ajustado)."
        )
    return found


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

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

    checkpoints = discover_epoch_checkpoints()
    # Tokenizer e identico entre epochs; carrega uma vez do primeiro checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0][1], use_fast=True)
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer sem [MASK]; necessario para scoring MLM.")

    rows = []
    for epoch, ckpt_dir in checkpoints:
        print(f"\n--- epoch {epoch} | {ckpt_dir.name} ---")
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
            "epoch": epoch,
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

    csv_path = OUTPUT_DIR / "distilbert_accuracy_vs_epochs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    png_path = OUTPUT_DIR / "distilbert_accuracy_vs_epochs.png"
    epochs = [r["epoch"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    f1s = [r["f1_score"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accs, marker="o", label="Accuracy")
    plt.plot(epochs, f1s, marker="s", label="F1 (FAKE)")
    for x, y in zip(epochs, accs):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("DistilBERT - Acuracia vs Epochs")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    print(f"\nCSV:  {csv_path}")
    print(f"PNG:  {png_path}")


if __name__ == "__main__":
    main()
