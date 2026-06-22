"""
Mede como a acuracia do ELECTRA varia ao longo do treino (epochs).

Aproveita os checkpoints intermediarios salvos em
electra/electra_output_final/discriminator-step-N e avalia cada um sobre o
dataset rotulado, usando exatamente a mesma metodologia da avaliacao final
(melhor threshold selecionado no proprio dataset). O total de steps do treino
corresponde a 3 epochs, entao epoch = step / (total_steps / 3).

Saidas (na pasta confusion_matrix_electra):
    - electra_accuracy_vs_epochs.csv
    - electra_accuracy_vs_epochs.png

Execucao a partir da raiz do projeto:
    venv\\Scripts\\python.exe confusion_matrix_electra\\electra_accuracy_vs_epochs.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import ElectraForPreTraining, PreTrainedTokenizerFast

from electra_confusion_matrix_analysis import (
    compute_metrics,
    find_best_threshold,
    load_labeled_csv,
    parse_csv_values,
    score_texts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "confusion_matrix_electra"
DATASET_PATH = PROJECT_ROOT / "dataset" / "dataset_labeled.csv"
ELECTRA_OUTPUT = PROJECT_ROOT / "electra" / "electra_output_final"
TOKENIZER_DIR = ELECTRA_OUTPUT
NUM_TRAIN_EPOCHS = 3

# Checkpoints escolhidos para cobrir bem as 3 epochs.
SELECTED_STEPS = [500, 4500, 8500, 13000, 17500, 21500]
BATCH_SIZE = 32
MAX_LENGTH = 512


def discover_max_step() -> int:
    steps = []
    for path in ELECTRA_OUTPUT.glob("discriminator-step-*"):
        match = re.search(r"discriminator-step-(\d+)", path.name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise RuntimeError("Nenhum checkpoint discriminator-step-* encontrado.")
    return max(steps)


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

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    max_step = discover_max_step()
    steps_per_epoch = max_step / NUM_TRAIN_EPOCHS

    # step -> caminho do checkpoint. O ultimo ponto usa discriminator-final.
    checkpoints: list[tuple[int, Path]] = []
    for step in SELECTED_STEPS:
        ckpt = ELECTRA_OUTPUT / f"discriminator-step-{step}"
        if ckpt.exists():
            checkpoints.append((step, ckpt))
    checkpoints.append((max_step, ELECTRA_OUTPUT / "discriminator-final"))

    rows = []
    for step, ckpt_dir in checkpoints:
        epoch = step / steps_per_epoch
        print(f"\n--- step={step} (~epoch {epoch:.2f}) | {ckpt_dir.name} ---")
        model = ElectraForPreTraining.from_pretrained(ckpt_dir).to(device)
        model.eval()
        scores = score_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
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

    csv_path = OUTPUT_DIR / "electra_accuracy_vs_epochs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    png_path = OUTPUT_DIR / "electra_accuracy_vs_epochs.png"
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
    plt.title("ELECTRA - Acuracia vs Epochs (pre-treino)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    print(f"\nCSV:  {csv_path}")
    print(f"PNG:  {png_path}")


if __name__ == "__main__":
    main()
