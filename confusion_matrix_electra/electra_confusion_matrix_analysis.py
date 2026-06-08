"""
Generate evaluation artifacts for the trained ELECTRA discriminator.

Default execution from the project root:

    venv\\Scripts\\python.exe confusion_matrix\\electra_confusion_matrix_analysis.py

Outputs are written to the confusion_matrix folder:
    - electra_confusion_matrix.png
    - electra_metrics.json
    - electra_metrics.txt
    - electra_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import ElectraForPreTraining, PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "confusion_matrix"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "dataset_labeled.csv"
DEFAULT_DISCRIMINATOR_DIR = PROJECT_ROOT / "electra" / "electra_output" / "discriminator-final"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "electra" / "electra_output"


def parse_csv_values(raw_values: str) -> set[str]:
    return {value.strip().lower() for value in raw_values.split(",") if value.strip()}


def parse_label(raw_value: object, fake_values: set[str], real_values: set[str]) -> int:
    value = str(raw_value).strip().lower()
    if value in fake_values:
        return 1
    if value in real_values:
        return 0

    try:
        return 1 if float(value) > 0 else 0
    except ValueError as exc:
        raise ValueError(f"Invalid label: {raw_value}") from exc


def load_labeled_csv(
    dataset_path: Path,
    text_column: str,
    label_column: str,
    fake_values: set[str],
    real_values: set[str],
    max_samples: int,
) -> tuple[list[str], list[int], int]:
    texts: list[str] = []
    labels: list[int] = []
    skipped = 0

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        if text_column not in reader.fieldnames:
            raise ValueError(f'Text column "{text_column}" not found. Columns: {reader.fieldnames}')
        if label_column not in reader.fieldnames:
            raise ValueError(f'Label column "{label_column}" not found. Columns: {reader.fieldnames}')

        for row in reader:
            if 0 < max_samples <= len(texts):
                break

            text = (row.get(text_column) or "").strip()
            raw_label = row.get(label_column)
            if not text or raw_label is None:
                skipped += 1
                continue

            try:
                label = parse_label(raw_label, fake_values, real_values)
            except ValueError:
                skipped += 1
                continue

            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError("No valid examples found in the CSV.")
    return texts, labels, skipped


def score_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerFast,
    model: ElectraForPreTraining,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[float]:
    scores: list[float] = []
    pin_memory = device.type == "cuda"

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        input_ids = encoded["input_ids"].to(device, non_blocking=pin_memory)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=pin_memory)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probabilities = torch.sigmoid(logits)
            token_mask = attention_mask.float()
            mean_scores = (probabilities * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)

        scores.extend(float(score) for score in mean_scores.detach().cpu())

    return scores


def confusion_counts(labels: list[int], predictions: list[int]) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(scores: list[float], labels: list[int]) -> dict[str, float]:
    candidates = sorted(set(scores), reverse=True)
    best_threshold = candidates[0] + 1e-12
    best_predictions = [0 for _ in labels]
    best_key = metric_sort_key(labels, best_predictions)

    for threshold in candidates:
        predictions = [1 if score >= threshold else 0 for score in scores]
        candidate_key = metric_sort_key(labels, predictions)
        if candidate_key > best_key:
            best_threshold = threshold
            best_predictions = predictions
            best_key = candidate_key

    counts = confusion_counts(labels, best_predictions)
    return {"threshold": float(best_threshold), **counts}


def metric_sort_key(labels: list[int], predictions: list[int]) -> tuple[float, float, float, float]:
    return (
        accuracy_score(labels, predictions),
        f1_score(labels, predictions, zero_division=0),
        precision_score(labels, predictions, zero_division=0),
        recall_score(labels, predictions, zero_division=0),
    )


def load_threshold_from_json(threshold_json: Path) -> float:
    with threshold_json.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)

    for key in ("best_threshold", "threshold"):
        if key in payload:
            return float(payload[key])
    if isinstance(payload.get("metrics"), dict):
        for key in ("best_threshold", "threshold"):
            if key in payload["metrics"]:
                return float(payload["metrics"][key])

    raise ValueError(f"Could not find a threshold value in {threshold_json}")


def compute_metrics(labels: list[int], predictions: list[int]) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    counts = confusion_counts(labels, predictions)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_counts": counts,
    }


def save_predictions(
    output_path: Path,
    texts: Iterable[str],
    labels: Iterable[int],
    scores: Iterable[float],
    predictions: Iterable[int],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text", "label", "label_name", "score", "prediction", "prediction_name"])
        for text, label, score, prediction in zip(texts, labels, scores, predictions):
            writer.writerow(
                [
                    text,
                    int(label),
                    "FAKE" if label == 1 else "REAL",
                    f"{score:.8f}",
                    int(prediction),
                    "FAKE" if prediction == 1 else "REAL",
                ]
            )


def save_confusion_matrix_plot(output_path: Path, matrix: list[list[int]], metrics: dict[str, object]) -> None:
    plt.figure(figsize=(7, 5.5))
    axis = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        annot_kws={"fontsize": 14},
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(
        "ELECTRA Confusion Matrix\n"
        f"Accuracy={metrics['accuracy']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1_score']:.4f}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_metrics_files(
    json_path: Path,
    txt_path: Path,
    metrics: dict[str, object],
    report: str,
    metadata: dict[str, object],
) -> None:
    payload = {**metadata, "metrics": metrics, "classification_report": report}
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)

    counts = metrics["confusion_counts"]
    with txt_path.open("w", encoding="utf-8") as txt_file:
        txt_file.write("ELECTRA evaluation metrics\n")
        txt_file.write("==========================\n")
        txt_file.write(f"Dataset: {metadata['dataset_path']}\n")
        txt_file.write(f"Discriminator: {metadata['discriminator_dir']}\n")
        txt_file.write(f"Tokenizer: {metadata['tokenizer_dir']}\n")
        txt_file.write(f"Samples: {metadata['samples']} | Skipped: {metadata['skipped']}\n")
        txt_file.write(f"Device: {metadata['device']}\n")
        txt_file.write(f"Threshold: {metadata['threshold']:.8f} ({metadata['threshold_source']})\n\n")
        txt_file.write(f"Accuracy: {metrics['accuracy']:.6f}\n")
        txt_file.write(f"Recall: {metrics['recall']:.6f}\n")
        txt_file.write(f"F1 Score: {metrics['f1_score']:.6f}\n")
        txt_file.write(f"Precision: {metrics['precision']:.6f}\n")
        txt_file.write(f"Macro Recall: {metrics['recall_macro']:.6f}\n")
        txt_file.write(f"Macro F1: {metrics['f1_macro']:.6f}\n\n")
        txt_file.write(
            "Confusion counts: "
            f"TP={counts['tp']} FP={counts['fp']} TN={counts['tn']} FN={counts['fn']}\n\n"
        )
        txt_file.write("Classification report\n")
        txt_file.write("---------------------\n")
        txt_file.write(report)
        txt_file.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the trained ELECTRA model and save a confusion matrix.")
    parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--text_column", type=str, default="content")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--fake_values", type=str, default="1,fake,falso,false")
    parser.add_argument("--real_values", type=str, default="0,true,real,verdadeiro")
    parser.add_argument("--discriminator_dir", type=Path, default=DEFAULT_DISCRIMINATOR_DIR)
    parser.add_argument("--tokenizer_dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional sample limit for quick checks.")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold. If omitted, one is selected.")
    parser.add_argument("--threshold_json", type=Path, default=None, help="JSON file containing best_threshold/threshold.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fake_values = parse_csv_values(args.fake_values)
    real_values = parse_csv_values(args.real_values)

    texts, labels, skipped = load_labeled_csv(
        dataset_path=args.dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
        fake_values=fake_values,
        real_values=real_values,
        max_samples=args.max_samples,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    model = ElectraForPreTraining.from_pretrained(args.discriminator_dir).to(device)
    model.eval()

    scores = score_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    threshold_source = "auto_selected_on_evaluation_dataset"
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "command_line"
    elif args.threshold_json is not None:
        threshold = load_threshold_from_json(args.threshold_json)
        threshold_source = str(args.threshold_json)
    else:
        best = find_best_threshold(scores, labels)
        threshold = float(best["threshold"])

    predictions = [1 if score >= threshold else 0 for score in scores]
    metrics = compute_metrics(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        zero_division=0,
    )

    metadata = {
        "dataset_path": str(args.dataset_path),
        "discriminator_dir": str(args.discriminator_dir),
        "tokenizer_dir": str(args.tokenizer_dir),
        "samples": len(texts),
        "skipped": skipped,
        "device": device.type,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "positive_class": "FAKE",
        "score_semantics": "higher_mean_discriminator_probability_means_more_likely_fake",
    }

    plot_path = args.output_dir / "electra_confusion_matrix.png"
    metrics_json_path = args.output_dir / "electra_metrics.json"
    metrics_txt_path = args.output_dir / "electra_metrics.txt"
    predictions_path = args.output_dir / "electra_predictions.csv"

    save_confusion_matrix_plot(plot_path, metrics["confusion_matrix"], metrics)
    save_metrics_files(metrics_json_path, metrics_txt_path, metrics, report, metadata)
    save_predictions(predictions_path, texts, labels, scores, predictions)

    print("ELECTRA evaluation completed.")
    print(f"Samples: {len(texts)} | Skipped: {skipped} | Device: {device.type}")
    print(f"Threshold: {threshold:.8f} ({threshold_source})")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1 Score: {metrics['f1_score']:.6f}")
    print(f"Confusion matrix PNG: {plot_path}")
    print(f"Metrics JSON: {metrics_json_path}")
    print(f"Metrics TXT: {metrics_txt_path}")
    print(f"Predictions CSV: {predictions_path}")


if __name__ == "__main__":
    main()
