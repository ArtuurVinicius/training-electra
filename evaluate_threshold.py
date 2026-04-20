import argparse
import csv
import math
from typing import Dict, List, Tuple

import torch
from transformers import ElectraForPreTraining, PreTrainedTokenizerFast


def parse_label(raw_value: str, fake_values: set, real_values: set) -> int:
    value = str(raw_value).strip().lower()
    if value in fake_values:
        return 1
    if value in real_values:
        return 0

    # Fallback numeric parsing: >0 means fake (1), 0 means real (0)
    try:
        numeric = float(value)
        return 1 if numeric > 0 else 0
    except ValueError as exc:
        raise ValueError(f'Label invalido: {raw_value}') from exc


def load_labeled_csv(
    dataset_path: str,
    text_column: str,
    label_column: str,
    fake_values: set,
    real_values: set,
    max_samples: int,
) -> Tuple[List[str], List[int], int]:
    texts: List[str] = []
    labels: List[int] = []
    skipped = 0

    with open(dataset_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV sem cabecalho.')
        if text_column not in reader.fieldnames:
            raise ValueError(f'Coluna de texto "{text_column}" nao encontrada. Colunas: {reader.fieldnames}')
        if label_column not in reader.fieldnames:
            raise ValueError(f'Coluna de label "{label_column}" nao encontrada. Colunas: {reader.fieldnames}')

        for row in reader:
            if 0 < max_samples <= len(texts):
                break
            text = (row.get(text_column) or '').strip()
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
        raise ValueError('Nenhum exemplo valido encontrado no CSV.')

    classes = set(labels)
    if len(classes) < 2:
        raise ValueError('Avaliacao precisa de ao menos 2 classes (REAL e FAKE) no CSV informado.')

    return texts, labels, skipped


def score_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizerFast,
    model: ElectraForPreTraining,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[float]:
    scores: List[float] = []
    pin_memory = device.type == 'cuda'

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        input_ids = encoded['input_ids'].to(device, non_blocking=pin_memory)
        attention_mask = encoded['attention_mask'].to(device, non_blocking=pin_memory)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits)
            mask = attention_mask.float()
            mean_probs = (probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            scores.extend(mean_probs.detach().cpu().tolist())

    return scores


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def find_best_threshold(scores: List[float], labels: List[int]) -> Dict[str, float]:
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    total = len(paired)
    positives = sum(labels)
    negatives = total - positives

    # threshold > max_score => all REAL
    max_score = paired[0][0]
    eps = 1e-12
    best = {
        'threshold': max_score + eps,
        **compute_metrics(tp=0, fp=0, tn=negatives, fn=positives),
    }
    best_key = (
        best['accuracy'],
        best['f1'],
        best['precision'],
        best['recall'],
    )

    tp = 0
    fp = 0
    index = 0
    while index < total:
        current_score = paired[index][0]
        group_pos = 0
        group_neg = 0

        while index < total and paired[index][0] == current_score:
            if paired[index][1] == 1:
                group_pos += 1
            else:
                group_neg += 1
            index += 1

        tp += group_pos
        fp += group_neg
        fn = positives - tp
        tn = negatives - fp
        metrics = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
        candidate = {'threshold': current_score, **metrics}

        candidate_key = (
            candidate['accuracy'],
            candidate['f1'],
            candidate['precision'],
            candidate['recall'],
        )
        if candidate_key > best_key:
            best = candidate
            best_key = candidate_key

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='CSV com texto e label')
    parser.add_argument('--text_column', type=str, default='content', help='Nome da coluna de texto')
    parser.add_argument('--label_column', type=str, default='label', help='Nome da coluna de label')
    parser.add_argument('--fake_values', type=str, default='1,fake,falso', help='Valores considerados FAKE')
    parser.add_argument('--real_values', type=str, default='0,real,verdadeiro', help='Valores considerados REAL')
    parser.add_argument('--max_samples', type=int, default=-1, help='Limita amostras (debug)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--discriminator_dir', type=str, default='./electra_output/discriminator-final')
    parser.add_argument('--tokenizer_dir', type=str, default='./electra_output')
    parser.add_argument('--save_scored_csv', type=str, default='', help='Caminho opcional para salvar score/predicao')
    args = parser.parse_args()

    fake_values = {v.strip().lower() for v in args.fake_values.split(',') if v.strip()}
    real_values = {v.strip().lower() for v in args.real_values.split(',') if v.strip()}

    texts, labels, skipped = load_labeled_csv(
        dataset_path=args.dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
        fake_values=fake_values,
        real_values=real_values,
        max_samples=args.max_samples,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    best = find_best_threshold(scores, labels)

    print('=== Threshold Search Result ===')
    print(f'samples={len(texts)} skipped={skipped} device={device.type}')
    print(f'best_threshold={best["threshold"]:.6f}')
    print(f'accuracy={best["accuracy"]:.6f} f1={best["f1"]:.6f} precision={best["precision"]:.6f} recall={best["recall"]:.6f}')
    print(f'confusion_matrix: TP={best["tp"]} FP={best["fp"]} TN={best["tn"]} FN={best["fn"]}')

    if args.save_scored_csv:
        threshold = best['threshold']
        with open(args.save_scored_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label', 'score', 'pred'])
            for text, label, score in zip(texts, labels, scores):
                pred = 1 if score >= threshold else 0
                writer.writerow([text, label, f'{score:.8f}', pred])
        print(f'scored_csv_saved={args.save_scored_csv}')


if __name__ == '__main__':
    main()
