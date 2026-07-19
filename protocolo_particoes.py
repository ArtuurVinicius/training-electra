"""Protocolo de particoes: calibra o limiar na validacao e mede no teste.

Motivacao: escolher o melhor limiar no mesmo conjunto em que se reporta a
metrica gera resultado otimista (viesado). Aqui dividimos os escores em dois
grupos 50/50 de forma estratificada:

    - Grupo A (validacao): usado APENAS para escolher o limiar.
    - Grupo B (teste):      usado APENAS para medir. E o numero honesto.

Entrada: um CSV com (ao menos) as colunas `label` e `score`.
Convencao de escore: score mais alto => mais provavel FAKE (label 1),
o mesmo sentido usado nos scripts de avaliacao dos dois modelos.

Uso:
    python protocolo_particoes.py escores_electra.csv --modelo ELECTRA
    python protocolo_particoes.py escores_distilbert.csv --modelo DistilBERT
"""

import argparse
import csv
import random
import sys
from typing import Dict, List, Tuple


def parse_label(raw_value: str, fake_values: set, real_values: set) -> int:
    value = str(raw_value).strip().lower()
    if value in fake_values:
        return 1
    if value in real_values:
        return 0
    try:
        numeric = float(value)
        return 1 if numeric > 0 else 0
    except ValueError as exc:
        raise ValueError(f'Label invalido: {raw_value}') from exc


def load_scores(
    path: str,
    label_column: str,
    score_column: str,
    fake_values: set,
    real_values: set,
) -> Tuple[List[float], List[int], int]:
    scores: List[float] = []
    labels: List[int] = []
    skipped = 0

    # csv.DictReader lida com quebras de linha dentro de campos de texto entre aspas.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV sem cabecalho.')
        if score_column not in reader.fieldnames:
            raise ValueError(f'Coluna de score "{score_column}" nao encontrada. Colunas: {reader.fieldnames}')
        if label_column not in reader.fieldnames:
            raise ValueError(f'Coluna de label "{label_column}" nao encontrada. Colunas: {reader.fieldnames}')

        for row in reader:
            raw_label = row.get(label_column)
            raw_score = row.get(score_column)
            if raw_label is None or raw_score is None or str(raw_score).strip() == '':
                skipped += 1
                continue
            try:
                label = parse_label(raw_label, fake_values, real_values)
                score = float(raw_score)
            except ValueError:
                skipped += 1
                continue
            labels.append(label)
            scores.append(score)

    if not scores:
        raise ValueError('Nenhum registro valido (label+score) encontrado no CSV.')
    if len(set(labels)) < 2:
        raise ValueError('E preciso ao menos 2 classes (REAL e FAKE) no CSV.')

    return scores, labels, skipped


def stratified_split(
    scores: List[float],
    labels: List[int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[float], List[int], List[float], List[int]]:
    """Divide de forma estratificada por classe (mantem a proporcao FAKE/REAL nos dois grupos)."""
    rng = random.Random(seed)
    idx_by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        idx_by_class.setdefault(y, []).append(i)

    val_idx: List[int] = []
    test_idx: List[int] = []
    for _, idxs in idx_by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = round(len(idxs) * val_ratio)
        val_idx.extend(idxs[:n_val])
        test_idx.extend(idxs[n_val:])

    val_scores = [scores[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_scores = [scores[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    return val_scores, val_labels, test_scores, test_labels


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0

    # Classe FAKE (positiva = 1)
    precision_f = tp / (tp + fp) if (tp + fp) else 0.0
    recall_f = tp / (tp + fn) if (tp + fn) else 0.0
    f1_f = (2 * precision_f * recall_f / (precision_f + recall_f)) if (precision_f + recall_f) else 0.0

    # Classe REAL (positiva = 0)
    precision_r = tn / (tn + fn) if (tn + fn) else 0.0
    recall_r = tn / (tn + fp) if (tn + fp) else 0.0
    f1_r = (2 * precision_r * recall_r / (precision_r + recall_r)) if (precision_r + recall_r) else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision_f,
        'recall': recall_f,
        'f1': f1_f,
        'macro_precision': (precision_f + precision_r) / 2,
        'macro_recall': (recall_f + recall_r) / 2,
        'macro_f1': (f1_f + f1_r) / 2,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def evaluate_at_threshold(scores: List[float], labels: List[int], threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    metrics = compute_metrics(tp, fp, tn, fn)
    metrics['threshold'] = threshold
    return metrics


def find_best_threshold(scores: List[float], labels: List[int], criterion: str) -> Dict[str, float]:
    """Escolhe o limiar que maximiza o criterio (accuracy ou macro_f1) na validacao.

    Varre todos os limiares candidatos (os proprios escores). Mesma logica de
    ordenacao usada nos scripts de avaliacao originais.
    """
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    total = len(paired)
    positives = sum(labels)
    negatives = total - positives

    def key_of(m: Dict[str, float]) -> Tuple[float, ...]:
        if criterion == 'macro_f1':
            return (m['macro_f1'], m['accuracy'], m['f1'])
        return (m['accuracy'], m['f1'], m['precision'], m['recall'])

    # Limiar acima do maior escore => tudo previsto como REAL.
    max_score = paired[0][0]
    best = compute_metrics(tp=0, fp=0, tn=negatives, fn=positives)
    best['threshold'] = max_score + 1e-12
    best_key = key_of(best)

    tp = 0
    fp = 0
    index = 0
    while index < total:
        current_score = paired[index][0]
        while index < total and paired[index][0] == current_score:
            if paired[index][1] == 1:
                tp += 1
            else:
                fp += 1
            index += 1
        fn = positives - tp
        tn = negatives - fp
        candidate = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
        candidate['threshold'] = current_score
        candidate_key = key_of(candidate)
        if candidate_key > best_key:
            best = candidate
            best_key = candidate_key

    return best


def print_block(title: str, m: Dict[str, float], n: int) -> None:
    print(f'--- {title} (n={n}) ---')
    print(f'  threshold  = {m["threshold"]:.6f}')
    print(f'  accuracy   = {m["accuracy"]:.6f}')
    print(f'  precision  = {m["precision"]:.6f}   (classe FAKE)')
    print(f'  recall     = {m["recall"]:.6f}   (classe FAKE)')
    print(f'  f1         = {m["f1"]:.6f}   (classe FAKE)')
    print(f'  macro_prec = {m["macro_precision"]:.6f}')
    print(f'  macro_rec  = {m["macro_recall"]:.6f}')
    print(f'  macro_f1   = {m["macro_f1"]:.6f}')
    print(f'  confusion  : TP={m["tp"]} FP={m["fp"]} TN={m["tn"]} FN={m["fn"]}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Calibra limiar na validacao e mede no teste (split 50/50 estratificado).')
    parser.add_argument('scores_csv', type=str, help='CSV com colunas label e score')
    parser.add_argument('--modelo', type=str, default='(modelo)', help='Nome do modelo (apenas rotulo de saida)')
    parser.add_argument('--label_column', type=str, default='label')
    parser.add_argument('--score_column', type=str, default='score')
    parser.add_argument('--fake_values', type=str, default='1,fake,falso')
    parser.add_argument('--real_values', type=str, default='0,real,verdadeiro')
    parser.add_argument('--val_ratio', type=float, default=0.5, help='Fracao usada para calibrar o limiar (validacao)')
    parser.add_argument('--seed', type=int, default=42, help='Semente do split (reprodutibilidade)')
    parser.add_argument('--criterion', type=str, default='accuracy', choices=['accuracy', 'macro_f1'],
                        help='Metrica maximizada ao escolher o limiar na validacao')
    args = parser.parse_args()

    fake_values = {v.strip().lower() for v in args.fake_values.split(',') if v.strip()}
    real_values = {v.strip().lower() for v in args.real_values.split(',') if v.strip()}

    scores, labels, skipped = load_scores(
        path=args.scores_csv,
        label_column=args.label_column,
        score_column=args.score_column,
        fake_values=fake_values,
        real_values=real_values,
    )

    val_scores, val_labels, test_scores, test_labels = stratified_split(
        scores, labels, val_ratio=args.val_ratio, seed=args.seed,
    )

    # 1) Calibra o limiar SOMENTE na validacao.
    best_val = find_best_threshold(val_scores, val_labels, criterion=args.criterion)
    threshold = best_val['threshold']

    # 2) Aplica esse MESMO limiar (fixo) no teste. Este e o numero honesto.
    test_metrics = evaluate_at_threshold(test_scores, test_labels, threshold)

    n_total = len(scores)
    n_fake = sum(labels)
    print('====================================================')
    print(f'  PROTOCOLO DE PARTICOES  |  modelo: {args.modelo}')
    print('====================================================')
    print(f'arquivo      = {args.scores_csv}')
    print(f'registros    = {n_total} (FAKE={n_fake} / REAL={n_total - n_fake}) | ignorados={skipped}')
    print(f'split        = {int(round((1-args.val_ratio)*100))}/{int(round(args.val_ratio*100))} teste/validacao'
          f' | seed={args.seed} | criterio={args.criterion}')
    print(f'sentido      = score >= threshold  =>  FAKE')
    print()
    print_block('VALIDACAO  (usada so p/ calibrar - OTIMISTA, ignore)', best_val, len(val_scores))
    print()
    print_block('TESTE  <<< NUMEROS PARA A MONOGRAFIA', test_metrics, len(test_scores))
    print('====================================================')


if __name__ == '__main__':
    main()
