"""Baseline SEM adaptacao de dominio.

Roda exatamente o mesmo esquema de pontuacao/calibracao usado na avaliacao dos
modelos adaptados, porem carregando os checkpoints ORIGINAIS do HuggingFace
(sem o pre-treinamento continuo). O objetivo e isolar o ganho atribuivel a
adaptacao de dominio.

Checkpoints:
    - ELECTRA    : google/electra-small-discriminator
    - DistilBERT : distilbert-base-multilingual-cased

Reutiliza SEM reimplementar:
    - electra/evaluate_threshold.py :: score_texts     (escore ELECTRA)
    - bert/evaluate_treshold.py     :: score_texts     (escore DistilBERT-MLM)
    - electra/evaluate_threshold.py :: load_labeled_csv (mesma leitura/ordem)
    - protocolo_particoes.py        :: stratified_split, find_best_threshold,
                                       evaluate_at_threshold, compute_metrics

As funcoes de pontuacao ja recebem o modelo como parametro, entao apenas
trocamos o modelo carregado. Nenhum script de treinamento ou artefato adaptado
e alterado.

Nao altera scripts de treinamento nem artefatos dos modelos adaptados.
"""

import csv
import importlib.util
import os
import sys
from typing import Callable, Dict, List, Tuple

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    ElectraForPreTraining,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(REPO_ROOT, 'dataset', 'dataset_labeled.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'baseline_sem_adaptacao')

MAX_LENGTH = 128        # mesmo truncamento da avaliacao atual
MASK_STRIDE = 7         # mesmo mascaramento deterministico do DistilBERT
BATCH_SIZE = 32
SEED = 42               # mesma semente do protocolo de particoes
VAL_RATIO = 0.5         # 50/50 estratificado

# Valores conhecidos dos modelos ADAPTADOS (particao de teste, seed=42).
ADAPTED_ACC = {'ELECTRA': 0.8553, 'DistilBERT': 0.8575}


def _load_module(name: str, relpath: str):
    """Carrega um modulo por caminho de arquivo (evita problemas de package)."""
    path = os.path.join(REPO_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Modulos existentes reutilizados.
electra_eval = _load_module('electra_eval', os.path.join('electra', 'evaluate_threshold.py'))
bert_eval = _load_module('bert_eval', os.path.join('bert', 'evaluate_treshold.py'))
protocolo = _load_module('protocolo', 'protocolo_particoes.py')


# O dataset_labeled.csv usa rotulos textuais: "fake" e "true".
FAKE_VALUES = {'1', 'fake', 'falso'}
REAL_VALUES = {'0', 'real', 'verdadeiro', 'true'}

# CSV com campos de texto grandes (content). Evita erro de field size limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_texts_labels() -> Tuple[List[str], List[int], int]:
    # Mesma funcao/ordem usada na avaliacao dos modelos adaptados => particoes identicas.
    return electra_eval.load_labeled_csv(
        dataset_path=DATASET_PATH,
        text_column='content',
        label_column='label',
        fake_values=FAKE_VALUES,
        real_values=REAL_VALUES,
        max_samples=-1,
    )


def score_electra(texts: List[str], device: torch.device) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator').to(device)
    model.eval()
    # Reutiliza a funcao de pontuacao existente, so trocando o modelo.
    return electra_eval.score_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )


def score_distilbert(texts: List[str], device: torch.device) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased', use_fast=True)
    if tokenizer.mask_token_id is None:
        raise ValueError('Tokenizer sem [MASK].')
    model = AutoModelForMaskedLM.from_pretrained('distilbert-base-multilingual-cased').to(device)
    model.eval()
    # Reutiliza a funcao de pontuacao existente, so trocando o modelo.
    return bert_eval.score_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        mask_stride=MASK_STRIDE,
    )


def save_scores_csv(path: str, labels: List[int], scores: List[float]) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'score'])
        for label, score in zip(labels, scores):
            writer.writerow([label, f'{score:.8f}'])


def run_protocol(labels: List[int], scores: List[float]) -> Dict[str, Dict[str, float]]:
    """Split 50/50 estratificado (seed=42), calibra na validacao, mede no teste."""
    val_scores, val_labels, test_scores, test_labels = protocolo.stratified_split(
        scores, labels, val_ratio=VAL_RATIO, seed=SEED,
    )
    best_val = protocolo.find_best_threshold(val_scores, val_labels, criterion='accuracy')
    threshold = best_val['threshold']
    test_metrics = protocolo.evaluate_at_threshold(test_scores, test_labels, threshold)
    return {
        'threshold': threshold,
        'test': test_metrics,
        'test_labels': test_labels,
        'test_scores': test_scores,
    }


def print_model_report(name: str, threshold: float, m: Dict[str, float]) -> None:
    print(f'--- {name} (SEM adaptacao) | particao de TESTE ---')
    print(f'  limiar calibrado    = {threshold:.6f}')
    print(f'  acuracia            = {m["accuracy"]:.6f}')
    print(f'  acuracia balanceada = {m["macro_recall"]:.6f}')
    print(f'  F1 macro            = {m["macro_f1"]:.6f}')
    print(f'  recall FAKE         = {m["recall"]:.6f}')
    real_recall = m['tn'] / (m['tn'] + m['fp']) if (m['tn'] + m['fp']) else 0.0
    print(f'  recall REAL         = {real_recall:.6f}')
    print(f'  precisao FAKE       = {m["precision"]:.6f}')
    print(f'  matriz de confusao  : TN={m["tn"]} FP={m["fp"]} FN={m["fn"]} TP={m["tp"]}')


def majority_baseline(test_labels: List[int]) -> Dict[str, float]:
    """Classe majoritaria: sempre prever FALSO (1) na particao de teste."""
    # threshold = -inf => todo escore >= threshold => prediz 1 (FAKE) para todos.
    dummy_scores = [0.0] * len(test_labels)
    return protocolo.evaluate_at_threshold(dummy_scores, test_labels, float('-inf'))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    texts, labels, skipped = load_texts_labels()
    n_total = len(labels)
    n_fake = sum(labels)
    print('====================================================')
    print('  BASELINE SEM ADAPTACAO DE DOMINIO')
    print('====================================================')
    print(f'dataset    = {DATASET_PATH}')
    print(f'registros  = {n_total} (FAKE={n_fake} / REAL={n_total - n_fake}) | ignorados={skipped}')
    print(f'device={device.type} | max_length={MAX_LENGTH} | mask_stride={MASK_STRIDE} | seed={SEED}')
    print(f'sentido    = score >= threshold  =>  FAKE')
    print()

    results: Dict[str, Dict] = {}
    test_labels_ref: List[int] = []

    scorers: List[Tuple[str, Callable[[List[str], torch.device], List[float]], str]] = [
        ('ELECTRA', score_electra, 'electra_baseline_scores.csv'),
        ('DistilBERT', score_distilbert, 'distilbert_baseline_scores.csv'),
    ]

    for name, scorer, csv_name in scorers:
        print(f'>> Pontuando {name} (checkpoint original)...')
        scores = scorer(texts, device)
        csv_path = os.path.join(OUT_DIR, csv_name)
        save_scores_csv(csv_path, labels, scores)
        print(f'   scores salvos em: {csv_path}')

        res = run_protocol(labels, scores)
        results[name] = res
        test_labels_ref = res['test_labels']
        print()
        print_model_report(name, res['threshold'], res['test'])
        print()

    # Baseline de classe majoritaria (sempre FALSO) na mesma particao de teste.
    maj = majority_baseline(test_labels_ref)
    real_recall_maj = maj['tn'] / (maj['tn'] + maj['fp']) if (maj['tn'] + maj['fp']) else 0.0
    print('--- CLASSE MAJORITARIA (sempre prever FALSO) | particao de TESTE ---')
    print(f'  acuracia            = {maj["accuracy"]:.6f}')
    print(f'  acuracia balanceada = {maj["macro_recall"]:.6f}')
    print(f'  F1 macro            = {maj["macro_f1"]:.6f}')
    print(f'  recall FAKE         = {maj["recall"]:.6f}')
    print(f'  recall REAL         = {real_recall_maj:.6f}')
    print(f'  precisao FAKE       = {maj["precision"]:.6f}')
    print(f'  matriz de confusao  : TN={maj["tn"]} FP={maj["fp"]} FN={maj["fn"]} TP={maj["tp"]}')
    print()

    # Tabela comparativa de acuracia (particao de teste).
    print('==================== TABELA COMPARATIVA (acuracia, particao de TESTE) ====================')
    print(f'{"Configuracao":<28}{"ELECTRA":>12}{"DistilBERT":>14}')
    print('-' * 54)
    print(f'{"Classe majoritaria":<28}{maj["accuracy"]:>12.4f}{maj["accuracy"]:>14.4f}')
    print(f'{"Sem adaptacao":<28}'
          f'{results["ELECTRA"]["test"]["accuracy"]:>12.4f}'
          f'{results["DistilBERT"]["test"]["accuracy"]:>14.4f}')
    print(f'{"Adaptados (conhecido)":<28}{ADAPTED_ACC["ELECTRA"]:>12.4f}{ADAPTED_ACC["DistilBERT"]:>14.4f}')
    print('=========================================================================================')


if __name__ == '__main__':
    main()
