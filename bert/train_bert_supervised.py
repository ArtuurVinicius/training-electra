import argparse
import csv
import json
import math
import os
import random
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        raise ValueError(f'Invalid label: {raw_value}') from exc


def choose_text_column(column_names: Iterable[str], requested: str) -> str:
    if requested in column_names:
        return requested
    for candidate in ['content', 'text', 'sentence']:
        if candidate in column_names:
            return candidate
    raise ValueError(f'Text column "{requested}" not found. Available: {list(column_names)}')


def load_labeled_csv(
    dataset_path: str,
    text_column: str,
    label_column: str,
    fake_values: set,
    real_values: set,
) -> Tuple[List[str], List[int], int]:
    texts: List[str] = []
    labels: List[int] = []
    skipped = 0

    with open(dataset_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV has no header.')

        effective_text_column = choose_text_column(reader.fieldnames, text_column)
        if label_column not in reader.fieldnames:
            raise ValueError(
                f'Label column "{label_column}" not found. Available: {reader.fieldnames}. '
                'Use a labeled CSV for supervised fine-tuning.'
            )

        for row in reader:
            text = (row.get(effective_text_column) or '').strip()
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
        raise ValueError('No valid labeled rows found in CSV.')
    if len(set(labels)) < 2:
        raise ValueError('Need at least 2 classes (REAL and FAKE) for supervised fine-tuning.')

    return texts, labels, skipped


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f'Dataset not found: {args.dataset_path}')
    if args.per_device_train_batch_size <= 0:
        raise ValueError('--per_device_train_batch_size must be > 0')
    if args.per_device_eval_batch_size <= 0:
        raise ValueError('--per_device_eval_batch_size must be > 0')
    if args.gradient_accumulation_steps <= 0:
        raise ValueError('--gradient_accumulation_steps must be > 0')
    if args.num_train_epochs <= 0:
        raise ValueError('--num_train_epochs must be > 0')
    if args.max_seq_length <= 0:
        raise ValueError('--max_seq_length must be > 0')
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError('train/val/test ratios must be > 0')

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f'train_ratio + val_ratio + test_ratio must be 1.0 (got {ratio_sum:.6f})')


def resolve_base_model_path(model_name_or_path: str, fallback_model: str) -> str:
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    print(f'Warning: model path not found ({model_name_or_path}). Falling back to {fallback_model}.')
    return fallback_model


def maybe_print_gpu_memory_hint(device: torch.device) -> None:
    if device.type != 'cuda':
        print('CUDA not available: training will run on CPU (very slow).')
        return

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    print(f'GPU: {props.name} | VRAM total: {total_gb:.2f} GB')
    if total_gb < 5.5:
        print('Warning: VRAM under 5.5 GB. Lower max_seq_length or batch size if OOM occurs.')
    elif total_gb < 7.0:
        print('Hint: 6 GB profile. Keep fp16 enabled and max_seq_length around 128.')


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> Dataset:
    def tokenize_batch(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_seq_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=['text'])
    tokenized.set_format(type='torch')
    return tokenized


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collator,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': collator,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        kwargs['prefetch_factor'] = max(prefetch_factor, 1)
    return DataLoader(**kwargs)


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='binary',
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'confusion_matrix': cm.tolist(),
    }


def evaluate_classifier(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    pin_memory: bool,
    class_weights,
    decision_threshold: float,
) -> Dict[str, object]:
    model.eval()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device, non_blocking=pin_memory)
            inputs = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items() if k != 'labels'}

            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            losses.append(float(loss.item()))

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= decision_threshold).long()

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics['loss'] = float(np.mean(losses)) if losses else 0.0
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    metrics['y_prob'] = y_prob
    return metrics


def save_training_checkpoint(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    optimizer: AdamW,
    scheduler,
    scaler,
    output_dir: str,
    name: str,
    global_step: int,
) -> str:
    ckpt_dir = os.path.join(output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    state = {
        'global_step': global_step,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, os.path.join(ckpt_dir, 'trainer_state.pt'))
    return ckpt_dir


def save_confusion_matrix_plot(cm: List[List[int]], save_path: str, title: str) -> None:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    matrix = np.array(cm)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['REAL', 'FAKE'],
        yticklabels=['REAL', 'FAKE'],
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Supervised fine-tuning for DistilBERT fake/real classification')
    parser.add_argument('--dataset_path', type=str, required=True, help='Labeled CSV path')
    parser.add_argument('--text_column', type=str, default='content')
    parser.add_argument('--label_column', type=str, default='label')
    parser.add_argument('--fake_values', type=str, default='1,fake,falso')
    parser.add_argument('--real_values', type=str, default='0,real,verdadeiro')

    parser.add_argument('--model_name_or_path', type=str, default='./bert/bert_output/model-final')
    parser.add_argument('--fallback_model_name', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--tokenizer_dir', type=str, default='./bert/tokenizer')
    parser.add_argument('--output_dir', type=str, default='./bert/bert_classifier_output')

    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_train_epochs', type=int, default=4)
    parser.add_argument('--max_train_steps', type=int, default=-1)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--decision_threshold', type=float, default=0.5)
    parser.add_argument('--use_class_weights', dest='use_class_weights', action='store_true')
    parser.add_argument('--no_class_weights', dest='use_class_weights', action='store_false')
    parser.set_defaults(use_class_weights=True)

    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--no_pin_memory', action='store_true')
    parser.add_argument('--no_persistent_workers', action='store_true')
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--early_stopping_patience', type=int, default=2)

    parser.add_argument('--fp16', dest='fp16', action='store_true')
    parser.add_argument('--no_fp16', dest='fp16', action='store_false')
    parser.set_defaults(fp16=True)
    parser.add_argument('--gradient_checkpointing', dest='gradient_checkpointing', action='store_true')
    parser.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing', action='store_false')
    parser.set_defaults(gradient_checkpointing=True)

    parser.add_argument('--save_plot', type=str, default='./bert/bert_classifier_output/confusion_matrix_test.png')
    parser.add_argument('--save_predictions_csv', type=str, default='./bert/bert_classifier_output/test_predictions.csv')
    args = parser.parse_args()

    validate_args(args)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tokenizer_dir, exist_ok=True)

    fake_values = {v.strip().lower() for v in args.fake_values.split(',') if v.strip()}
    real_values = {v.strip().lower() for v in args.real_values.split(',') if v.strip()}

    texts, labels, skipped = load_labeled_csv(
        dataset_path=args.dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
        fake_values=fake_values,
        real_values=real_values,
    )

    texts_train, texts_temp, labels_train, labels_temp = train_test_split(
        texts,
        labels,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=labels,
    )
    val_share = args.val_ratio / (args.val_ratio + args.test_ratio)
    texts_val, texts_test, labels_val, labels_test = train_test_split(
        texts_temp,
        labels_temp,
        test_size=(1.0 - val_share),
        random_state=args.seed,
        stratify=labels_temp,
    )

    train_ds = Dataset.from_dict({'text': texts_train, 'labels': labels_train})
    val_ds = Dataset.from_dict({'text': texts_val, 'labels': labels_val})
    test_ds = Dataset.from_dict({'text': texts_test, 'labels': labels_test})

    model_source = resolve_base_model_path(args.model_name_or_path, args.fallback_model_name)
    tokenizer_source = args.tokenizer_dir if os.path.exists(args.tokenizer_dir) else model_source

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.model_max_length = args.max_seq_length
    tokenizer.save_pretrained(args.tokenizer_dir)

    train_ds = tokenize_dataset(train_ds, tokenizer, args.max_seq_length)
    val_ds = tokenize_dataset(val_ds, tokenizer, args.max_seq_length)
    test_ds = tokenize_dataset(test_ds, tokenizer, args.max_seq_length)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maybe_print_gpu_memory_hint(device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if device.type == 'cuda' else None,
        return_tensors='pt',
    )

    pin_memory = (device.type == 'cuda') and (not args.no_pin_memory)
    train_loader = make_dataloader(
        dataset=train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collator=collator,
        num_workers=max(args.num_workers, 0),
        pin_memory=pin_memory,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = make_dataloader(
        dataset=val_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collator=collator,
        num_workers=max(args.num_workers, 0),
        pin_memory=pin_memory,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = make_dataloader(
        dataset=test_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collator=collator,
        num_workers=max(args.num_workers, 0),
        pin_memory=pin_memory,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    class_weights = None
    if args.use_class_weights:
        class_counts = np.bincount(np.array(labels_train), minlength=2)
        weights = len(labels_train) / (2.0 * np.maximum(class_counts, 1))
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args.num_train_epochs
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_fp16 = bool(args.fp16 and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    print('=== Supervised DistilBERT Fine-tuning ===')
    print(f'samples={len(texts)} train={len(texts_train)} val={len(texts_val)} test={len(texts_test)} skipped={skipped}')
    print(f'device={device.type} fp16={use_fp16} grad_checkpointing={args.gradient_checkpointing}')
    print(f'batch_train={args.per_device_train_batch_size} batch_eval={args.per_device_eval_batch_size} grad_accum={args.gradient_accumulation_steps}')
    print(f'total_steps={total_steps} warmup_steps={warmup_steps}')

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    best_dir = os.path.join(args.output_dir, 'best-model')

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            labels_batch = batch['labels'].to(device, non_blocking=pin_memory)
            inputs = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items() if k != 'labels'}

            try:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss = F.cross_entropy(logits, labels_batch, weight=class_weights)
                    loss = loss / args.gradient_accumulation_steps
            except RuntimeError as exc:
                if 'out of memory' in str(exc).lower() and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        'CUDA OOM. Try lower max_seq_length, smaller batch size, or higher grad accumulation.'
                    ) from exc
                raise

            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss encountered: {loss.item()}')

            scaler.scale(loss).backward()
            running_loss += float(loss.item() * args.gradient_accumulation_steps)

            if step % args.gradient_accumulation_steps != 0:
                continue

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step % args.log_steps == 0:
                avg_loss = running_loss / args.log_steps
                lr = scheduler.get_last_lr()[0]
                print(f'epoch={epoch} step={global_step} train_loss={avg_loss:.4f} lr={lr:.8f}')
                running_loss = 0.0

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_training_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    output_dir=args.output_dir,
                    name=f'checkpoint-step-{global_step}',
                    global_step=global_step,
                )

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        val_metrics = evaluate_classifier(
            model=model,
            dataloader=val_loader,
            device=device,
            pin_memory=pin_memory,
            class_weights=class_weights,
            decision_threshold=args.decision_threshold,
        )
        print(
            f'epoch={epoch} val_loss={val_metrics["loss"]:.4f} val_acc={val_metrics["accuracy"]:.4f} '
            f'val_precision={val_metrics["precision"]:.4f} val_recall={val_metrics["recall"]:.4f} val_f1={val_metrics["f1"]:.4f}'
        )

        save_training_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            output_dir=args.output_dir,
            name=f'checkpoint-epoch-{epoch}',
            global_step=global_step,
        )

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            epochs_without_improvement = 0
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            torch.save({'best_val_f1': best_val_f1, 'global_step': global_step}, os.path.join(best_dir, 'best_state.pt'))
            print(f'Best model updated at epoch {epoch} with val_f1={best_val_f1:.4f}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f'Early stopping: no val_f1 improvement for {args.early_stopping_patience} epoch(s).')
                break

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    if not os.path.exists(best_dir):
        os.makedirs(best_dir, exist_ok=True)
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    test_metrics = evaluate_classifier(
        model=best_model,
        dataloader=test_loader,
        device=device,
        pin_memory=pin_memory,
        class_weights=class_weights,
        decision_threshold=args.decision_threshold,
    )

    print('=== Test Metrics ===')
    print(
        f'test_loss={test_metrics["loss"]:.4f} test_acc={test_metrics["accuracy"]:.4f} '
        f'test_precision={test_metrics["precision"]:.4f} test_recall={test_metrics["recall"]:.4f} test_f1={test_metrics["f1"]:.4f}'
    )
    print(
        f'confusion_matrix: TN={test_metrics["tn"]} FP={test_metrics["fp"]} '
        f'FN={test_metrics["fn"]} TP={test_metrics["tp"]}'
    )

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'best_val_f1': best_val_f1,
                'decision_threshold': args.decision_threshold,
                'test_metrics': {
                    'loss': test_metrics['loss'],
                    'accuracy': test_metrics['accuracy'],
                    'precision': test_metrics['precision'],
                    'recall': test_metrics['recall'],
                    'f1': test_metrics['f1'],
                    'tn': test_metrics['tn'],
                    'fp': test_metrics['fp'],
                    'fn': test_metrics['fn'],
                    'tp': test_metrics['tp'],
                    'confusion_matrix': test_metrics['confusion_matrix'],
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.save_predictions_csv:
        save_pred_dir = os.path.dirname(args.save_predictions_csv)
        if save_pred_dir:
            os.makedirs(save_pred_dir, exist_ok=True)
        with open(args.save_predictions_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label', 'prob_fake', 'pred'])
            for text, label, prob, pred in zip(texts_test, test_metrics['y_true'], test_metrics['y_prob'], test_metrics['y_pred']):
                writer.writerow([text, label, f'{prob:.8f}', pred])

    if args.save_plot:
        save_confusion_matrix_plot(
            cm=test_metrics['confusion_matrix'],
            save_path=args.save_plot,
            title='BERT Supervised Confusion Matrix (Test)',
        )

    tokenizer.save_pretrained(args.tokenizer_dir)
    print(f'Best model saved at: {best_dir}')
    print(f'Test metrics saved at: {metrics_path}')


if __name__ == '__main__':
    main()
