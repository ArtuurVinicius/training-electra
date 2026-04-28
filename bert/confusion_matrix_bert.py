import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from evaluate_treshold import find_best_threshold, load_labeled_csv, score_texts


def main() -> None:
	parser = argparse.ArgumentParser(description='Compute and plot confusion matrix for BERT MLM score (Electra-like flow)')
	parser.add_argument('--dataset_path', type=str, required=True, help='Labeled CSV with text and label')
	parser.add_argument('--text_column', type=str, default='content')
	parser.add_argument('--label_column', type=str, default='label')
	parser.add_argument('--fake_values', type=str, default='1,fake,falso')
	parser.add_argument('--real_values', type=str, default='0,real,verdadeiro')
	parser.add_argument('--model_dir', type=str, default='./bert/bert_output/model-final')
	parser.add_argument('--tokenizer_dir', type=str, default='./bert/tokenizer')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--max_length', type=int, default=128)
	parser.add_argument('--mask_stride', type=int, default=7, help='Mask one token each N positions to compute MLM score')
	parser.add_argument('--threshold', type=float, default=None, help='If omitted, search best threshold automatically')
	parser.add_argument('--save_plot', type=str, default='', help='Optional output path for confusion matrix image')
	parser.add_argument('--save_scored_csv', type=str, default='', help='Optional output path for scores and predictions')
	parser.add_argument('--show', action='store_true', help='Show plot interactively')
	args = parser.parse_args()

	fake_values = {v.strip().lower() for v in args.fake_values.split(',') if v.strip()}
	real_values = {v.strip().lower() for v in args.real_values.split(',') if v.strip()}

	texts, labels, skipped = load_labeled_csv(
		dataset_path=args.dataset_path,
		text_column=args.text_column,
		label_column=args.label_column,
		fake_values=fake_values,
		real_values=real_values,
		max_samples=-1,
	)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
	if tokenizer.mask_token_id is None:
		raise ValueError('Tokenizer has no mask token. Provide a BERT-like tokenizer with [MASK].')
	model = AutoModelForMaskedLM.from_pretrained(args.model_dir).to(device)
	model.eval()

	scores = score_texts(
		texts=texts,
		tokenizer=tokenizer,
		model=model,
		device=device,
		batch_size=args.batch_size,
		max_length=args.max_length,
		mask_stride=args.mask_stride,
	)

	threshold = args.threshold
	if threshold is None:
		best = find_best_threshold(scores, labels)
		threshold = best['threshold']
		print(f'Best threshold: {threshold:.6f}')
		print(f'Confusion (TP={best["tp"]} FP={best["fp"]} TN={best["tn"]} FN={best["fn"]})')
	else:
		print(f'Using provided threshold: {threshold:.6f}')

	preds = [1 if s >= threshold else 0 for s in scores]
	cm = confusion_matrix(labels, preds, labels=[0, 1])

	print('Confusion matrix:')
	print(cm)
	print('\nClassification report:')
	print(classification_report(labels, preds, target_names=['REAL', 'FAKE']))

	tn, fp, fn, tp = cm.ravel()
	total = tp + tn + fp + fn
	accuracy = (tp + tn) / total if total else 0.0
	precision = tp / (tp + fp) if (tp + fp) else 0.0
	recall = tp / (tp + fn) if (tp + fn) else 0.0
	f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
	print(f'Computed from CM: accuracy={accuracy:.6f} precision={precision:.6f} f1={f1:.6f} (TP={tp} FP={fp} TN={tn} FN={fn})')
	print('score_semantics: higher_mlm_loss => more_likely_fake')
	print(f'samples={len(texts)} skipped={skipped} device={device.type}')

	if args.save_scored_csv:
		with open(args.save_scored_csv, 'w', encoding='utf-8', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['text', 'label', 'score', 'pred'])
			for text, label, score, pred in zip(texts, labels, scores, preds):
				writer.writerow([text, label, f'{score:.8f}', pred])
		print(f'Scored CSV saved to {args.save_scored_csv}')

	if args.save_plot or args.show:
		plt.figure(figsize=(5, 4))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.title('BERT MLM Confusion Matrix')
		plt.tight_layout()
		if args.save_plot:
			plt.savefig(args.save_plot, dpi=150)
			print(f'Plot saved to {args.save_plot}')
		if args.show:
			plt.show()


if __name__ == '__main__':
	main()

