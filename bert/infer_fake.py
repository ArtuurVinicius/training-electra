import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from evaluate_treshold import score_texts


def main() -> None:
	parser = argparse.ArgumentParser(description='Run fake/real inference with BERT MLM score (Electra-like flow)')
	parser.add_argument('--text', type=str, default=None, help='Single text to classify')
	parser.add_argument('--file', type=str, default=None, help='Text file (one line per sample)')
	parser.add_argument('--threshold', type=float, default=6.0, help='Decision threshold over MLM loss score')
	parser.add_argument('--max_length', type=int, default=128)
	parser.add_argument('--mask_stride', type=int, default=7, help='Mask one token each N positions to compute MLM score')
	parser.add_argument('--model_dir', type=str, default='./bert/bert_output/model-final')
	parser.add_argument('--tokenizer_dir', type=str, default='./bert/tokenizer')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
	if tokenizer.mask_token_id is None:
		raise ValueError('Tokenizer has no mask token. Provide a BERT-like tokenizer with [MASK].')
	model = AutoModelForMaskedLM.from_pretrained(args.model_dir).to(device)
	model.eval()

	samples = []
	if args.text:
		samples.append(args.text)
	if args.file:
		with open(args.file, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if line:
					samples.append(line)

	if not samples:
		print('No input text provided. Use --text or --file.')
		return

	scores = score_texts(
		texts=samples,
		tokenizer=tokenizer,
		model=model,
		device=device,
		batch_size=32,
		max_length=args.max_length,
		mask_stride=args.mask_stride,
	)

	for index, (text, score) in enumerate(zip(samples, scores), start=1):
		label = 'FAKE' if score >= args.threshold else 'REAL'
		print(f'[{index}] mlm_score={score:.4f} threshold={args.threshold:.2f} -> {label}\n{text}\n')


if __name__ == '__main__':
	main()

