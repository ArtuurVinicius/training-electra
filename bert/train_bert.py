import argparse
import math
import os
from typing import Dict, Iterable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
	AutoModelForMaskedLM,
	AutoTokenizer,
	DataCollatorForLanguageModeling,
	get_scheduler,
)


def choose_text_column(column_names: Iterable[str], requested: str) -> str:
	if requested in column_names:
		return requested
	for candidate in ['content', 'text', 'sentence']:
		if candidate in column_names:
			return candidate
	raise ValueError(f'Text column "{requested}" not found. Available: {list(column_names)}')


def validate_environment_and_args(args: argparse.Namespace) -> None:
	if not os.path.exists(args.dataset_path):
		raise FileNotFoundError(f'Dataset not found: {args.dataset_path}')
	if args.per_device_train_batch_size <= 0:
		raise ValueError('--per_device_train_batch_size must be > 0')
	if args.gradient_accumulation_steps <= 0:
		raise ValueError('--gradient_accumulation_steps must be > 0')
	if args.max_seq_length <= 0:
		raise ValueError('--max_seq_length must be > 0')
	if args.num_train_epochs <= 0:
		raise ValueError('--num_train_epochs must be > 0')


def maybe_print_gpu_memory_hint(device: torch.device) -> None:
	if device.type != 'cuda':
		print('CUDA not available: training will run on CPU (very slow).')
		return

	props = torch.cuda.get_device_properties(0)
	total_gb = props.total_memory / (1024 ** 3)
	print(f'GPU: {props.name} | VRAM total: {total_gb:.2f} GB')

	if total_gb < 5.5:
		print('Warning: VRAM under 5.5 GB. Use lower max_seq_length and lower batch size.')
	elif total_gb < 7.0:
		print('Hint: 6 GB profile active. Keep max_seq_length=128 and fp16 enabled.')


def save_training_checkpoint(
	model: AutoModelForMaskedLM,
	tokenizer: AutoTokenizer,
	optimizer: AdamW,
	scheduler,
	scaler,
	output_dir: str,
	global_step: int,
) -> None:
	ckpt_dir = os.path.join(output_dir, f'model-step-{global_step}')
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


def load_text_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer):
	ds = load_dataset('csv', data_files=args.dataset_path, split='train')
	text_column = choose_text_column(ds.column_names, args.text_column)

	ds = ds.filter(
		lambda x: isinstance(x.get(text_column), str) and x.get(text_column).strip() != '',
		num_proc=args.num_proc,
	)
	if len(ds) == 0:
		raise ValueError('No valid text rows found in dataset after filtering empty values.')

	def tokenize_function(examples: Dict[str, list]) -> Dict[str, list]:
		return tokenizer(
			examples[text_column],
			truncation=True,
			max_length=args.max_seq_length,
			return_special_tokens_mask=True,
		)

	tokenized = ds.map(
		tokenize_function,
		batched=True,
		num_proc=args.num_proc,
		remove_columns=ds.column_names,
		desc='Tokenizing dataset',
	)

	tokenized.set_format(type='torch')
	return tokenized


def main():
	parser = argparse.ArgumentParser(description='Fine-tune DistilBERT (MLM) on custom corpus.csv')
	parser.add_argument('--dataset_path', type=str, default='./dataset/final_corpus.csv')
	parser.add_argument('--text_column', type=str, default='content')
	parser.add_argument('--model_name_or_path', type=str, default='distilbert-base-multilingual-cased')
	parser.add_argument('--output_dir', type=str, default='./bert/bert_output_final')
	parser.add_argument('--tokenizer_dir', type=str, default='./bert/tokenizer_final')
	parser.add_argument('--max_seq_length', type=int, default=128)
	parser.add_argument('--per_device_train_batch_size', type=int, default=8)
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=5e-5)
	parser.add_argument('--weight_decay', type=float, default=0.01)
	parser.add_argument('--num_train_epochs', type=int, default=3)
	parser.add_argument('--max_train_steps', type=int, default=-1)
	parser.add_argument('--warmup_ratio', type=float, default=0.06)
	parser.add_argument('--mlm_probability', type=float, default=0.15)
	parser.add_argument('--save_steps', type=int, default=500)
	parser.add_argument('--log_steps', type=int, default=20)
	parser.add_argument('--max_grad_norm', type=float, default=1.0)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--num_proc', type=int, default=1)
	parser.add_argument('--prefetch_factor', type=int, default=2)
	parser.add_argument('--no_pin_memory', action='store_true')
	parser.add_argument('--no_persistent_workers', action='store_true')
	parser.add_argument('--fp16', dest='fp16', action='store_true')
	parser.add_argument('--no_fp16', dest='fp16', action='store_false')
	parser.set_defaults(fp16=True)
	parser.add_argument('--gradient_checkpointing', dest='gradient_checkpointing', action='store_true')
	parser.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing', action='store_false')
	parser.set_defaults(gradient_checkpointing=True)
	args = parser.parse_args()

	validate_environment_and_args(args)

	os.makedirs(args.output_dir, exist_ok=True)
	os.makedirs(args.tokenizer_dir, exist_ok=True)

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

	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
	if tokenizer.mask_token_id is None:
		raise ValueError('Selected tokenizer has no [MASK] token, required for MLM training.')

	tokenizer.model_max_length = args.max_seq_length
	tokenizer.save_pretrained(args.tokenizer_dir)

	dataset = load_text_dataset(args, tokenizer)

	model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
	model.resize_token_embeddings(len(tokenizer))
	if args.gradient_checkpointing:
		model.gradient_checkpointing_enable()
	model.to(device)
	model.train()

	collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=True,
		mlm_probability=args.mlm_probability,
		pad_to_multiple_of=8 if device.type == 'cuda' else None,
		return_tensors='pt',
	)

	num_workers = max(args.num_workers, 0)
	pin_memory = (device.type == 'cuda') and (not args.no_pin_memory)
	dataloader_kwargs = {
		'dataset': dataset,
		'batch_size': args.per_device_train_batch_size,
		'shuffle': True,
		'collate_fn': collator,
		'num_workers': num_workers,
		'pin_memory': pin_memory,
	}
	if num_workers > 0:
		dataloader_kwargs['persistent_workers'] = not args.no_persistent_workers
		dataloader_kwargs['prefetch_factor'] = max(args.prefetch_factor, 1)
	dataloader = DataLoader(**dataloader_kwargs)

	optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

	updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
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

	print('=== DistilBERT Fine-tuning (MLM) ===')
	print(f'samples={len(dataset)} device={device.type} fp16={use_fp16}')
	print(f'batch_size={args.per_device_train_batch_size} grad_accum={args.gradient_accumulation_steps} effective_batch={args.per_device_train_batch_size * args.gradient_accumulation_steps}')
	print(f'max_seq_length={args.max_seq_length} total_steps={total_steps} warmup_steps={warmup_steps}')

	global_step = 0
	running_loss = 0.0
	optimizer.zero_grad(set_to_none=True)

	for epoch in range(args.num_train_epochs):
		for step, batch in enumerate(dataloader, start=1):
			batch = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items()}

			try:
				with torch.cuda.amp.autocast(enabled=use_fp16):
					outputs = model(**batch)
					loss = outputs.loss / args.gradient_accumulation_steps
			except RuntimeError as exc:
				if 'out of memory' in str(exc).lower() and device.type == 'cuda':
					torch.cuda.empty_cache()
					raise RuntimeError(
						'CUDA OOM during training. Try: --max_seq_length 96, '
						'--per_device_train_batch_size 4, --gradient_accumulation_steps 2.'
					) from exc
				raise

			if not torch.isfinite(loss):
				raise RuntimeError(f'Non-finite loss detected: {loss.item()}')

			scaler.scale(loss).backward()
			running_loss += loss.item() * args.gradient_accumulation_steps

			should_update = (step % args.gradient_accumulation_steps == 0)
			if not should_update:
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
				current_lr = scheduler.get_last_lr()[0]
				print(f'step={global_step} epoch={epoch + 1} loss={avg_loss:.4f} lr={current_lr:.8f}')
				running_loss = 0.0

			if global_step % args.save_steps == 0:
				save_training_checkpoint(model, tokenizer, optimizer, scheduler, scaler, args.output_dir, global_step)

			if args.max_train_steps > 0 and global_step >= args.max_train_steps:
				break

		if args.max_train_steps > 0 and global_step >= args.max_train_steps:
			break

	final_dir = os.path.join(args.output_dir, 'model-final')
	os.makedirs(final_dir, exist_ok=True)
	model.save_pretrained(final_dir)
	tokenizer.save_pretrained(final_dir)
	tokenizer.save_pretrained(args.tokenizer_dir)
	print(f'Final model saved to: {final_dir}')


if __name__ == '__main__':
	main()
