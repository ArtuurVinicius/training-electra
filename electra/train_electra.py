import argparse
import os
import math
from itertools import chain
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    PreTrainedTokenizerFast,
    RobertaTokenizerFast,
    ElectraTokenizerFast,
    ElectraConfig,
    ElectraForPreTraining,
    ElectraForMaskedLM,
    get_scheduler,
)


def mask_tokens(inputs, special_tokens_mask, tokenizer, mlm_probability):
    labels = inputs.clone()
    rand = torch.rand(labels.shape, device=labels.device)
    masked_indices = (rand < mlm_probability) & ~special_tokens_mask.bool()
    labels[~masked_indices] = -100
    if tokenizer.mask_token_id is None:
        raise ValueError('Tokenizer has no mask token.')
    probability_matrix = torch.rand(labels.shape, device=labels.device)
    mask_token_id = tokenizer.mask_token_id
    indices_replaced = (probability_matrix < 0.8) & masked_indices
    inputs[indices_replaced] = mask_token_id
    indices_random = (probability_matrix >= 0.8) & (probability_matrix < 0.9) & masked_indices
    if indices_random.any():
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        inputs[indices_random] = random_words[indices_random]
    return inputs, labels


def group_texts(examples, max_seq_length):
    concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
    total_length = len(concatenated['input_ids'])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {}
    for k, v in concatenated.items():
        result[k] = [v[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
    return result


def collate_fn(examples):
    input_ids = torch.stack([
        e['input_ids'] if isinstance(e['input_ids'], torch.Tensor) else torch.tensor(e['input_ids'], dtype=torch.long)
        for e in examples
    ])
    attention_mask = torch.stack([
        e['attention_mask'] if isinstance(e.get('attention_mask'), torch.Tensor)
        else torch.tensor(e.get('attention_mask', [1] * len(e['input_ids'])), dtype=torch.long)
        for e in examples
    ])
    special_tokens_mask = torch.stack([
        e['special_tokens_mask'] if isinstance(e['special_tokens_mask'], torch.Tensor)
        else torch.tensor(e['special_tokens_mask'], dtype=torch.bool)
        for e in examples
    ]).bool()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'special_tokens_mask': special_tokens_mask}


def build_fast_tokenizer_from_bpe_files(tokenizer_dir, max_seq_length):
    vocab_file = os.path.join(tokenizer_dir, 'vocab.json')
    merges_file = os.path.join(tokenizer_dir, 'merges.txt')
    tokenizer_json_file = os.path.join(tokenizer_dir, 'tokenizer.json')
    bpe_from_files = ByteLevelBPETokenizer(vocab_file, merges_file)
    bpe_from_files.save(tokenizer_json_file)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json_file,
        bos_token='<s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>',
        model_max_length=max_seq_length,
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset/dataset_labeled.csv')
    parser.add_argument('--text_column', type=str, default='content')
    parser.add_argument('--output_dir', type=str, default='./electra/electra_output')
    parser.add_argument('--tokenizer_dir', type=str, default='./electra/tokenizer')
    parser.add_argument('--train_tokenizer', action='store_true')
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--max_train_steps', type=int, default=-1)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--no_pin_memory', action='store_true')
    parser.add_argument('--no_persistent_workers', action='store_true')
    parser.add_argument('--generator_model_name_or_path', type=str, default='google/electra-small-generator')
    parser.add_argument('--discriminator_model_name_or_path', type=str, default='google/electra-small-discriminator')
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--gen_loss_weight', type=float, default=1.0)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    if args.train_tokenizer:
        ds = load_dataset('csv', data_files=args.dataset_path, split='train')
        tmp_txt = os.path.join(args.tokenizer_dir, 'train.txt')
        os.makedirs(args.tokenizer_dir, exist_ok=True)
        with open(tmp_txt, 'w', encoding='utf-8') as f:
            for row in ds:
                txt = row.get(args.text_column) or row.get('text') or row.get('content')
                if txt:
                    f.write(txt.replace('\n', ' ') + '\n')
        import json
        tokenizer_obj = ByteLevelBPETokenizer()
        tokenizer_obj.train(files=[tmp_txt], vocab_size=args.vocab_size, min_frequency=2, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
        tokenizer_obj.save_model(args.tokenizer_dir)
        tokenizer = build_fast_tokenizer_from_bpe_files(args.tokenizer_dir, args.max_seq_length)
        if len(tokenizer) <= 5:
            raise RuntimeError('Tokenizer treinado ficou com vocabulário inválido (<= 5 tokens).')
        tokenizer.save_pretrained(args.tokenizer_dir)
    else:
        custom_tokenizer_json = os.path.join(args.tokenizer_dir, 'tokenizer.json')
        custom_vocab = os.path.join(args.tokenizer_dir, 'vocab.json')
        custom_merges = os.path.join(args.tokenizer_dir, 'merges.txt')
        try:
            if os.path.exists(custom_tokenizer_json):
                tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir, model_max_length=args.max_seq_length)
                if len(tokenizer) <= 5 and os.path.exists(custom_vocab) and os.path.exists(custom_merges):
                    tokenizer = build_fast_tokenizer_from_bpe_files(args.tokenizer_dir, args.max_seq_length)
                    tokenizer.save_pretrained(args.tokenizer_dir)
            else:
                tokenizer = ElectraTokenizerFast.from_pretrained(args.generator_model_name_or_path)
        except Exception:
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', model_max_length=args.max_seq_length)

    ds = load_dataset('csv', data_files=args.dataset_path, split='train')
    if args.text_column not in ds.column_names:
        possible = [c for c in ['content', 'text', 'sentence'] if c in ds.column_names]
        if not possible:
            raise ValueError('Texto não encontrado no CSV; especifique --text_column corretamente')
        args.text_column = possible[0]

    def tokenize_function(examples):
        return tokenizer(examples[args.text_column], return_special_tokens_mask=True)

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    grouped = tokenized.map(lambda ex: group_texts(ex, args.max_seq_length), batched=True)
    grouped.set_format(type='torch')

    num_workers = max(args.num_workers, 0)
    pin_memory = (device.type == 'cuda') and (not args.no_pin_memory)
    dataloader_kwargs = {
        'dataset': grouped,
        'batch_size': args.per_device_train_batch_size,
        'shuffle': True,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = not args.no_persistent_workers
        dataloader_kwargs['prefetch_factor'] = max(args.prefetch_factor, 1)
    dataloader = DataLoader(**dataloader_kwargs)

    if args.from_scratch:
        vocab_size = len(tokenizer)
        gen_config = ElectraConfig(vocab_size=vocab_size)
        disc_config = ElectraConfig(vocab_size=vocab_size)
        generator = ElectraForMaskedLM(gen_config)
        discriminator = ElectraForPreTraining(disc_config)
    else:
        generator = ElectraForMaskedLM.from_pretrained(args.generator_model_name_or_path)
        discriminator = ElectraForPreTraining.from_pretrained(args.discriminator_model_name_or_path)
        if len(tokenizer) != generator.config.vocab_size:
            generator.resize_token_embeddings(len(tokenizer))
        if len(tokenizer) != discriminator.config.vocab_size:
            discriminator.resize_token_embeddings(len(tokenizer))

    generator.to(device)
    discriminator.to(device)

    optimizer = AdamW(list(generator.parameters()) + list(discriminator.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = math.ceil(len(dataloader) * args.num_train_epochs)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    global_step = 0
    generator.train()
    discriminator.train()
    optimizer.zero_grad(set_to_none=True)

    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == 'cuda' else None

    for epoch in range(args.num_train_epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=pin_memory)
            attention_mask = batch['attention_mask'].to(device, non_blocking=pin_memory)
            special_tokens_mask = batch['special_tokens_mask'].to(device, non_blocking=pin_memory)
            original_input_ids = input_ids.clone()

            masked_input_ids, gen_labels = mask_tokens(input_ids.clone(), special_tokens_mask, tokenizer, args.mlm_probability)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    gen_outputs = generator(input_ids=masked_input_ids, attention_mask=attention_mask, labels=gen_labels)
                    gen_loss = gen_outputs.loss
            else:
                gen_outputs = generator(input_ids=masked_input_ids, attention_mask=attention_mask, labels=gen_labels)
                gen_loss = gen_outputs.loss

            with torch.no_grad():
                gen_logits = gen_outputs.logits
                masked_positions = gen_labels != -100
                if masked_positions.sum() == 0:
                    continue
                coords = masked_positions.nonzero(as_tuple=True)
                selected_logits = gen_logits[coords]
                probs = torch.softmax(selected_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                corrupted_input_ids = original_input_ids.clone()
                corrupted_input_ids[coords] = sampled_tokens

            disc_labels = (corrupted_input_ids != original_input_ids).long()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    disc_outputs = discriminator(input_ids=corrupted_input_ids, attention_mask=attention_mask, labels=disc_labels)
                    disc_loss = disc_outputs.loss
                    loss = disc_loss + args.gen_loss_weight * gen_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                disc_outputs = discriminator(input_ids=corrupted_input_ids, attention_mask=attention_mask, labels=disc_labels)
                disc_loss = disc_outputs.loss
                loss = disc_loss + args.gen_loss_weight * gen_loss
                loss.backward()
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step % 20 == 0:
                print(f'step={global_step} loss={loss.item():.4f} gen_loss={gen_loss.item():.4f} disc_loss={disc_loss.item():.4f}')

            if global_step % args.save_steps == 0:
                gen_path = os.path.join(args.output_dir, f'generator-step-{global_step}')
                disc_path = os.path.join(args.output_dir, f'discriminator-step-{global_step}')
                os.makedirs(gen_path, exist_ok=True)
                os.makedirs(disc_path, exist_ok=True)
                generator.save_pretrained(gen_path)
                discriminator.save_pretrained(disc_path)
                tokenizer.save_pretrained(args.output_dir)

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    generator.save_pretrained(os.path.join(args.output_dir, 'generator-final'))
    discriminator.save_pretrained(os.path.join(args.output_dir, 'discriminator-final'))
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
