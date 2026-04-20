import argparse
import torch
from transformers import PreTrainedTokenizerFast, ElectraForPreTraining


def score_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Electra discriminator logits: higher -> more likely 'replaced'
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        # ignore special tokens (assume tokenizer uses pad/bos/eos)
        mask = attention_mask.bool()
        token_probs = probs[0, :mask.sum()].cpu()
        mean_prob = float(token_probs.mean().item())
    return mean_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default=None, help='Texto a avaliar')
    parser.add_argument('--file', type=str, default=None, help='Arquivo de texto (uma linha por exemplo)')
    parser.add_argument('--threshold', type=float, default=0.16, help='Limiar médio para considerar "fake"')
    parser.add_argument('--discriminator_dir', type=str, default='./electra_output/discriminator-final')
    parser.add_argument('--tokenizer_dir', type=str, default='./electra_output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    model = ElectraForPreTraining.from_pretrained(args.discriminator_dir).to(device)
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
        print('Nenhum texto fornecido. Use --text ou --file')
        return

    for i, s in enumerate(samples, 1):
        score = score_text(s, tokenizer, model, device)
        label = 'FAKE' if score >= args.threshold else 'REAL'
        print(f'[{i}] mean_replaced_prob={score:.4f} -> {label}\n{s}\n')


if __name__ == '__main__':
    main()
