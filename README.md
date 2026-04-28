# Treinamento Electra Small (local)

Prerequisitos:

- GPU recomendada (CUDA)
- Python 3.8+
- Instalar dependências: `pip install -r requirements.txt`

Uso básico (treina tokenizer e continua pré-treinamento):

```bash
python electra/train_electra.py --dataset_path ./dataset/corpus.csv --train_tokenizer --tokenizer_dir ./electra/tokenizer --output_dir ./electra/electra_output \
  --per_device_train_batch_size 8 --num_train_epochs 3 --max_seq_length 128
```

Uso sem treinar tokenizer (usa tokenizer do modelo gerador especificado):

```bash
python electra/train_electra.py --dataset_path ./dataset/corpus.csv --output_dir ./electra/electra_output \
  --per_device_train_batch_size 8 --num_train_epochs 3
```

Notas:

- O script realiza pré-treinamento no objetivo ELECTRA (gerador + discriminador) em loop customizado.
- Ajuste `--per_device_train_batch_size`, `--num_train_epochs` e `--max_seq_length` conforme sua GPU.
- Este repositório grava checkpoints em `--output_dir` (p.ex. `generator-final` e `discriminator-final`).
- Não executei o treinamento — é necessário rodar manualmente por custo computacional.

Calibracao de threshold (CSV rotulado):

```bash
python electra/evaluate_threshold.py --dataset_path seu_dataset_rotulado.csv --text_column content --label_column label --discriminator_dir ./electra/electra_output/discriminator-final --tokenizer_dir ./electra/electra_output --batch_size 32 --save_scored_csv scored_output.csv
```

Observacoes:

- Labels esperados por padrao: `0,real,verdadeiro` para REAL e `1,fake,falso` para FAKE.
- O script testa todos os thresholds possiveis com base nos scores e escolhe o melhor por acuracia (desempate por F1/precision/recall).
- Se quiser mapear outros nomes de label, use `--fake_values` e `--real_values`.

Inferencia de noticia de saude:

```bash
python electra/infer_fake.py --text "NOTICIA AQUI"
```

Matriz de confusao (CSV rotulado):

```bash
python electra/confusion_matrix_electra.py --dataset_path seu_dataset_rotulado.csv --text_column content --label_column label --save_plot cm_electra.png --save_scored_csv scored_electra.csv
```

## BERT (DistilBERT) - Fluxo equivalente ao Electra

Passo 1 - Continual pretraining (MLM) sem labels:

```bash
python bert/train_bert.py --dataset_path ./dataset/corpus.csv --text_column content --output_dir ./bert/bert_output --tokenizer_dir ./bert/tokenizer --per_device_train_batch_size 8 --max_seq_length 128 --num_train_epochs 3 --fp16 --gradient_checkpointing
```

Passo 2 - Calibrar threshold com CSV rotulado (opcional, recomendado para definir corte):

```bash
python bert/evaluate_treshold.py --dataset_path seu_dataset_rotulado.csv --text_column content --label_column label --model_dir ./bert/bert_output/model-final --tokenizer_dir ./bert/tokenizer --batch_size 32 --max_length 128 --mask_stride 7 --save_scored_csv scored_bert_mlm.csv
```

Passo 3 - Matriz de confusao com o threshold:

```bash
python bert/confusion_matrix_bert.py --dataset_path seu_dataset_rotulado.csv --text_column content --label_column label --model_dir ./bert/bert_output/model-final --tokenizer_dir ./bert/tokenizer --save_plot cm_bert_mlm.png --save_scored_csv scored_bert_cm.csv
```

Passo 4 - Inferencia (usa score MLM + threshold):

```bash
python bert/infer_fake.py --model_dir ./bert/bert_output/model-final --tokenizer_dir ./bert/tokenizer --threshold 6.0 --text "NOTICIA AQUI"
```

Observacoes:

- Nesse fluxo, o score do BERT e baseado em perda MLM (cross-entropy mascarada): score maior tende a indicar texto mais "fake".
- O valor ideal de threshold deve ser calibrado no seu dataset rotulado com `bert/evaluate_treshold.py`.
- Se voce quiser classificacao supervisionada tradicional (com cabeca de classificacao), o script opcional e `bert/train_bert_supervised.py`.
