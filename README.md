# Treinamento Electra Small (local)

Prerequisitos:

- GPU recomendada (CUDA)
- Python 3.8+
- Instalar dependências: `pip install -r requirements.txt`

Uso básico (treina tokenizer e continua pré-treinamento):

```bash
python electra/train_electra.py --dataset_path ./dataset/dataset_labeled.csv --train_tokenizer --tokenizer_dir ./electra/tokenizer --output_dir ./electra/electra_output \
  --per_device_train_batch_size 8 --num_train_epochs 3 --max_seq_length 128
```

No Windows, rode em uma linha só (ou use o caractere de continuação correto do seu terminal). Exemplos:

PowerShell:

```powershell
python .\electra\train_electra.py `
  --dataset_path .\dataset\dataset_labeled.csv `
  --train_tokenizer `
  --tokenizer_dir .\electra\tokenizer `
  --output_dir .\electra\electra_output `
  --per_device_train_batch_size 8 `
  --num_train_epochs 3 `
  --max_seq_length 128
```

CMD:

```bat
python electra\train_electra.py ^
  --dataset_path dataset\dataset_labeled.csv ^
  --train_tokenizer ^
  --tokenizer_dir electra\tokenizer ^
  --output_dir electra\electra_output ^
  --per_device_train_batch_size 8 ^
  --num_train_epochs 3 ^
  --max_seq_length 128
```

Uso sem treinar tokenizer (usa tokenizer do modelo gerador especificado):

```bash
python electra/train_electra.py --dataset_path ./dataset/dataset_labeled.csv --output_dir ./electra/electra_output \
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

Exemplo com o dataset gerado (labels `fake`/`true`) salvando o melhor threshold em JSON (CMD):

```bat
python electra\evaluate_threshold.py --dataset_path dataset\dataset_labeled.csv --text_column content --label_column label --fake_values fake --real_values true --discriminator_dir electra\electra_output\discriminator-final --tokenizer_dir electra\electra_output --save_threshold_json electra\threshold.json
```

Gerar uma tabela com métricas para cada threshold (útil para escolher um corte com menos falsos positivos) (CMD):

```bat
python electra\evaluate_threshold.py --dataset_path dataset\dataset_labeled.csv --text_column content --label_column label --fake_values fake --real_values true --discriminator_dir electra\electra_output\discriminator-final --tokenizer_dir electra\electra_output --save_threshold_curve_csv electra\threshold_curve.csv
```

Usar o threshold salvo na inferência (CMD):

```bat
python electra\infer_fake.py --text "NOTICIA AQUI" --discriminator_dir electra\electra_output\discriminator-final --tokenizer_dir electra\electra_output --threshold_json electra\threshold.json
```

Observacoes:

- Labels esperados por padrao: `0,real,verdadeiro` para REAL e `1,fake,falso` para FAKE.
- O script testa todos os thresholds possiveis com base nos scores e escolhe o melhor por acuracia (desempate por F1/precision/recall).
- Se quiser mapear outros nomes de label, use `--fake_values` e `--real_values`.

## Gerar dataset rotulado (fake/true)

Para unir `dataset/corpus.csv` (fake) + `dataset/true_news.csv` (true) em um novo CSV com a coluna `label`:

```bash
python dataset/merge_labeled_dataset.py --output_path dataset/dataset_labeled.csv
```

Inferencia de noticia de saude:

```bash
python electra/infer_fake.py --discriminator_dir electra/electra_output/discriminator-final --tokenizer_dir electra/electra_output --threshold 0.056245 --text "COLE A NOTICIA AQUI"
```

Matriz de confusao (CSV rotulado):

```bash
python electra/confusion_matrix_electra.py --dataset_path seu_dataset_rotulado.csv --text_column content --label_column label --save_plot cm_electra.png --save_scored_csv scored_electra.csv
```

## Treinamento distilBert (local)

Modelo usado: DistilBERT (`distilbert-base-multilingual-cased`).

Passo 1 - Continual pretraining (MLM) sem labels:

```bash
python bert/train_bert.py --dataset_path ./dataset/dataset_labeled.csv --text_column content --output_dir ./bert/bert_output --tokenizer_dir ./bert/tokenizer --per_device_train_batch_size 8 --max_seq_length 128 --num_train_epochs 3 --fp16 --gradient_checkpointing
```

Passo 2 - Calibrar threshold com CSV rotulado (opcional, recomendado para definir corte):

```bash
python bert/evaluate_treshold.py --dataset_path seu_dataset_rotulado.csv --text_column content --label_column label --model_dir ./bert/bert_output/model-final --tokenizer_dir ./bert/tokenizer --batch_size 32 --max_length 128 --mask_stride 7 --save_scored_csv scored_bert_mlm.csv
```

Passo 3 - Matriz de confusao com o threshold:

```bash
python bert/confusion_matrix_bert.py --dataset_path seu_dataset_rotulado.csv --text_column content --label_column label --model_dir ./bert/bert_output/model-final --tokenizer_dir ./bert/tokenizer --save_plot cm_bert_mlm.png --save_scored_csv scored_bert_cm.csv
```

Passo 4 - Inferencia de noticia de saude (usa score MLM + threshold):

```bash
python bert/infer_fake.py --model_dir ./bert/bert_output/model-final --tokenizer_dir ./bert/tokenizer --max_length 128 --mask_stride 7 --text "COLE A NOTICIA AQUI"
```

Observacoes:

- Nesse fluxo, o score do BERT e baseado em perda MLM (cross-entropy mascarada): score maior tende a indicar texto mais "fake".
- O valor ideal de threshold deve ser calibrado no seu dataset rotulado com `bert/evaluate_treshold.py`.
- No codigo atual, o `bert/infer_fake.py` ja vem com um threshold padrao (0.023989). Se quiser sobrescrever, use `--threshold`.
