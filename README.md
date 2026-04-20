# Treinamento Electra Small (local)

Prerequisitos:

- GPU recomendada (CUDA)
- Python 3.8+
- Instalar dependências: `pip install -r requirements.txt`

Uso básico (treina tokenizer e continua pré-treinamento):

```bash
python train_electra.py --dataset_path corpus.csv --train_tokenizer --tokenizer_dir ./tokenizer --output_dir ./electra_output \
  --per_device_train_batch_size 8 --num_train_epochs 3 --max_seq_length 128
```

Uso sem treinar tokenizer (usa tokenizer do modelo gerador especificado):

```bash
python train_electra.py --dataset_path corpus.csv --output_dir ./electra_output \
  --per_device_train_batch_size 8 --num_train_epochs 3
```

Notas:

- O script realiza pré-treinamento no objetivo ELECTRA (gerador + discriminador) em loop customizado.
- Ajuste `--per_device_train_batch_size`, `--num_train_epochs` e `--max_seq_length` conforme sua GPU.
- Este repositório grava checkpoints em `--output_dir` (p.ex. `generator-final` e `discriminator-final`).
- Não executei o treinamento — é necessário rodar manualmente por custo computacional.

Calibracao de threshold (CSV rotulado):

```bash
python evaluate_threshold.py --dataset_path seu_dataset_rotulado.csv --text_column content --label_column label --discriminator_dir ./electra_output/discriminator-final --tokenizer_dir ./electra_output --batch_size 32 --save_scored_csv scored_output.csv
```

Observacoes:

- Labels esperados por padrao: `0,real,verdadeiro` para REAL e `1,fake,falso` para FAKE.
- O script testa todos os thresholds possiveis com base nos scores e escolhe o melhor por acuracia (desempate por F1/precision/recall).
- Se quiser mapear outros nomes de label, use `--fake_values` e `--real_values`.

Inferencia de noticia de saude:

```bash
python infer_fake.py --text "NOTICIA AQUI"
```
