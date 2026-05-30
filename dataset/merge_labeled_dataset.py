import argparse
import csv
from pathlib import Path
from typing import Iterable


def _read_fieldnames(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV sem header: {csv_path}")
        return list(reader.fieldnames)


def _iter_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV sem header: {csv_path}")
        for row in reader:
            yield row


def merge_with_labels(
    fake_csv: Path,
    true_csv: Path,
    output_csv: Path,
    fake_label: str = "fake",
    true_label: str = "true",
) -> None:
    if not fake_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {fake_csv}")
    if not true_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {true_csv}")

    fake_fields = _read_fieldnames(fake_csv)
    true_fields = _read_fieldnames(true_csv)

    all_fields: list[str] = []
    for name in fake_fields + true_fields:
        if name not in all_fields:
            all_fields.append(name)

    if "label" not in all_fields:
        all_fields.append("label")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()

        for row in _iter_rows(fake_csv):
            row_out = {**row, "label": fake_label}
            writer.writerow(row_out)

        for row in _iter_rows(true_csv):
            row_out = {**row, "label": true_label}
            writer.writerow(row_out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Une dois CSVs (fake + true) em um único dataset, adicionando a coluna 'label'. "
            "Não altera os arquivos de entrada."
        )
    )
    parser.add_argument(
        "--fake_path",
        default=str(Path("dataset") / "corpus.csv"),
        help="Caminho para o CSV de fakes (default: dataset/corpus.csv)",
    )
    parser.add_argument(
        "--true_path",
        default=str(Path("dataset") / "true_news.csv"),
        help="Caminho para o CSV de true news (default: dataset/true_news.csv)",
    )
    parser.add_argument(
        "--output_path",
        default=str(Path("dataset") / "dataset_labeled.csv"),
        help="Caminho do CSV de saída (default: dataset/dataset_labeled.csv)",
    )
    parser.add_argument(
        "--fake_label",
        default="fake",
        help="Valor da label para linhas vindas do fake_path (default: fake)",
    )
    parser.add_argument(
        "--true_label",
        default="true",
        help="Valor da label para linhas vindas do true_path (default: true)",
    )

    args = parser.parse_args()

    merge_with_labels(
        fake_csv=Path(args.fake_path),
        true_csv=Path(args.true_path),
        output_csv=Path(args.output_path),
        fake_label=args.fake_label,
        true_label=args.true_label,
    )

    print(f"OK: gerado {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
