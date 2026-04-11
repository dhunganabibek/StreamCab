"""Convert TLC parquet files to CSV.

Usage:
    python scripts/convert-to-csv.py --input data/raw-data/parquet --output data/raw-data/csv
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def convert(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in input_dir.glob("*.parquet") if not f.name.startswith("._"))
    if not files:
        print(f"No parquet files found in {input_dir}")
        return

    print(f"Converting {len(files)} file(s) from {input_dir} → {output_dir}\n")

    for parquet_file in files:
        csv_file = output_dir / f"{parquet_file.stem}.csv"
        if csv_file.exists():
            print(f"  skip  {parquet_file.name} (already converted)")
            continue

        print(f"  converting {parquet_file.name} …")
        pf = pq.ParquetFile(parquet_file)
        first_batch = True
        with csv_file.open("w", newline="", encoding="utf-8") as f:
            for batch in pf.iter_batches(batch_size=100_000):
                batch.to_pandas().to_csv(f, index=False, header=first_batch)
                first_batch = False
        print(f"  done  → {csv_file.name}")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TLC parquet files to CSV")
    parser.add_argument("--input", required=True, help="Directory containing .parquet files")
    parser.add_argument("--output", required=True, help="Directory to write .csv files")
    args = parser.parse_args()

    convert(Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve())


if __name__ == "__main__":
    main()
