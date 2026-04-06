#!/usr/bin/env python3
"""Download NYC TLC parquet files to a local directory or external drive.

Requires:  pip install httpx

Features
--------
- Streaming chunk writes — files are never loaded into RAM (safe for 500 GB+)
- Concurrent downloads via --parallel
- Automatic retry with exponential back-off
- Per-file progress (bytes received / total) while downloading

Examples
--------
# 3 years of yellow taxi → data/raw  (~30–60 GB)
  python scripts/download_tlc_data.py \\
      --taxi-type yellow --start 2022-01 --end 2024-12 --output data/raw

# External drive, 4 workers
  python scripts/download_tlc_data.py \\
      --taxi-type yellow --start 2019-01 --end 2024-12 \\
      --output /Volumes/MyExternalDrive/nyc_taxi --parallel 4

# Dry-run
  python scripts/download_tlc_data.py \\
      --taxi-type yellow --start 2024-01 --end 2024-06 --output data/raw --dry-run

After downloading
-----------------
  Train model on historical data:  docker compose run --rm -e ONE_SHOT=true trainer
  Start live pipeline:             docker compose up --build
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB chunks — good balance for large parquet files
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds; wait = base ** attempt


def parse_month(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m")
    except ValueError as ex:
        raise argparse.ArgumentTypeError("Month must be YYYY-MM format") from ex


def month_iter(start: datetime, end: datetime):
    current = datetime(start.year, start.month, 1)
    end_month = datetime(end.year, end.month, 1)
    while current <= end_month:
        yield current
        current = (
            datetime(current.year + 1, 1, 1)
            if current.month == 12
            else datetime(current.year, current.month + 1, 1)
        )


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def _download_streaming(client: httpx.Client, url: str, target: Path) -> str:
    """Stream-download *url* to *target*, returning a human-readable size string."""
    tmp = target.with_suffix(".part")
    received = 0

    with client.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=CHUNK_SIZE):
                fh.write(chunk)
                received += len(chunk)
                if total:
                    pct = received / total * 100
                    print(
                        f"    {target.name}  {_fmt_bytes(received)}/{_fmt_bytes(total)} ({pct:.0f}%)",
                        end="\r",
                    )

    tmp.rename(target)
    print(f"    {target.name}  {_fmt_bytes(received)} — done          ")
    return _fmt_bytes(received)


def download_one(url: str, target: Path, dry_run: bool, overwrite: bool) -> tuple[str, str]:
    """Download a single file. Returns (file_name, status_string)."""
    if target.exists() and not overwrite:
        return target.name, "skipped"

    if dry_run:
        return target.name, "dry-run"

    with httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=None, pool=None)) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                size = _download_streaming(client, url, target)
                return target.name, f"downloaded ({size})"
            except httpx.HTTPStatusError as ex:
                if ex.response.status_code == 404:
                    return target.name, f"FAILED: 404 not found — {url}"
                if attempt == MAX_RETRIES:
                    return target.name, f"FAILED: HTTP {ex.response.status_code}"
            except (httpx.RequestError, OSError) as ex:
                if attempt == MAX_RETRIES:
                    return target.name, f"FAILED: {ex}"
            # Clean up partial file before retry
            tmp = target.with_suffix(".part")
            if tmp.exists():
                tmp.unlink()
            time.sleep(RETRY_BACKOFF_BASE**attempt)

    return target.name, "FAILED: max retries exceeded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NYC TLC monthly parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--taxi-type",
        choices=["yellow", "green", "fhv", "fhvhv"],
        default="yellow",
        help="Taxi type (default: yellow)",
    )
    parser.add_argument("--start", type=parse_month, required=True, help="Start month YYYY-MM")
    parser.add_argument("--end", type=parse_month, required=True, help="End month YYYY-MM")
    parser.add_argument(
        "--output",
        required=True,
        help="Target directory (e.g. data/raw or /Volumes/MyDrive/nyc_taxi)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        metavar="N",
        help="Parallel download workers (default: 2, max recommended: 4)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.end < args.start:
        raise SystemExit("--end must be >= --start")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    months = list(month_iter(args.start, args.end))
    tasks = [
        (
            f"{BASE_URL}/{args.taxi_type}_tripdata_{m.year:04d}-{m.month:02d}.parquet",
            output_dir / f"{args.taxi_type}_tripdata_{m.year:04d}-{m.month:02d}.parquet",
        )
        for m in months
    ]

    print(f"Target   : {output_dir}")
    print(f"Type     : {args.taxi_type}")
    print(f"Range    : {args.start:%Y-%m} → {args.end:%Y-%m}  ({len(tasks)} files)")
    print(f"Workers  : {args.parallel}")
    if args.dry_run:
        print("Mode     : DRY RUN\n")
        for _, target in tasks:
            print(f"  {target.name}")
        return
    print()

    results: dict[str, str] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(download_one, url, target, args.dry_run, args.overwrite): target.name
            for url, target in tasks
        }
        for future in as_completed(futures):
            name, status = future.result()
            results[name] = status
            done += 1
            ok = "downloaded" in status
            tag = "[ok]" if ok else f"[{status.split(':')[0].strip().lower()}]"
            print(f"  {tag:10s} {name}  —  {status}  ({done}/{len(tasks)})")

    n_ok = sum(1 for s in results.values() if "downloaded" in s)
    n_skip = sum(1 for s in results.values() if s == "skipped")
    n_fail = sum(1 for s in results.values() if "FAILED" in s)

    print(f"\nDone  —  downloaded: {n_ok}  skipped: {n_skip}  failed: {n_fail}")

    if n_ok:
        total_bytes = sum(
            (output_dir / name).stat().st_size
            for name, status in results.items()
            if "downloaded" in status and (output_dir / name).exists()
        )
        print(f"Total downloaded size: {_fmt_bytes(total_bytes)}")

    if n_fail:
        print("\nFailed files:")
        for name, status in results.items():
            if "FAILED" in status:
                print(f"  {name}: {status}")

    if n_ok:
        print(
            "\nNext steps:\n"
            "  Train on historical data:  docker compose run --rm -e ONE_SHOT=true trainer\n"
            "  Start live pipeline:       docker compose up --build"
        )


if __name__ == "__main__":
    main()
