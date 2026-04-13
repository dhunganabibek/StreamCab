"""
Download NYC TLC yellow taxi parquet files (2015-2025).

Usage:
python scripts/download_tlc_data.py --output /Volumes/PNY/StreamCab
"""

import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import httpx

# url to fetch traffic data
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def all_months(start, end):
    cur = start
    while cur <= end:
        yield cur
        cur = datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip  {dest.name}")
        return

    print(f"  →     {dest.name}")
    tmp = dest.with_suffix(".part")

    for attempt in range(1, 8):
        try:
            with httpx.Client(
                timeout=httpx.Timeout(connect=15, read=300, write=None, pool=None)
            ) as client:
                with client.stream("GET", url, follow_redirects=True) as r:
                    r.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in r.iter_bytes(chunk_size=4 * 1024 * 1024):
                            f.write(chunk)
            tmp.rename(dest)
            print(f"  done  {dest.name}")
            return

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 404:
                print(f"  404   {dest.name} — skipping")
                return
            wait = (
                random.uniform(45, 90) * attempt
                if code in (403, 429)
                else random.uniform(2, 6) * attempt
            )
            print(f"  HTTP {code} — retrying in {wait:.0f}s (attempt {attempt}) …")
            time.sleep(wait)

        except Exception as e:
            wait = random.uniform(2, 6) * attempt
            print(f"  error: {e} — retrying in {wait:.0f}s …")
            time.sleep(wait)

        finally:
            if tmp.exists() and not dest.exists():
                tmp.unlink()

    print(f"  FAILED {dest.name}")


def main():
    parser = argparse.ArgumentParser(description="Download NYC yellow taxi parquet files")
    parser.add_argument("--output", required=True, help="Destination directory")
    parser.add_argument("--start", required=False, default="2022-01", help="Start date")
    parser.add_argument("--end", required=False, default="2024-12", help="End date")
    args = parser.parse_args()

    out = Path(args.output).expanduser().resolve()
    start = datetime.strptime(args.start, "%Y-%m")
    end = datetime.strptime(args.end, "%Y-%m")

    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)

    months = list(all_months(start, end))
    print(f"Downloading {len(months)} files to {out}\n")

    for m in months:
        name = f"yellow_tripdata_{m:%Y-%m}.parquet"
        download(f"{BASE_URL}/{name}", out / name)
        time.sleep(random.uniform(2, 4))  # polite delay between requests


if __name__ == "__main__":
    main()
