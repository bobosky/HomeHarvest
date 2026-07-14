"""
Nationwide US property scraper built on HomeHarvest.

Iterates every US zip code (downloaded automatically from GeoNames) and pulls
all listings for the requested listing types, with checkpointing so the run
can be stopped and resumed at any time.

Realtor.com caps any single search at 10,000 results, which is why the
country must be swept zip-by-zip. Zips that still hit the cap are
automatically re-fetched in price bands.

Usage:
    # Scrape all for-sale listings nationwide (resumable)
    python scrape_us_nationwide.py scrape --output data_us

    # Scrape multiple listing types, restrict to states, more workers
    python scrape_us_nationwide.py scrape --listing-types for_sale,sold \
        --states AZ,CA,TX --workers 6 --output data_us

    # Test drive on 20 zips
    python scrape_us_nationwide.py scrape --max-zips 20 --output data_test

    # Merge everything scraped so far into one deduplicated file
    python scrape_us_nationwide.py combine --output data_us --csv

    # Show progress
    python scrape_us_nationwide.py status --output data_us

Notes:
    - A full national sweep is ~41,000 zip codes per listing type. At ~1
      request/second expect it to take a few days per listing type; use
      --states to shard the job across machines/proxies if needed.
    - --extra-data enriches each property with agent/broker/tax data but
      multiplies the request count; leave it off for bulk sweeps.
    - Use --proxy to route traffic through a proxy if you start seeing 403s.
"""

import argparse
import io
import sys
import time
import threading
import traceback
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from homeharvest import scrape_property

GEONAMES_URL = "https://download.geonames.org/export/zip/US.zip"
RESULT_CAP = 10000  # realtor.com hard limit per search

# Price bands used to subdivide zips that hit the 10k cap.
SALE_BANDS = [0, 100_000, 200_000, 300_000, 400_000, 500_000, 750_000,
              1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000, None]
RENT_BANDS = [0, 1_000, 1_500, 2_000, 2_500, 3_000, 4_000, 5_000, 10_000, None]

print_lock = threading.Lock()


def log(msg: str):
    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- zip codes

def load_zip_codes(cache_dir: Path, states: list[str] | None) -> list[tuple[str, str]]:
    """Return [(zip, state_code), ...] for the US, cached from GeoNames."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "us_zips.tsv"
    if not cache.exists():
        log(f"Downloading US zip code list from {GEONAMES_URL} ...")
        with urllib.request.urlopen(GEONAMES_URL, timeout=120) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            raw = zf.read("US.txt").decode("utf-8")
        rows = []
        for line in raw.splitlines():
            parts = line.split("\t")
            if len(parts) >= 5 and parts[1] and parts[4]:
                rows.append(f"{parts[1]}\t{parts[4]}")
        cache.write_text("\n".join(sorted(set(rows))))
        log(f"Cached {len(set(rows))} zip codes to {cache}")

    zips = []
    for line in cache.read_text().splitlines():
        zipcode, state = line.split("\t")
        if not states or state in states:
            zips.append((zipcode, state))
    return zips


# --------------------------------------------------------------- checkpoint

class Checkpoint:
    """Append-only record of completed (zip, listing_type) units."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.done: set[str] = set()
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.done.add(line.split(",")[0] + "," + line.split(",")[1])

    def is_done(self, zipcode: str, listing_type: str) -> bool:
        return f"{zipcode},{listing_type}" in self.done

    def mark(self, zipcode: str, listing_type: str, rows: int):
        with self.lock:
            self.done.add(f"{zipcode},{listing_type}")
            with self.path.open("a") as f:
                f.write(f"{zipcode},{listing_type},{rows}\n")


# ------------------------------------------------------------- rate limiter

class RateLimiter:
    """Enforces a minimum interval between request starts, across threads."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.min_interval
        if delay > 0:
            time.sleep(delay)


# ----------------------------------------------------------------- scraping

def fetch(zipcode: str, listing_type: str, args, limiter: RateLimiter,
          price_min=None, price_max=None, depth=0) -> pd.DataFrame:
    """Fetch one zip/listing_type, subdividing by price if the cap is hit."""
    limiter.wait()
    last_err = None
    for attempt in range(args.retries):
        try:
            df = scrape_property(
                location=zipcode,
                listing_type=listing_type,
                extra_property_data=args.extra_data,
                past_days=args.past_days,
                proxy=args.proxy,
                price_min=price_min,
                price_max=price_max,
                limit=RESULT_CAP,
            )
            break
        except Exception as e:
            last_err = e
            wait = min(300, 15 * 2 ** attempt)
            log(f"  {zipcode}/{listing_type}: attempt {attempt + 1} failed "
                f"({type(e).__name__}: {e}), retrying in {wait}s")
            time.sleep(wait)
            limiter.wait()
    else:
        raise last_err

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Hit the cap: subdivide by price so nothing is silently dropped.
    if len(df) >= RESULT_CAP:
        if depth == 0:
            bands = RENT_BANDS if listing_type == "for_rent" else SALE_BANDS
            log(f"  {zipcode}/{listing_type}: hit {RESULT_CAP}-result cap, "
                f"splitting into {len(bands) - 1} price bands")
            parts = [
                fetch(zipcode, listing_type, args, limiter,
                      price_min=bands[i], price_max=bands[i + 1], depth=1)
                for i in range(len(bands) - 1)
            ]
            df = pd.concat([p for p in parts if len(p)], ignore_index=True)
        elif depth < 8 and price_max is not None:
            mid = (price_min or 0) + ((price_max - (price_min or 0)) // 2)
            if mid > (price_min or 0):
                lo = fetch(zipcode, listing_type, args, limiter,
                           price_min=price_min, price_max=mid, depth=depth + 1)
                hi = fetch(zipcode, listing_type, args, limiter,
                           price_min=mid + 1, price_max=price_max, depth=depth + 1)
                df = pd.concat([lo, hi], ignore_index=True)
        else:
            log(f"  WARNING {zipcode}/{listing_type}: still capped at "
                f"{RESULT_CAP} rows in band [{price_min}, {price_max}]; "
                f"some listings may be missing")

    if "property_url" in df.columns:
        df = df.drop_duplicates(subset=["property_url"], ignore_index=True)
    return df


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except ImportError:
        df.astype(str).to_csv(path.with_suffix(".csv.gz"), index=False,
                              compression="gzip")


def worker(zipcode: str, state: str, listing_type: str, args,
           checkpoint: Checkpoint, limiter: RateLimiter, out_dir: Path):
    try:
        df = fetch(zipcode, listing_type, args, limiter)
        if len(df):
            save(df, out_dir / "raw" / listing_type / state / zipcode)
        checkpoint.mark(zipcode, listing_type, len(df))
        return len(df)
    except Exception:
        with (out_dir / "failed.txt").open("a") as f:
            f.write(f"{zipcode},{listing_type}\n")
        log(f"  FAILED {zipcode}/{listing_type}:\n{traceback.format_exc(limit=2)}")
        return 0


def cmd_scrape(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = [s.strip().upper() for s in args.states.split(",")] if args.states else None
    listing_types = [t.strip() for t in args.listing_types.split(",")]

    zips = load_zip_codes(out_dir / "cache", states)
    if args.max_zips:
        zips = zips[: args.max_zips]

    checkpoint = Checkpoint(out_dir / "checkpoint.txt")
    limiter = RateLimiter(args.delay)

    tasks = [(z, s, lt) for lt in listing_types for z, s in zips
             if not checkpoint.is_done(z, lt)]
    total = len(zips) * len(listing_types)
    log(f"{total} zip/type units total, {total - len(tasks)} already done, "
        f"{len(tasks)} to go ({args.workers} workers, {args.delay}s between requests)")
    if not tasks:
        log("Nothing to do — everything is checkpointed. Run 'combine' next.")
        return

    done = rows_total = 0
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(worker, z, s, lt, args, checkpoint, limiter, out_dir): (z, lt)
            for z, s, lt in tasks
        }
        try:
            for fut in as_completed(futures):
                done += 1
                rows_total += fut.result()
                if done % 25 == 0 or done == len(tasks):
                    rate = done / (time.monotonic() - start)
                    eta_h = (len(tasks) - done) / rate / 3600 if rate else 0
                    log(f"progress: {done}/{len(tasks)} units, "
                        f"{rows_total:,} rows this session, "
                        f"{rate * 3600:.0f} units/hr, ETA {eta_h:.1f}h")
        except KeyboardInterrupt:
            log("Interrupted — progress is checkpointed; rerun to resume.")
            pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)

    failed = out_dir / "failed.txt"
    log(f"Done. {rows_total:,} rows scraped this session.")
    if failed.exists():
        n = len(set(failed.read_text().splitlines()))
        log(f"{n} unit(s) failed (see {failed}); rerun 'scrape' to retry them.")
    log(f"Run 'combine' to merge into a single dataset.")


def cmd_combine(args):
    out_dir = Path(args.output)
    files = sorted((out_dir / "raw").rglob("*.parquet")) + \
            sorted((out_dir / "raw").rglob("*.csv.gz"))
    if not files:
        log(f"No scraped files found under {out_dir / 'raw'}")
        return
    log(f"Combining {len(files)} files ...")
    frames = []
    for i, f in enumerate(files, 1):
        frames.append(pd.read_parquet(f) if f.suffix == ".parquet"
                      else pd.read_csv(f))
        if i % 2000 == 0:
            log(f"  read {i}/{len(files)}")
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    if "property_url" in df.columns:
        df = df.drop_duplicates(subset=["property_url"], ignore_index=True)
    log(f"{before:,} rows -> {len(df):,} after dedup")

    out = out_dir / "us_properties.parquet"
    df.to_parquet(out, index=False)
    log(f"Wrote {out}")
    if args.csv:
        csv_out = out_dir / "us_properties.csv.gz"
        df.to_csv(csv_out, index=False, compression="gzip")
        log(f"Wrote {csv_out}")


def cmd_status(args):
    out_dir = Path(args.output)
    cp = out_dir / "checkpoint.txt"
    if not cp.exists():
        log("No checkpoint yet — nothing scraped.")
        return
    rows = zero = 0
    by_type: dict[str, int] = {}
    lines = cp.read_text().splitlines()
    for line in lines:
        parts = line.split(",")
        if len(parts) == 3:
            by_type[parts[1]] = by_type.get(parts[1], 0) + 1
            rows += int(parts[2])
            zero += parts[2] == "0"
    log(f"{len(lines)} zip/type units completed, {rows:,} rows total, "
        f"{zero} empty zips")
    for lt, n in sorted(by_type.items()):
        log(f"  {lt}: {n} zips done")
    failed = out_dir / "failed.txt"
    if failed.exists():
        log(f"{len(set(failed.read_text().splitlines()))} unit(s) in failed.txt")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("scrape", help="scrape zips (resumable)")
    ps.add_argument("--output", default="data_us", help="output directory")
    ps.add_argument("--listing-types", default="for_sale",
                    help="comma list: for_sale,for_rent,sold,pending")
    ps.add_argument("--states", default=None,
                    help="comma list of state codes to restrict to, e.g. AZ,CA")
    ps.add_argument("--workers", type=int, default=4)
    ps.add_argument("--delay", type=float, default=1.0,
                    help="min seconds between requests (global)")
    ps.add_argument("--retries", type=int, default=4)
    ps.add_argument("--past-days", type=int, default=None,
                    help="only listings listed/sold in the last N days")
    ps.add_argument("--extra-data", action="store_true",
                    help="fetch agent/broker/tax details (much slower)")
    ps.add_argument("--proxy", default=None, help="proxy URL")
    ps.add_argument("--max-zips", type=int, default=None,
                    help="limit zip count (for testing)")
    ps.set_defaults(func=cmd_scrape)

    pc = sub.add_parser("combine", help="merge scraped files into one dataset")
    pc.add_argument("--output", default="data_us")
    pc.add_argument("--csv", action="store_true", help="also write csv.gz")
    pc.set_defaults(func=cmd_combine)

    pt = sub.add_parser("status", help="show progress")
    pt.add_argument("--output", default="data_us")
    pt.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
