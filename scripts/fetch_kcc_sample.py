"""Sample REAL farmer questions from the Kisan Call Centre (KCC) open data.

Source: Kisan Call Centre query records, Government of India open data (data.gov.in), as mirrored in the
Hugging Face dataset AnmolNimmala0/kcc-farmer-query-transcripts. Each record has the crop, state, query
type, the query text (English, typed by the call-centre operator) and the expert's answer.

IMPORTANT about what this is: the English QueryText is the OPERATOR'S short summary of what the farmer
said on the phone, not the farmer's own words. It is real and realistically terse ("tomato leaf curl
control"), but it is not verbatim speech.

We keep the sample out of git (data/kcc/). Run:  python scripts/fetch_kcc_sample.py --files 16
Writes data/kcc/tomato_potato.csv (target crops) and data/kcc/other_crops.csv (negatives for the topic gate).
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "kcc"
REPO = "AnmolNimmala0/kcc-farmer-query-transcripts"
COLS = ["Crop", "Category", "QueryType", "QueryText", "KccAns", "StateName", "_Year"]


def retry(fn, tries=10):
    """The network sometimes fails certificate checks for a few seconds: wait and retry, never disable checks."""
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:
            print(f"  retry {i + 1}: {type(exc).__name__}", flush=True)
            time.sleep(4 * (i + 1))
    raise RuntimeError("network kept failing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=16, help="how many of the ~1000 parquet files to sample")
    ap.add_argument("--other-per-file", type=int, default=150)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    files = sorted(retry(lambda: fs.ls(f"datasets/{REPO}/data", detail=False)))
    step = max(1, len(files) // args.files)
    chosen = files[::step][: args.files]
    print(f"{len(files)} files; sampling {len(chosen)}", flush=True)

    target, other = [], []
    for f in chosen:
        def read():
            return pq.ParquetFile(fs.open(f, "rb")).read(columns=COLS).to_pandas()
        df = retry(read)
        crop = df["Crop"].fillna("").str.lower()
        is_target = crop.str.contains(r"\btomato\b|\bpotato\b") & ~crop.str.contains("sweet potato")
        target.append(df[is_target])
        other.append(df[~is_target].sample(min(args.other_per_file, int((~is_target).sum())), random_state=0))
        print(f"  {f.split('/')[-1]}: {len(df)} rows, {int(is_target.sum())} tomato/potato", flush=True)
        pd.concat(target).to_csv(OUT / "tomato_potato.csv", index=False)
        pd.concat(other).to_csv(OUT / "other_crops.csv", index=False)
    t = pd.concat(target)
    t["QueryType"] = t["QueryType"].str.strip()
    print("\ntomato/potato rows:", len(t))
    print(t["QueryType"].value_counts().head(10).to_string())
    print(t["StateName"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
