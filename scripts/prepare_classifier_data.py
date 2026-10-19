"""Download and prepare the tomato/potato leaf images for the crop-disease classifier.

Two public Hugging Face datasets (parquet files; we read only the row groups we need):
  geraldmc/plantvillage-full  (CC0)       clean lab photos, one leaf on a plain background
  geraldmc/plantdoc-full      (CC BY 4.0) messy photos taken in real fields

Output (outside OneDrive, because it is thousands of small files):
  <OUT>/<source>/<split>/<class>/<n>.jpg   images resized to at most 256 px
  <OUT>/manifest.csv                        one row per image

The script is resumable: finished row groups are remembered in <OUT>/done.json.

    python scripts/prepare_classifier_data.py --out C:/Users/jayag/datasets/plant
"""
import argparse
import io
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

# canonical class name per (crop, disease text in each dataset)
PV = {
    ("Tomato", "Bacterial spot"): "tomato__bacterial_spot",
    ("Tomato", "Early blight"): "tomato__early_blight",
    ("Tomato", "Late blight"): "tomato__late_blight",
    ("Tomato", "Leaf Mold"): "tomato__leaf_mold",
    ("Tomato", "Septoria leaf spot"): "tomato__septoria_leaf_spot",
    ("Tomato", "Tomato Yellow Leaf Curl Virus"): "tomato__yellow_leaf_curl",
    ("Tomato", "Tomato mosaic virus"): "tomato__mosaic_virus",
    ("Tomato", "healthy"): "tomato__healthy",
    ("Potato", "Early blight"): "potato__early_blight",
    ("Potato", "Late blight"): "potato__late_blight",
}
DOC = {
    ("Tomato", "bacterial spot"): "tomato__bacterial_spot",
    ("Tomato", "early blight"): "tomato__early_blight",
    ("Tomato", "late blight"): "tomato__late_blight",
    ("Tomato", "mold"): "tomato__leaf_mold",
    ("Tomato", "septoria leaf spot"): "tomato__septoria_leaf_spot",
    ("Tomato", "yellow virus"): "tomato__yellow_leaf_curl",
    ("Tomato", "mosaic virus"): "tomato__mosaic_virus",
    ("Tomato", "healthy"): "tomato__healthy",
    ("Potato", "early blight"): "potato__early_blight",
    ("Potato", "late blight"): "potato__late_blight",
}
# PlantVillage is huge and lab-clean: cap images per class so CPU training stays quick.
PV_CAP = {"train": 500, "test": 150}
MAX_SIDE = 256


def with_retries(fn, tries=8):
    """The development network sometimes fails certificate checks for a few seconds.
    We wait and retry; we never turn certificate verification off."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            print(f"  network retry {attempt + 1}/{tries}: {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("network kept failing")


def plan_row_groups(fs, repo, files, mapping):
    """Read only the small label columns; return {(file, rg): number of wanted rows}."""
    plan = {}
    for fi in range(files):
        pf = pq.ParquetFile(fs.open(f"datasets/{repo}/data/train-0000{fi}-of-0000{files}.parquet", "rb"))
        for g in range(pf.num_row_groups):
            df = pf.read_row_group(g, columns=["host", "disease"]).to_pandas()
            n = sum((h, d) in mapping for h, d in zip(df.host, df.disease))
            if n:
                plan[(fi, g)] = n
    return plan


def save_image(raw: bytes, path: Path):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=90)


def process(fs, repo, files, mapping, source, out: Path, done: dict, manifest: list, caps, min_rows):
    plan = with_retries(lambda: plan_row_groups(fs, repo, files, mapping))
    # skip row groups where only a handful of wanted rows would cost 100+ MB of download
    plan = {k: v for k, v in plan.items() if v >= min_rows}
    print(f"[{source}] row groups to read: {len(plan)}", flush=True)
    counts: dict = {}
    for row in manifest:
        if row["source"] == source:
            counts[(row["split"], row["label"])] = counts.get((row["split"], row["label"]), 0) + 1
    for (fi, g), want in sorted(plan.items()):
        key = f"{source}:{fi}:{g}"
        if key in done:
            continue
        for attempt in range(6):
            try:
                pf = pq.ParquetFile(fs.open(f"datasets/{repo}/data/train-0000{fi}-of-0000{files}.parquet", "rb"))
                table = pf.read_row_group(g)
                break
            except Exception as exc:  # flaky network: wait and retry (never disable SSL checks)
                print(f"  retry {attempt + 1}: {type(exc).__name__}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            print(f"  giving up on {key}", flush=True)
            continue
        df = table.to_pandas()
        for _, r in df.iterrows():
            label = mapping.get((r["host"], r["disease"]))
            if not label:
                continue
            split = r["split"]
            if caps and counts.get((split, label), 0) >= caps.get(split, 10**9):
                continue
            n = counts.get((split, label), 0)
            rel = Path(source) / split / label / f"{fi}_{g}_{n}.jpg"
            save_image(r["image"]["bytes"], out / rel)
            counts[(split, label)] = n + 1
            manifest.append({"path": str(rel).replace("\\", "/"), "source": source, "split": split, "label": label})
        done[key] = True
        (out / "done.json").write_text(json.dumps(done))
        pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)
        print(f"  done {key}  total images so far: {len(manifest)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="C:/Users/jayag/datasets/plant")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = json.loads((out / "done.json").read_text()) if (out / "done.json").exists() else {}
    manifest = pd.read_csv(out / "manifest.csv").to_dict("records") if (out / "manifest.csv").exists() else []
    fs = HfFileSystem()
    process(fs, "geraldmc/plantvillage-full", 2, PV, "village", out, done, manifest, PV_CAP, min_rows=500)
    process(fs, "geraldmc/plantdoc-full", 2, DOC, "doc", out, done, manifest, None, min_rows=1)
    print(pd.DataFrame(manifest).groupby(["source", "split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
