"""Re-choose the "not sure" cut-off for the saved classifier WITHOUT retraining.

The first training run asked for 90% accuracy on answered validation photos, which was never
reachable (only 133 validation photos), so it silently fell back to 0.5. This script reuses the
same validation split and asks for a reachable 80%, and records whether the target was reached.

    python scripts/recalibrate_tau.py --data C:/Users/jayag/datasets/plant
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_classifier import choose_tau, evaluate  # noqa: E402

from pingfarm.vision.classifier import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/Users/jayag/datasets/plant")
    ap.add_argument("--target", type=float, default=0.80)
    args = ap.parse_args()
    root = Path(args.data)
    m = pd.read_csv(root / "manifest.csv")
    ckpt_path = ROOT / "models" / "leaf_classifier.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    classes = ckpt["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])

    d_all_tr = m[(m.source == "doc") & (m.split == "train")]
    _, d_val = train_test_split(d_all_tr, test_size=0.15, stratify=d_all_tr["label"], random_state=0)  # same split as training
    _, probs_val, y_val = evaluate(model, root, d_val, classes)
    tau, reached = choose_tau(probs_val, y_val, target=args.target)
    print(f"validation photos: {len(d_val)}  target accuracy {args.target}  ->  tau={tau}  reached={reached}")

    ckpt["tau"] = tau
    torch.save(ckpt, ckpt_path)
    res_path = ROOT / "results" / "classifier_results.json"
    res = json.loads(res_path.read_text())
    res["tau"], res["tau_target_reached"], res["tau_target"] = tau, reached, args.target
    res_path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
