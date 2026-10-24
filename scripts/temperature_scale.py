"""Temperature scaling: make the classifier's confidence match how often it is actually right.

A network is often over-confident ("97% sure" but right only 70% of the time). Temperature scaling divides the
network's raw scores (logits) by one number T before the softmax. T is fitted on the validation photos only
(minimising negative log-likelihood); accuracy never changes, only the confidence numbers do.

We then report calibration (ECE) before and after on the held-out FIELD test photos, and re-choose the "not sure"
cut-off tau on the validation split using the scaled confidence.

    python scripts/temperature_scale.py --data C:/Users/jayag/datasets/plant
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_classifier import choose_tau, make_loader  # noqa: E402

from pingfarm.vision.classifier import build_model  # noqa: E402


@torch.no_grad()
def logits_for(model, root, df, classes):
    model.eval()
    out = []
    for x, _ in make_loader(root, df, classes, False):
        out.append(model(x))
    return torch.cat(out)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: average gap between confidence and accuracy across confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / t.clamp(min=0.05), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.clamp(min=0.05).item())


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
    idx = {c: i for i, c in enumerate(classes)}

    d_all = m[(m.source == "doc") & (m.split == "train")]
    _, d_val = train_test_split(d_all, test_size=0.15, stratify=d_all["label"], random_state=0)  # same split as training
    d_te = m[(m.source == "doc") & (m.split == "test")]
    lv, lt = logits_for(model, root, d_val, classes), logits_for(model, root, d_te, classes)
    yv = torch.tensor(d_val["label"].map(idx).values)
    yt = d_te["label"].map(idx).values

    T = fit_temperature(lv, yv)
    p_before, p_after = torch.softmax(lt, 1).numpy(), torch.softmax(lt / T, 1).numpy()
    pv_after = torch.softmax(lv / T, 1).numpy()
    tau, reached = choose_tau(pv_after, yv.numpy(), target=args.target)

    def selective(p, tau_):
        keep = p.max(1) >= tau_
        return {"coverage": round(float(keep.mean()), 3),
                "accuracy_on_answered": round(float((p.argmax(1)[keep] == yt[keep]).mean()), 3) if keep.any() else None}

    res_path = ROOT / "results" / "classifier_results.json"
    res = json.loads(res_path.read_text())
    res["temperature_scaling"] = {
        "temperature": round(T, 3),
        "ece_before_on_field_test": round(ece(p_before, yt), 3), "ece_after_on_field_test": round(ece(p_after, yt), 3),
        "mean_confidence_before": round(float(p_before.max(1).mean()), 3),
        "mean_confidence_after": round(float(p_after.max(1).mean()), 3),
        "field_test_accuracy": round(float((p_before.argmax(1) == yt).mean()), 3),
        "old_tau": res["tau"], "new_tau": tau, "new_tau_target_reached": reached,
        "field_test_at_old_tau_unscaled": selective(p_before, res["tau"]),
        "field_test_at_new_tau_scaled": selective(p_after, tau),
    }
    res_path.write_text(json.dumps(res, indent=2))
    ckpt["temperature"], ckpt["tau"] = T, tau
    torch.save(ckpt, ckpt_path)
    print(json.dumps(res["temperature_scaling"], indent=2))


if __name__ == "__main__":
    main()
