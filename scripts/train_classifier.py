"""Train and evaluate the leaf classifier. Two experiments tell the real story:

  A. train on clean lab photos (PlantVillage)            -> test on lab photos AND on field photos (PlantDoc)
  B. train on lab photos + a share of field photos       -> test on the same held-out field photos

Expected finding (from the project notes): ~95%+ on lab photos, far lower on messy field photos,
and mixing in field photos recovers part of the gap. We report what actually happens.

    python scripts/train_classifier.py --data C:/Users/jayag/datasets/plant --epochs 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pingfarm.vision.classifier import EVAL_TF, TRAIN_TF, build_model  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)


class Leaves(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, classes: list[str], train: bool):
        self.root, self.df, self.train = root, df.reset_index(drop=True), train
        self.idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("RGB")
        return (TRAIN_TF if self.train else EVAL_TF)(img), self.idx[row["label"]]


def make_loader(root, df, classes, train, batch=32):
    ds = Leaves(root, df, classes, train)
    if not train:
        return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    # balance classes by sampling rare ones more often
    freq = df["label"].value_counts()
    weights = df["label"].map(lambda c: 1.0 / freq[c]).values
    sampler = WeightedRandomSampler(weights, num_samples=len(df), replacement=True)
    return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=0)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x), dim=1))
    return torch.cat(out).numpy()


def train(model, root, train_df, classes, epochs, lr=1e-3):
    loader = make_loader(root, train_df, classes, True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for ep in range(epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(y)
            n += len(y)
        print(f"  epoch {ep + 1}/{epochs}  loss {total / n:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return model


def evaluate(model, root, df, classes):
    probs = predict_probs(model, make_loader(root, df, classes, False))
    y = df["label"].map({c: i for i, c in enumerate(classes)}).values
    pred = probs.argmax(1)
    return {
        "n": int(len(df)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
    }, probs, y


def selective(probs, y, taus=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Accuracy on the photos we ANSWER, and the share we answer, for each confidence cut-off."""
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in taus:
        keep = conf >= t
        rows.append({"tau": t, "coverage": round(float(keep.mean()), 3),
                     "accuracy_on_answered": round(float((pred[keep] == y[keep]).mean()), 3) if keep.any() else None})
    return rows


def choose_tau(probs_val, y_val, target=0.80, min_answered=20):
    """Smallest confidence cut-off whose accuracy on ANSWERED validation photos reaches `target`.

    Returns (tau, reached). If the target is never reached we return the cut-off with the best
    accuracy and reached=False, so nobody mistakes a fallback for a calibrated value."""
    best = (0.5, 0.0)
    for t in np.arange(0.3, 0.96, 0.05):
        keep = probs_val.max(1) >= t
        if keep.sum() < min_answered:
            break
        acc = float((probs_val.argmax(1)[keep] == y_val[keep]).mean())
        if acc >= target:
            return float(round(t, 2)), True
        if acc > best[1]:
            best = (float(round(t, 2)), acc)
    return best[0], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/Users/jayag/datasets/plant")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "models"))
    args = ap.parse_args()
    root = Path(args.data)
    m = pd.read_csv(root / "manifest.csv")
    classes = sorted(m["label"].unique())
    print("classes:", classes, flush=True)

    v_tr = m[(m.source == "village") & (m.split == "train")]
    v_te = m[(m.source == "village") & (m.split == "test")]
    d_all_tr = m[(m.source == "doc") & (m.split == "train")]
    d_te = m[(m.source == "doc") & (m.split == "test")]
    # keep 15% of field-train aside to choose the "I'm not sure" cut-off (never tune on the test set)
    d_tr, d_val = train_test_split(d_all_tr, test_size=0.15, stratify=d_all_tr["label"], random_state=0)
    print(f"village train {len(v_tr)} test {len(v_te)} | field train {len(d_tr)} val {len(d_val)} test {len(d_te)}", flush=True)

    results = {"classes": classes}

    print("\n== Experiment A: lab photos only ==", flush=True)
    model_a = train(build_model(len(classes)), root, v_tr, classes, args.epochs)
    results["A_lab_only"] = {"on_lab_test": evaluate(model_a, root, v_te, classes)[0],
                             "on_field_test": evaluate(model_a, root, d_te, classes)[0]}
    print(json.dumps(results["A_lab_only"], indent=1), flush=True)

    print("\n== Experiment B: lab + field photos ==", flush=True)
    model_b = train(build_model(len(classes)), root, pd.concat([v_tr, d_tr]), classes, args.epochs)
    res_lab, _, _ = evaluate(model_b, root, v_te, classes)
    res_field, probs_te, y_te = evaluate(model_b, root, d_te, classes)
    results["B_lab_plus_field"] = {"on_lab_test": res_lab, "on_field_test": res_field}
    print(json.dumps(results["B_lab_plus_field"], indent=1), flush=True)

    # choose tau on the validation split (never on the test set)
    _, probs_val, y_val = evaluate(model_b, root, d_val, classes)
    tau, reached = choose_tau(probs_val, y_val)
    results["tau_target_reached"] = reached
    results["tau"] = tau
    results["selective_on_field_test"] = selective(probs_te, y_te)
    cm = confusion_matrix(y_te, probs_te.argmax(1), labels=range(len(classes)))
    results["confusion_matrix_field_test"] = cm.tolist()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model_b.state_dict(), "classes": classes, "tau": tau}, out / "leaf_classifier.pt")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "classifier_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved model and results; tau =", tau, flush=True)


if __name__ == "__main__":
    main()
