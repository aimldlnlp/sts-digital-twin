\
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import load_yaml, save_json, set_seed
from src.features.dataset import make_segments, select_split_groups, split_indices, StandardScaler1D
from src.models.nets import PhaseClassifier
from src.models.train_utils import SegmentDataset, get_device, macro_f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--modality", type=str, default="fusion", choices=["fusion","eeg","emg"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_yaml(Path(args.config)) if args.config else load_yaml(run_dir/"config.yaml")
    set_seed(int(cfg["seed"]))

    seg = make_segments(run_dir, cfg, modality=args.modality)
    X, y = seg["X"], seg["y_phase"]
    groups = select_split_groups(seg, cfg)

    tr_idx, va_idx, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"])+1, groups=groups)

    scaler = StandardScaler1D().fit(X[tr_idx])
    Xs = scaler.transform(X)

    device = get_device()
    model = PhaseClassifier(in_ch=Xs.shape[1], n_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()

    bs = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(SegmentDataset(Xs[tr_idx], y[tr_idx]), batch_size=bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(SegmentDataset(Xs[va_idx], y[va_idx]), batch_size=bs, shuffle=False, drop_last=False)

    best_val = -1.0
    best_path = None

    for epoch in range(int(cfg["train"]["epochs_phase"])):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"[phase-{args.modality}] epoch {epoch+1}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # val
        model.eval()
        preds = []
        ys = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                preds.append(logits.argmax(axis=1))
                ys.append(yb.numpy())
        pred = np.concatenate(preds)
        yv = np.concatenate(ys)
        f1 = macro_f1(pred, yv, n_classes=4)

        if f1 > best_val:
            best_val = f1
            out = run_dir/"artifacts"
            out.mkdir(parents=True, exist_ok=True)
            best_path = out/f"phase_model_{args.modality}.pt"
            torch.save(model.state_dict(), best_path)

    # test
    test_loader = DataLoader(SegmentDataset(Xs[te_idx], y[te_idx]), batch_size=bs, shuffle=False)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            preds.append(logits.argmax(axis=1))
            ys.append(yb.numpy())
    pred = np.concatenate(preds)
    yt = np.concatenate(ys)
    f1_test = macro_f1(pred, yt, n_classes=4)
    acc_test = float((pred == yt).mean())

    # save
    meta = {
        "modality": args.modality,
        "split_strategy": cfg["train"].get("split_strategy", "segment"),
        "val_macro_f1_best": float(best_val),
        "test_macro_f1": float(f1_test),
        "test_acc": float(acc_test),
        "n_test_segments": int(len(te_idx)),
        "model_path": str(best_path),
    }
    # scaler saved per modality
    with open(run_dir/"artifacts"/f"scaler_{args.modality}.json", "w") as f:
        json.dump(scaler.to_dict(), f)
    save_json(run_dir/"artifacts"/f"phase_metrics_{args.modality}.json", meta)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
