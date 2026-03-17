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
from src.models.nets import TorqueRegressor
from src.models.train_utils import SegmentDataset, get_device, rmse

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
    X, y = seg["X"], seg["y_tau"]
    groups = select_split_groups(seg, cfg)

    tr_idx, va_idx, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"])+2, groups=groups)

    scaler = StandardScaler1D().fit(X[tr_idx])
    Xs = scaler.transform(X)

    device = get_device()
    model = TorqueRegressor(in_ch=Xs.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    loss_fn = nn.MSELoss()

    bs = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(SegmentDataset(Xs[tr_idx], y[tr_idx]), batch_size=bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(SegmentDataset(Xs[va_idx], y[va_idx]), batch_size=bs, shuffle=False, drop_last=False)

    best_val = 1e9
    best_path = None

    for epoch in range(int(cfg["train"]["epochs_torque"])):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"[torque-{args.modality}] epoch {epoch+1}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device).float()
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # val
        model.eval()
        preds = []
        ys = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                ys.append(yb.numpy())
        pred = np.concatenate(preds)
        yv = np.concatenate(ys)
        val_rmse = rmse(pred, yv)

        if val_rmse < best_val:
            best_val = val_rmse
            out = run_dir/"artifacts"
            out.mkdir(parents=True, exist_ok=True)
            best_path = out/f"torque_model_{args.modality}.pt"
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
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            ys.append(yb.numpy())
    pred = np.concatenate(preds)
    yt = np.concatenate(ys)
    test_rmse = rmse(pred, yt)

    meta = {
        "modality": args.modality,
        "split_strategy": cfg["train"].get("split_strategy", "segment"),
        "val_rmse_best": float(best_val),
        "test_rmse": float(test_rmse),
        "n_test_segments": int(len(te_idx)),
        "model_path": str(best_path),
    }
    with open(run_dir/"artifacts"/f"scaler_{args.modality}_torque.json", "w") as f:
        json.dump(scaler.to_dict(), f)
    save_json(run_dir/"artifacts"/f"torque_metrics_{args.modality}.json", meta)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
