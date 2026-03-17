from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common import ensure_dir, load_yaml, save_json, set_seed
from src.features.dataset import load_index, make_segments, select_split_groups, split_indices
from src.models.nets import PhaseClassifier, TorqueRegressor
from src.models.train_utils import SegmentDataset, macro_f1, rmse


def metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def override_split_strategy(cfg: Dict[str, Any], split_strategy: str | None) -> Dict[str, Any]:
    cfg_use = copy.deepcopy(cfg)
    if split_strategy is not None:
        cfg_use.setdefault("train", {})["split_strategy"] = split_strategy
    cfg_use.setdefault("train", {}).setdefault("split_strategy", "segment")
    return cfg_use


def dataset_summary(run_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    index = load_index(run_dir)
    seg = make_segments(run_dir, cfg, modality="fusion")
    groups = select_split_groups(seg, cfg)
    tr_idx, va_idx, te_idx = split_indices(len(seg["y_phase"]), cfg, seed=int(cfg["seed"]) + 1, groups=groups)
    strategy = cfg["train"]["split_strategy"]

    out = {
        "split_strategy": strategy,
        "n_subjects": len({int(item["subject"]) for item in index}),
        "n_trials": len(index),
        "n_segments": int(len(seg["y_phase"])),
        "segment_len_samples": int(cfg["train"]["segment_len"]),
        "segment_stride_samples": int(cfg["train"]["segment_stride"]),
        "segment_len_s": float(cfg["train"]["segment_len"] / cfg["sample_rate_hz"]),
        "segment_stride_s": float(cfg["train"]["segment_stride"] / cfg["sample_rate_hz"]),
        "split_counts": {
            "train": int(len(tr_idx)),
            "val": int(len(va_idx)),
            "test": int(len(te_idx)),
        },
    }
    if groups is not None:
        train_groups = set(np.unique(groups[tr_idx]).tolist())
        val_groups = set(np.unique(groups[va_idx]).tolist())
        test_groups = set(np.unique(groups[te_idx]).tolist())
        out["group_counts"] = {
            "train": len(train_groups),
            "val": len(val_groups),
            "test": len(test_groups),
        }
        out["group_overlap"] = {
            "train_val": len(train_groups & val_groups),
            "train_test": len(train_groups & test_groups),
            "val_test": len(val_groups & test_groups),
        }
    return out


def evaluate_phase(run_dir: Path, cfg: Dict[str, Any], modality: str) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    seg = make_segments(run_dir, cfg, modality=modality)
    X, y = seg["X"], seg["y_phase"]
    groups = select_split_groups(seg, cfg)
    tr_idx, va_idx, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 1, groups=groups)

    mean = X[tr_idx].mean(axis=(0, 2), keepdims=True)
    std = X[tr_idx].std(axis=(0, 2), keepdims=True) + 1e-6
    Xs = (X - mean) / std

    model = PhaseClassifier(in_ch=Xs.shape[1], n_classes=4)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()
    bs = int(cfg["train"]["batch_size"])

    train_loader = DataLoader(SegmentDataset(Xs[tr_idx], y[tr_idx]), batch_size=bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(SegmentDataset(Xs[va_idx], y[va_idx]), batch_size=bs, shuffle=False, drop_last=False)

    best_val = -1.0
    best_state = None
    for _ in range(int(cfg["train"]["epochs_phase"])):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        preds = []
        ys = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb).argmax(dim=1).numpy())
                ys.append(yb.numpy())
        pred = np.concatenate(preds)
        yt = np.concatenate(ys)
        val_f1 = macro_f1(pred, yt, n_classes=4)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loader = DataLoader(SegmentDataset(Xs[te_idx], y[te_idx]), batch_size=bs, shuffle=False, drop_last=False)
    preds = []
    ys = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb).argmax(dim=1).numpy())
            ys.append(yb.numpy())
    pred = np.concatenate(preds)
    yt = np.concatenate(ys)

    majority = int(np.bincount(y[tr_idx], minlength=4).argmax())
    majority_pred = np.full_like(yt, majority)
    return {
        "modality": modality,
        "val_macro_f1_best": float(best_val),
        "test_macro_f1": float(macro_f1(pred, yt, n_classes=4)),
        "test_acc": float((pred == yt).mean()),
        "chance_macro_f1": 0.25,
        "majority_macro_f1": float(macro_f1(majority_pred, yt, n_classes=4)),
        "majority_acc": float((majority_pred == yt).mean()),
        "n_test_segments": int(len(te_idx)),
        "source": "recomputed",
    }


def evaluate_torque(run_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    seg = make_segments(run_dir, cfg, modality="fusion")
    X, y = seg["X"], seg["y_tau"]
    groups = select_split_groups(seg, cfg)
    tr_idx, va_idx, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 2, groups=groups)

    mean = X[tr_idx].mean(axis=(0, 2), keepdims=True)
    std = X[tr_idx].std(axis=(0, 2), keepdims=True) + 1e-6
    Xs = (X - mean) / std

    model = TorqueRegressor(in_ch=Xs.shape[1])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    bs = int(cfg["train"]["batch_size"])

    train_loader = DataLoader(SegmentDataset(Xs[tr_idx], y[tr_idx]), batch_size=bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(SegmentDataset(Xs[va_idx], y[va_idx]), batch_size=bs, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None
    for _ in range(int(cfg["train"]["epochs_torque"])):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb.float())
            loss.backward()
            opt.step()

        model.eval()
        preds = []
        ys = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb).numpy())
                ys.append(yb.numpy())
        pred = np.concatenate(preds)
        yt = np.concatenate(ys)
        val_rmse = rmse(pred, yt)
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loader = DataLoader(SegmentDataset(Xs[te_idx], y[te_idx]), batch_size=bs, shuffle=False, drop_last=False)
    preds = []
    ys = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb).numpy())
            ys.append(yb.numpy())
    pred = np.concatenate(preds)
    yt = np.concatenate(ys)
    baseline_mean = np.full_like(yt, float(y[tr_idx].mean()))
    test_rmse = rmse(pred, yt)
    test_std = float(yt.std())
    approx_r2 = None if test_std <= 1e-9 else float(1.0 - (test_rmse ** 2) / (test_std ** 2))
    corr = float(np.corrcoef(yt, pred)[0, 1]) if len(yt) > 1 else None
    return {
        "modality": "fusion",
        "val_rmse_best": float(best_val),
        "test_rmse": float(test_rmse),
        "baseline_mean_rmse": float(rmse(baseline_mean, yt)),
        "test_std": test_std,
        "approx_r2": approx_r2,
        "corr": corr,
        "n_test_segments": int(len(te_idx)),
        "source": "recomputed",
    }


def load_artifact_metrics(run_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    strategy = cfg["train"]["split_strategy"]
    phase = {}
    for modality in ["eeg", "emg", "fusion"]:
        path = run_dir / "artifacts" / f"phase_metrics_{modality}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact metrics: {path}")
        phase[modality] = json.loads(path.read_text()) | {"source": "artifacts"}

    torque_path = run_dir / "artifacts" / "torque_metrics_fusion.json"
    if not torque_path.exists():
        raise FileNotFoundError(f"Missing artifact metrics: {torque_path}")
    torque = json.loads(torque_path.read_text()) | {"source": "artifacts"}

    seg_phase = make_segments(run_dir, cfg, modality="fusion")
    phase_groups = select_split_groups(seg_phase, cfg)
    tr_idx, _, te_idx = split_indices(len(seg_phase["y_phase"]), cfg, seed=int(cfg["seed"]) + 1, groups=phase_groups)
    yt_phase = seg_phase["y_phase"][te_idx]
    majority = int(np.bincount(seg_phase["y_phase"][tr_idx], minlength=4).argmax())
    majority_pred = np.full_like(yt_phase, majority)
    for modality in phase.values():
        modality["chance_macro_f1"] = 0.25
        modality["majority_macro_f1"] = float(macro_f1(majority_pred, yt_phase, n_classes=4))
        modality["majority_acc"] = float((majority_pred == yt_phase).mean())
        modality["n_test_segments"] = int(len(te_idx))
        modality["split_strategy"] = strategy

    seg_tau = make_segments(run_dir, cfg, modality="fusion")
    tau_groups = select_split_groups(seg_tau, cfg)
    tr_idx, _, te_idx = split_indices(len(seg_tau["y_tau"]), cfg, seed=int(cfg["seed"]) + 2, groups=tau_groups)
    yt_tau = seg_tau["y_tau"][te_idx]
    baseline_mean = np.full_like(yt_tau, float(seg_tau["y_tau"][tr_idx].mean()))
    torque["baseline_mean_rmse"] = float(rmse(baseline_mean, yt_tau))
    torque["test_std"] = float(yt_tau.std())
    torque["approx_r2"] = None if torque["test_std"] <= 1e-9 else float(1.0 - (torque["test_rmse"] ** 2) / (torque["test_std"] ** 2))
    torque["split_strategy"] = strategy
    torque["n_test_segments"] = int(len(te_idx))
    return {"phase": phase, "torque": torque}


def synergy_summary(run_dir: Path) -> Dict[str, Any] | None:
    path = run_dir / "artifacts" / "nmf_synergy_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def render_report_markdown(report: Dict[str, Any]) -> str:
    ds = report["dataset"]
    phase = report["phase"]
    torque = report["torque"]
    syn = report.get("synergy")

    lines = [
        "# STS Digital Twin Validation Report",
        "",
        "This run is positioned as an in-silico STS development bench for multimodal phase decoding and assistive torque prediction.",
        "",
        "## Snapshot",
        "",
        f"- Split strategy: `{ds['split_strategy']}`",
        f"- Subjects: `{ds['n_subjects']}`",
        f"- Trials: `{ds['n_trials']}`",
        f"- Segments: `{ds['n_segments']}`",
        f"- Segment window: `{ds['segment_len_s']:.2f}s` with `{ds['segment_stride_s']:.2f}s` stride",
        f"- Split counts: train `{ds['split_counts']['train']}`, val `{ds['split_counts']['val']}`, test `{ds['split_counts']['test']}`",
    ]
    if "group_counts" in ds:
        lines.append(
            f"- Group counts: train `{ds['group_counts']['train']}`, val `{ds['group_counts']['val']}`, test `{ds['group_counts']['test']}`"
        )
        lines.append(
            f"- Group overlap: train/val `{ds['group_overlap']['train_val']}`, train/test `{ds['group_overlap']['train_test']}`, val/test `{ds['group_overlap']['val_test']}`"
        )
    if syn is not None:
        lines.append(f"- Synergy VAF: `{syn['vaf']:.3f}` with `K={syn['K']}` synergies over `{syn['M']}` muscles")

    lines.extend(
        [
            "",
            "## Phase Decoding",
            "",
            "| Modality | Test Macro-F1 | Test Acc | Majority F1 | Majority Acc | Source |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for modality in ["eeg", "emg", "fusion"]:
        m = phase[modality]
        lines.append(
            f"| {modality.upper()} | {metric(m.get('test_macro_f1'))} | {metric(m.get('test_acc'))} | "
            f"{metric(m.get('majority_macro_f1'))} | {metric(m.get('majority_acc'))} | {m.get('source', '-')} |"
        )

    lines.extend(
        [
            "",
            "## Torque Prediction",
            "",
            "| Model | Test RMSE | Mean Baseline RMSE | Approx. R2 | Corr | Source |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            f"| Fusion | {metric(torque.get('test_rmse'))} | {metric(torque.get('baseline_mean_rmse'))} | "
            f"{metric(torque.get('approx_r2'))} | {metric(torque.get('corr'))} | {torque.get('source', '-')} |",
            "",
            "## Interpretation",
            "",
            "- This benchmark reflects controller-development conditions, not a clinical deployment claim.",
            "- Subject-wise or trial-wise splits are the preferred protocol for any headline number.",
            "- The data source remains in-silico, but the evaluation and reporting are structured to match a serious offline R&D workflow.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--split_strategy", type=str, default=None, choices=["segment", "trial", "subject"])
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)

    run_dir = Path(args.run_dir)
    cfg = override_split_strategy(load_yaml(run_dir / "config.yaml"), args.split_strategy)
    strategy = cfg["train"]["split_strategy"]

    if args.split_strategy is not None and not args.recompute:
        run_cfg = load_yaml(run_dir / "config.yaml")
        run_strategy = run_cfg.get("train", {}).get("split_strategy", "segment")
        if run_strategy != strategy:
            raise ValueError(
                "Refusing to reuse artifact metrics with a different split strategy. "
                "Run with --recompute to generate a benchmark under the requested protocol."
            )

    dataset = dataset_summary(run_dir, cfg)
    synergy = synergy_summary(run_dir)
    if args.recompute:
        phase = {
            modality: evaluate_phase(run_dir, cfg, modality)
            for modality in ["eeg", "emg", "fusion"]
        }
        torque = evaluate_torque(run_dir, cfg)
    else:
        loaded = load_artifact_metrics(run_dir, cfg)
        phase = loaded["phase"]
        torque = loaded["torque"]

    report = {
        "run_dir": str(run_dir),
        "dataset": dataset,
        "phase": phase,
        "torque": torque,
        "synergy": synergy,
    }

    reports_dir = ensure_dir(run_dir / "reports")
    json_path = reports_dir / f"validation_{strategy}.json"
    md_path = reports_dir / f"validation_{strategy}.md"
    save_json(json_path, report)
    md_path.write_text(render_report_markdown(report))
    print(f"Saved JSON report to: {json_path}")
    print(f"Saved Markdown report to: {md_path}")


if __name__ == "__main__":
    main()
