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
from src.features.dataset import StandardScaler1D, StandardScalerTarget, get_channel_counts, load_index, make_segments, select_split_groups, split_indices
from src.models.nets import build_phase_model, build_torque_model, load_model_checkpoint
from src.models.train_utils import (
    SegmentDataset,
    apply_eval_stress_np,
    balanced_accuracy,
    corrcoef,
    mae,
    macro_f1,
    phase_selection_score,
    r2_score,
    rmse,
    torque_error_by_range,
    torque_selection_score,
)


REFERENCE_RUNS = {
    "v3": "sts_20260420_150333",
    "v4": "sts_20260420_151550",
    "v5": "sts_20260420_215231",
}


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


def _load_phase_artifact_model(run_dir: Path, cfg: Dict[str, Any], modality: str, device: torch.device) -> tuple[torch.nn.Module, StandardScaler1D]:
    model = load_model_checkpoint(run_dir / "artifacts" / f"phase_model_{modality}.pt", cfg, task="phase", modality=modality, device=device)
    scaler = StandardScaler1D.from_dict(json.loads((run_dir / "artifacts" / f"scaler_{modality}.json").read_text()))
    model.eval()
    return model, scaler


def _load_torque_artifact_model(
    run_dir: Path,
    cfg: Dict[str, Any],
    modality: str,
    device: torch.device,
) -> tuple[torch.nn.Module, StandardScaler1D, StandardScalerTarget | None]:
    model = load_model_checkpoint(run_dir / "artifacts" / f"torque_model_{modality}.pt", cfg, task="torque", modality=modality, device=device)
    scaler = StandardScaler1D.from_dict(json.loads((run_dir / "artifacts" / f"scaler_{modality}_torque.json").read_text()))
    target_scaler_path = run_dir / "artifacts" / f"target_scaler_{modality}_torque.json"
    target_scaler = StandardScalerTarget.from_dict(json.loads(target_scaler_path.read_text())) if target_scaler_path.exists() else None
    model.eval()
    return model, scaler, target_scaler


def _predict_phase_labels(
    run_dir: Path,
    cfg: Dict[str, Any],
    modality: str,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model, scaler = _load_phase_artifact_model(run_dir, cfg, modality, device)
    xs = scaler.transform(x)
    loader = DataLoader(SegmentDataset(xs, np.zeros(len(xs), dtype=np.int64)), batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def _predict_torque_values(
    run_dir: Path,
    cfg: Dict[str, Any],
    modality: str,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model, scaler, target_scaler = _load_torque_artifact_model(run_dir, cfg, modality, device)
    xs = scaler.transform(x)
    loader = DataLoader(SegmentDataset(xs, np.zeros(len(xs), dtype=np.float32)), batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    pred = np.concatenate(preds).astype(np.float32)
    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred)
    return pred


def _phase_stress_summary(run_dir: Path, cfg: Dict[str, Any], modality: str) -> Dict[str, Any]:
    device = torch.device("cpu")
    seg = make_segments(run_dir, cfg, modality=modality)
    groups = select_split_groups(seg, cfg)
    _, _, te_idx = split_indices(len(seg["y_phase"]), cfg, seed=int(cfg["seed"]) + 1, groups=groups)
    x_test = seg["X"][te_idx]
    y_test = seg["y_phase"][te_idx]

    stress_cases = ["clean", "emg_noise", "temporal_shift"]
    if modality == "fusion":
        stress_cases += ["drop_eeg", "drop_emg"]

    summary = {}
    for offset, stress_case in enumerate(stress_cases):
        x_use = apply_eval_stress_np(x_test, cfg, modality, stress_case, seed=int(cfg["seed"]) + 400 + offset)
        pred = _predict_phase_labels(run_dir, cfg, modality, x_use, device)
        summary[stress_case] = {
            "macro_f1": float(macro_f1(pred, y_test, n_classes=4)),
            "acc": float((pred == y_test).mean()),
            "balanced_acc": float(balanced_accuracy(pred, y_test, n_classes=4)),
        }
    return summary


def _torque_stress_summary(run_dir: Path, cfg: Dict[str, Any], modality: str) -> Dict[str, Any]:
    device = torch.device("cpu")
    seg = make_segments(run_dir, cfg, modality=modality)
    groups = select_split_groups(seg, cfg)
    _, _, te_idx = split_indices(len(seg["y_tau"]), cfg, seed=int(cfg["seed"]) + 2, groups=groups)
    x_test = seg["X"][te_idx]
    y_test = seg["y_tau"][te_idx]

    stress_cases = ["clean", "emg_noise", "temporal_shift"]
    if modality == "fusion":
        stress_cases += ["drop_eeg", "drop_emg"]

    summary = {}
    clean_pred = None
    for offset, stress_case in enumerate(stress_cases):
        x_use = apply_eval_stress_np(x_test, cfg, modality, stress_case, seed=int(cfg["seed"]) + 700 + offset)
        pred = _predict_torque_values(run_dir, cfg, modality, x_use, device)
        if stress_case == "clean":
            clean_pred = pred
        summary[stress_case] = {
            "rmse": float(rmse(pred, y_test)),
            "mae": float(mae(pred, y_test)),
            "corr": corrcoef(pred, y_test),
            "r2": r2_score(pred, y_test),
        }
    summary["error_by_target_range"] = torque_error_by_range(y_test, clean_pred if clean_pred is not None else y_test)
    return summary


def evaluate_phase(run_dir: Path, cfg: Dict[str, Any], modality: str) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    seg = make_segments(run_dir, cfg, modality=modality)
    X, y = seg["X"], seg["y_phase"]
    groups = select_split_groups(seg, cfg)
    tr_idx, va_idx, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 1, groups=groups)

    mean = X[tr_idx].mean(axis=(0, 2), keepdims=True)
    std = X[tr_idx].std(axis=(0, 2), keepdims=True) + 1e-6
    Xs = (X - mean) / std

    model = build_phase_model(cfg, modality)
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
        "test_balanced_acc": float(balanced_accuracy(pred, yt, n_classes=4)),
        "chance_macro_f1": 0.25,
        "majority_macro_f1": float(macro_f1(majority_pred, yt, n_classes=4)),
        "majority_acc": float((majority_pred == yt).mean()),
        "n_test_segments": int(len(te_idx)),
        "source": "recomputed",
    }


def evaluate_torque(run_dir: Path, cfg: Dict[str, Any], modality: str = "fusion") -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    seg = make_segments(run_dir, cfg, modality=modality)
    X, y = seg["X"], seg["y_tau"]
    groups = select_split_groups(seg, cfg)
    tr_idx, va_idx, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 2, groups=groups)

    mean = X[tr_idx].mean(axis=(0, 2), keepdims=True)
    std = X[tr_idx].std(axis=(0, 2), keepdims=True) + 1e-6
    Xs = (X - mean) / std

    model = build_torque_model(cfg, modality)
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
        "modality": modality,
        "val_rmse_best": float(best_val),
        "test_rmse": float(test_rmse),
        "test_mae": float(mae(pred, yt)),
        "baseline_mean_rmse": float(rmse(baseline_mean, yt)),
        "test_std": test_std,
        "approx_r2": approx_r2 if approx_r2 is not None else r2_score(pred, yt),
        "corr": corr if corr is not None else corrcoef(pred, yt),
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

    torque = {}
    for modality in ["fusion", "emg"]:
        torque_path = run_dir / "artifacts" / f"torque_metrics_{modality}.json"
        if torque_path.exists():
            torque[modality] = json.loads(torque_path.read_text()) | {"source": "artifacts"}

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
    for modality in torque.values():
        modality["baseline_mean_rmse"] = float(rmse(baseline_mean, yt_tau))
        modality["test_std"] = float(yt_tau.std())
        modality["approx_r2"] = None if modality["test_std"] <= 1e-9 else float(1.0 - (modality["test_rmse"] ** 2) / (modality["test_std"] ** 2))
        modality["split_strategy"] = strategy
        modality["n_test_segments"] = int(len(te_idx))

    if "emg" in phase and "fusion" in phase:
        phase["fusion"]["delta_over_emg_macro_f1"] = float(phase["fusion"]["test_macro_f1"] - phase["emg"]["test_macro_f1"])
        phase["fusion"]["delta_over_emg_acc"] = float(phase["fusion"]["test_acc"] - phase["emg"]["test_acc"])
    if "emg" in torque and "fusion" in torque:
        torque["fusion"]["delta_over_emg_torque_rmse"] = float(torque["fusion"]["test_rmse"] - torque["emg"]["test_rmse"])
    stress = {
        "phase": {
            "eeg": _phase_stress_summary(run_dir, cfg, "eeg"),
            "emg": _phase_stress_summary(run_dir, cfg, "emg"),
            "fusion": _phase_stress_summary(run_dir, cfg, "fusion"),
        },
        "torque": {
            "emg": _torque_stress_summary(run_dir, cfg, "emg"),
            "fusion": _torque_stress_summary(run_dir, cfg, "fusion"),
        },
    }
    ensure_selection_metrics(phase, torque, stress, cfg)
    return {"phase": phase, "torque": torque, "stress": stress}


def synergy_summary(run_dir: Path) -> Dict[str, Any] | None:
    path = run_dir / "artifacts" / "nmf_synergy_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def ensure_selection_metrics(phase: Dict[str, Any], torque: Dict[str, Any], stress: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    fusion_phase = phase.get("fusion", {})
    fusion_phase.setdefault(
        "phase_selection_score",
        phase_selection_score(
            float(fusion_phase.get("test_macro_f1", 0.0)),
            float(stress.get("phase", {}).get("fusion", {}).get("emg_noise", {}).get("macro_f1", 0.0)),
            float(stress.get("phase", {}).get("fusion", {}).get("drop_eeg", {}).get("macro_f1", 0.0)),
            cfg,
        ),
    )
    fusion_torque = torque.get("fusion", {})
    high_target_rmse = fusion_torque.get("high_target_rmse")
    if high_target_rmse is None:
        high_target_rmse = stress.get("torque", {}).get("fusion", {}).get("error_by_target_range", {}).get("high", {}).get("rmse")
        fusion_torque["high_target_rmse"] = high_target_rmse
    fusion_torque.setdefault(
        "torque_selection_score",
        torque_selection_score(
            float(fusion_torque.get("test_rmse", 0.0)),
            float(stress.get("torque", {}).get("fusion", {}).get("emg_noise", {}).get("rmse", fusion_torque.get("test_rmse", 0.0))),
            float(stress.get("torque", {}).get("fusion", {}).get("drop_eeg", {}).get("rmse", fusion_torque.get("test_rmse", 0.0))),
            float(high_target_rmse if high_target_rmse is not None else fusion_torque.get("test_rmse", 0.0)),
            cfg,
        ),
    )
    fusion_phase["generalization_gap_clean"] = (
        None
        if fusion_phase.get("val_macro_f1_best") is None
        else float(fusion_phase.get("val_macro_f1_best", 0.0) - fusion_phase.get("test_macro_f1", 0.0))
    )
    fusion_phase["generalization_gap_emg_noise"] = (
        None
        if fusion_phase.get("val_emg_noise_macro_f1_best") is None
        else float(fusion_phase.get("val_emg_noise_macro_f1_best", 0.0) - stress.get("phase", {}).get("fusion", {}).get("emg_noise", {}).get("macro_f1", 0.0))
    )
    fusion_torque["generalization_gap_clean"] = (
        None
        if fusion_torque.get("val_rmse_best") is None
        else float(fusion_torque.get("test_rmse", 0.0) - fusion_torque.get("val_rmse_best", 0.0))
    )
    fusion_torque["generalization_gap_emg_noise"] = (
        None
        if fusion_torque.get("val_emg_noise_rmse_best") is None
        else float(stress.get("torque", {}).get("fusion", {}).get("emg_noise", {}).get("rmse", 0.0) - fusion_torque.get("val_emg_noise_rmse_best", 0.0))
    )


def load_reference_report(run_dir: Path, ref_run_id: str) -> Dict[str, Any] | None:
    path = run_dir.parent / ref_run_id / "reports" / "validation_subject.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summarize_reference(report: Dict[str, Any]) -> Dict[str, float | None]:
    phase = report.get("phase", {}).get("fusion", {})
    torque = report.get("torque", {}).get("fusion", {})
    stress_phase = report.get("stress", {}).get("phase", {}).get("fusion", {})
    stress_torque = report.get("stress", {}).get("torque", {}).get("fusion", {})
    high_target_rmse = torque.get("high_target_rmse")
    if high_target_rmse is None:
        high_target_rmse = stress_torque.get("error_by_target_range", {}).get("high", {}).get("rmse")
    phase_sel = phase.get("phase_selection_score")
    torque_sel = torque.get("torque_selection_score")
    if phase_sel is None:
        phase_sel = phase_selection_score(
            float(phase.get("test_macro_f1", 0.0)),
            float(stress_phase.get("emg_noise", {}).get("macro_f1", 0.0)),
            float(stress_phase.get("drop_eeg", {}).get("macro_f1", 0.0)),
            load_yaml(Path(report["run_dir"]) / "config.yaml"),
        )
    if torque_sel is None:
        torque_sel = torque_selection_score(
            float(torque.get("test_rmse", 0.0)),
            float(stress_torque.get("emg_noise", {}).get("rmse", torque.get("test_rmse", 0.0))),
            float(stress_torque.get("drop_eeg", {}).get("rmse", torque.get("test_rmse", 0.0))),
            float(high_target_rmse if high_target_rmse is not None else torque.get("test_rmse", 0.0)),
            load_yaml(Path(report["run_dir"]) / "config.yaml"),
        )
    return {
        "phase_clean_macro_f1": phase.get("test_macro_f1"),
        "phase_gain_over_emg_macro_f1": phase.get("delta_over_emg_macro_f1"),
        "phase_emg_noise_macro_f1": stress_phase.get("emg_noise", {}).get("macro_f1"),
        "phase_drop_eeg_macro_f1": stress_phase.get("drop_eeg", {}).get("macro_f1"),
        "torque_clean_rmse": torque.get("test_rmse"),
        "torque_drop_eeg_rmse": stress_torque.get("drop_eeg", {}).get("rmse"),
        "torque_emg_noise_rmse": stress_torque.get("emg_noise", {}).get("rmse"),
        "high_target_rmse": high_target_rmse,
        "phase_selection_score": phase_sel,
        "torque_selection_score": torque_sel,
    }


def build_comparison(current: Dict[str, float | None], reference: Dict[str, float | None] | None) -> Dict[str, float | None] | None:
    if reference is None:
        return None
    keys = [
        "phase_clean_macro_f1",
        "phase_gain_over_emg_macro_f1",
        "phase_emg_noise_macro_f1",
        "phase_drop_eeg_macro_f1",
        "torque_clean_rmse",
        "torque_emg_noise_rmse",
        "torque_drop_eeg_rmse",
        "high_target_rmse",
        "phase_selection_score",
        "torque_selection_score",
    ]
    out: Dict[str, float | None] = {}
    for key in keys:
        cur = current.get(key)
        ref = reference.get(key)
        out[f"{key}_delta"] = None if cur is None or ref is None else float(cur - ref)
    return out


def compute_run_flags(current: Dict[str, float | None], refs: Dict[str, Dict[str, float | None]]) -> Dict[str, bool]:
    clean_score = float(current["phase_clean_macro_f1"] or 0.0) - 0.25 * float(current["torque_clean_rmse"] or 0.0)
    robust_phase_score = float(current["phase_selection_score"] or 0.0)
    robust_torque_score = -float(current["torque_selection_score"] or 0.0)
    tradeoff_score = robust_phase_score - 0.25 * float(current["torque_selection_score"] or 0.0)

    clean_refs = [float(ref["phase_clean_macro_f1"] or 0.0) - 0.25 * float(ref["torque_clean_rmse"] or 0.0) for ref in refs.values()]
    robust_phase_refs = [float(ref["phase_selection_score"] or 0.0) for ref in refs.values()]
    robust_torque_refs = [-float(ref["torque_selection_score"] or 0.0) for ref in refs.values()]
    tradeoff_refs = [float(ref["phase_selection_score"] or 0.0) - 0.25 * float(ref["torque_selection_score"] or 0.0) for ref in refs.values()]

    return {
        "best_clean": clean_score >= max(clean_refs) if clean_refs else True,
        "best_robust_phase": robust_phase_score >= max(robust_phase_refs) if robust_phase_refs else True,
        "best_robust_torque": robust_torque_score >= max(robust_torque_refs) if robust_torque_refs else True,
        "best_tradeoff": tradeoff_score >= max(tradeoff_refs) if tradeoff_refs else True,
    }


def acceptance_summary(current: Dict[str, float | None]) -> Dict[str, bool]:
    return {
        "phase_clean_macro_f1_gte_0_67": float(current["phase_clean_macro_f1"] or 0.0) >= 0.67,
        "phase_gain_over_emg_gte_0_10": float(current["phase_gain_over_emg_macro_f1"] or 0.0) >= 0.10,
        "phase_emg_noise_gte_0_40": float(current["phase_emg_noise_macro_f1"] or 0.0) >= 0.40,
        "phase_drop_eeg_gte_0_50": float(current["phase_drop_eeg_macro_f1"] or 0.0) >= 0.50,
        "torque_clean_rmse_lte_0_87": float(current["torque_clean_rmse"] or 9e9) <= 0.87,
        "torque_gain_over_emg_rmse_lte_neg_0_05": float(current.get("torque_gain_over_emg_rmse", 9e9) or 9e9) <= -0.05,
        "high_target_rmse_lte_1_30": float(current["high_target_rmse"] or 9e9) <= 1.30,
    }


def render_report_markdown(report: Dict[str, Any]) -> str:
    ds = report["dataset"]
    phase = report["phase"]
    torque = report["torque"]
    torque_fusion = torque.get("fusion", torque)
    torque_emg = torque.get("emg") if isinstance(torque, dict) and "fusion" in torque else None
    stress = report.get("stress", {})
    stress_phase = stress.get("phase", {})
    stress_torque = stress.get("torque", {})
    syn = report.get("synergy")
    comparison_v3 = report.get("comparison_to_v3")
    comparison_v4 = report.get("comparison_to_v4")
    comparison_v5 = report.get("comparison_to_v5")
    acceptance = report.get("acceptance", {})

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
        f"- Flags: clean `{report.get('best_clean')}`, robust-phase `{report.get('best_robust_phase')}`, robust-torque `{report.get('best_robust_torque')}`, tradeoff `{report.get('best_tradeoff')}`",
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
            "| Modality | Test Macro-F1 | Smoothed Macro-F1 | Test Acc | Balanced Acc | Source |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for modality in ["eeg", "emg", "fusion"]:
        m = phase[modality]
        lines.append(
            f"| {modality.upper()} | {metric(m.get('test_macro_f1'))} | {metric(m.get('test_macro_f1_smoothed'))} | {metric(m.get('test_acc'))} | "
            f"{metric(m.get('test_balanced_acc'))} | {m.get('source', '-')} |"
        )

    lines.extend(
        [
            "",
            "## Fusion Gain",
            "",
            f"- `delta_over_emg_macro_f1`: `{metric(phase.get('fusion', {}).get('delta_over_emg_macro_f1'))}`",
            f"- `delta_over_emg_acc`: `{metric(phase.get('fusion', {}).get('delta_over_emg_acc'))}`",
            f"- `delta_over_emg_torque_rmse`: `{metric(torque_fusion.get('delta_over_emg_torque_rmse'))}`",
            f"- `phase_selection_score`: `{metric(report.get('phase_selection_score'))}`",
            f"- `torque_selection_score`: `{metric(report.get('torque_selection_score'))}`",
            f"- `phase_history_path`: `{report.get('phase_history_path') or '-'}`",
            f"- `torque_history_path`: `{report.get('torque_history_path') or '-'}`",
            "",
            "## Stress Phase",
            "",
            "| Model | Clean F1 | EMG Noise | Temporal Shift | Drop EEG | Drop EMG |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for modality in ["emg", "fusion"]:
        item = stress_phase.get(modality, {})
        lines.append(
            f"| {modality.upper()} | {metric(item.get('clean', {}).get('macro_f1'))} | {metric(item.get('emg_noise', {}).get('macro_f1'))} | "
            f"{metric(item.get('temporal_shift', {}).get('macro_f1'))} | {metric(item.get('drop_eeg', {}).get('macro_f1'))} | {metric(item.get('drop_emg', {}).get('macro_f1'))} |"
        )

    lines.extend(
        [
            "",
            "## Torque Prediction",
            "",
            "| Model | Test RMSE | MAE | Mean Baseline RMSE | Approx. R2 | Corr | Source |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            f"| Fusion | {metric(torque_fusion.get('test_rmse'))} | {metric(torque_fusion.get('test_mae'))} | {metric(torque_fusion.get('baseline_mean_rmse'))} | {metric(torque_fusion.get('approx_r2'))} | {metric(torque_fusion.get('corr') or torque_fusion.get('test_corr'))} | {torque_fusion.get('source', '-')} |",
        ]
    )
    if torque_emg is not None:
        lines.append(
            f"| EMG | {metric(torque_emg.get('test_rmse'))} | {metric(torque_emg.get('test_mae'))} | {metric(torque_emg.get('baseline_mean_rmse'))} | {metric(torque_emg.get('approx_r2'))} | {metric(torque_emg.get('corr') or torque_emg.get('test_corr'))} | {torque_emg.get('source', '-')} |"
        )
    torque_range = stress_torque.get("fusion", {}).get("error_by_target_range", {})
    lines.extend(
        [
            "",
            "## Stress Torque",
            "",
            "| Model | Clean RMSE | EMG Noise | Temporal Shift | Drop EEG | Drop EMG |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| FUSION | {metric(stress_torque.get('fusion', {}).get('clean', {}).get('rmse'))} | {metric(stress_torque.get('fusion', {}).get('emg_noise', {}).get('rmse'))} | {metric(stress_torque.get('fusion', {}).get('temporal_shift', {}).get('rmse'))} | {metric(stress_torque.get('fusion', {}).get('drop_eeg', {}).get('rmse'))} | {metric(stress_torque.get('fusion', {}).get('drop_emg', {}).get('rmse'))} |",
            f"| EMG | {metric(stress_torque.get('emg', {}).get('clean', {}).get('rmse'))} | {metric(stress_torque.get('emg', {}).get('emg_noise', {}).get('rmse'))} | {metric(stress_torque.get('emg', {}).get('temporal_shift', {}).get('rmse'))} | - | - |",
            "",
            "## Torque Error Range",
            "",
            f"- Low target RMSE: `{metric(torque_range.get('low', {}).get('rmse'))}` over `{torque_range.get('low', {}).get('count', 0)}` segments",
            f"- Mid target RMSE: `{metric(torque_range.get('mid', {}).get('rmse'))}` over `{torque_range.get('mid', {}).get('count', 0)}` segments",
            f"- High target RMSE: `{metric(torque_range.get('high', {}).get('rmse'))}` over `{torque_range.get('high', {}).get('count', 0)}` segments",
            "",
            "## Generalization Gap",
            "",
            f"- Phase clean gap (val-test): `{metric(phase.get('fusion', {}).get('generalization_gap_clean'))}`",
            f"- Phase EMG-noise gap (val-test): `{metric(phase.get('fusion', {}).get('generalization_gap_emg_noise'))}`",
            f"- Torque clean gap (test-val): `{metric(torque.get('fusion', {}).get('generalization_gap_clean'))}`",
            f"- Torque EMG-noise gap (test-val): `{metric(torque.get('fusion', {}).get('generalization_gap_emg_noise'))}`",
            "",
            "## Reference Comparison",
            "",
            "| Reference | dPhase Clean | dPhase EMG Noise | dPhase Drop EEG | dTorque Clean RMSE | dHigh-Target RMSE | dPhase Selection | dTorque Selection |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| v3 | {metric((comparison_v3 or {}).get('phase_clean_macro_f1_delta'))} | {metric((comparison_v3 or {}).get('phase_emg_noise_macro_f1_delta'))} | {metric((comparison_v3 or {}).get('phase_drop_eeg_macro_f1_delta'))} | {metric((comparison_v3 or {}).get('torque_clean_rmse_delta'))} | {metric((comparison_v3 or {}).get('high_target_rmse_delta'))} | {metric((comparison_v3 or {}).get('phase_selection_score_delta'))} | {metric((comparison_v3 or {}).get('torque_selection_score_delta'))} |",
            f"| v4 | {metric((comparison_v4 or {}).get('phase_clean_macro_f1_delta'))} | {metric((comparison_v4 or {}).get('phase_emg_noise_macro_f1_delta'))} | {metric((comparison_v4 or {}).get('phase_drop_eeg_macro_f1_delta'))} | {metric((comparison_v4 or {}).get('torque_clean_rmse_delta'))} | {metric((comparison_v4 or {}).get('high_target_rmse_delta'))} | {metric((comparison_v4 or {}).get('phase_selection_score_delta'))} | {metric((comparison_v4 or {}).get('torque_selection_score_delta'))} |",
            f"| v5 | {metric((comparison_v5 or {}).get('phase_clean_macro_f1_delta'))} | {metric((comparison_v5 or {}).get('phase_emg_noise_macro_f1_delta'))} | {metric((comparison_v5 or {}).get('phase_drop_eeg_macro_f1_delta'))} | {metric((comparison_v5 or {}).get('torque_clean_rmse_delta'))} | {metric((comparison_v5 or {}).get('high_target_rmse_delta'))} | {metric((comparison_v5 or {}).get('phase_selection_score_delta'))} | {metric((comparison_v5 or {}).get('torque_selection_score_delta'))} |",
            "",
            "## Acceptance",
            "",
            f"- Phase clean >= 0.67: `{acceptance.get('phase_clean_macro_f1_gte_0_67')}`",
            f"- Phase gain over EMG >= 0.10: `{acceptance.get('phase_gain_over_emg_gte_0_10')}`",
            f"- Phase EMG noise >= 0.40: `{acceptance.get('phase_emg_noise_gte_0_40')}`",
            f"- Phase drop EEG >= 0.50: `{acceptance.get('phase_drop_eeg_gte_0_50')}`",
            f"- Torque clean RMSE <= 0.87: `{acceptance.get('torque_clean_rmse_lte_0_87')}`",
            f"- Torque gain over EMG RMSE <= -0.05: `{acceptance.get('torque_gain_over_emg_rmse_lte_neg_0_05')}`",
            f"- High-target RMSE <= 1.30: `{acceptance.get('high_target_rmse_lte_1_30')}`",
            "",
            "## Interpretation",
            "",
            "- This benchmark reflects controller-development conditions, not a clinical deployment claim.",
            "- The headline scientific question is whether fusion beats EMG under subject split and stays competitive under stress.",
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
        torque = {
            "fusion": evaluate_torque(run_dir, cfg, modality="fusion"),
            "emg": evaluate_torque(run_dir, cfg, modality="emg"),
        }
        phase["fusion"]["delta_over_emg_macro_f1"] = float(phase["fusion"]["test_macro_f1"] - phase["emg"]["test_macro_f1"])
        phase["fusion"]["delta_over_emg_acc"] = float(phase["fusion"]["test_acc"] - phase["emg"]["test_acc"])
        torque["fusion"]["delta_over_emg_torque_rmse"] = float(torque["fusion"]["test_rmse"] - torque["emg"]["test_rmse"])
        stress = {
            "phase": {
                "eeg": _phase_stress_summary(run_dir, cfg, "eeg"),
                "emg": _phase_stress_summary(run_dir, cfg, "emg"),
                "fusion": _phase_stress_summary(run_dir, cfg, "fusion"),
            },
            "torque": {
                "emg": _torque_stress_summary(run_dir, cfg, "emg"),
                "fusion": _torque_stress_summary(run_dir, cfg, "fusion"),
            },
        }
        ensure_selection_metrics(phase, torque, stress, cfg)
    else:
        loaded = load_artifact_metrics(run_dir, cfg)
        phase = loaded["phase"]
        torque = loaded["torque"]
        stress = loaded["stress"]

    current_summary = {
        "phase_clean_macro_f1": phase.get("fusion", {}).get("test_macro_f1"),
        "phase_gain_over_emg_macro_f1": phase.get("fusion", {}).get("delta_over_emg_macro_f1"),
        "phase_emg_noise_macro_f1": stress.get("phase", {}).get("fusion", {}).get("emg_noise", {}).get("macro_f1"),
        "phase_drop_eeg_macro_f1": stress.get("phase", {}).get("fusion", {}).get("drop_eeg", {}).get("macro_f1"),
        "torque_clean_rmse": torque.get("fusion", {}).get("test_rmse"),
        "torque_gain_over_emg_rmse": torque.get("fusion", {}).get("delta_over_emg_torque_rmse"),
        "torque_emg_noise_rmse": stress.get("torque", {}).get("fusion", {}).get("emg_noise", {}).get("rmse"),
        "torque_drop_eeg_rmse": stress.get("torque", {}).get("fusion", {}).get("drop_eeg", {}).get("rmse"),
        "high_target_rmse": torque.get("fusion", {}).get("high_target_rmse"),
        "phase_selection_score": phase.get("fusion", {}).get("phase_selection_score"),
        "torque_selection_score": torque.get("fusion", {}).get("torque_selection_score"),
    }
    references = {
        name: summarize_reference(ref_report)
        for name, ref_id in REFERENCE_RUNS.items()
        if (ref_report := load_reference_report(run_dir, ref_id)) is not None
    }
    comparison_to_v3 = build_comparison(current_summary, references.get("v3"))
    comparison_to_v4 = build_comparison(current_summary, references.get("v4"))
    comparison_to_v5 = build_comparison(current_summary, references.get("v5"))
    flags = compute_run_flags(current_summary, references)
    acceptance = acceptance_summary(current_summary)

    report = {
        "run_dir": str(run_dir),
        "dataset": dataset,
        "phase": phase,
        "torque": torque,
        "stress": stress,
        "synergy": synergy,
        "phase_selection_score": current_summary["phase_selection_score"],
        "torque_selection_score": current_summary["torque_selection_score"],
        "phase_history_path": phase.get("fusion", {}).get("phase_history_path"),
        "torque_history_path": torque.get("fusion", {}).get("torque_history_path"),
        "comparison_to_v3": comparison_to_v3,
        "comparison_to_v4": comparison_to_v4,
        "comparison_to_v5": comparison_to_v5,
        "best_clean": flags["best_clean"],
        "best_robust_phase": flags["best_robust_phase"],
        "best_robust_torque": flags["best_robust_torque"],
        "best_tradeoff": flags["best_tradeoff"],
        "acceptance": acceptance,
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
