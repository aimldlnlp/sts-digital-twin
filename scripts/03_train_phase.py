from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import load_yaml, save_json, set_seed
from src.features.dataset import StandardScaler1D, get_channel_counts, make_segments, select_split_groups, split_indices
from src.models.nets import build_phase_model, save_model_checkpoint
from src.models.train_utils import (
    apply_eval_stress_np,
    apply_training_modality_augmentation,
    FocalLoss,
    SegmentDataset,
    balanced_accuracy,
    classwise_f1,
    compute_class_weights,
    confusion_matrix,
    get_device,
    kl_consistency_loss,
    macro_f1,
    phase_selection_score,
    smooth_predictions_by_group,
    summarize_confusion,
)


def build_loss(cfg: dict, y_train: np.ndarray, device: torch.device) -> tuple[torch.nn.Module, np.ndarray]:
    phase_cfg = cfg.get("model", {}).get("phase", {})
    weights = compute_class_weights(y_train, n_classes=4)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_type = str(phase_cfg.get("loss_type", "weighted_ce"))
    if loss_type == "focal":
        gamma = float(phase_cfg.get("focal_gamma", 2.0))
        return FocalLoss(weight=weight_tensor, gamma=gamma), weights
    return torch.nn.CrossEntropyLoss(weight=weight_tensor), weights


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    ys = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            preds.append(logits.argmax(axis=1))
            ys.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def predict_labels(
    model: torch.nn.Module,
    x_scaled: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        SegmentDataset(x_scaled.astype(np.float32), np.zeros(len(x_scaled), dtype=np.int64)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_phase_case(
    model: torch.nn.Module,
    x_raw: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler1D,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    pred = predict_labels(model, scaler.transform(x_raw), batch_size=batch_size, device=device)
    return {
        "macro_f1": float(macro_f1(pred, y, n_classes=4)),
        "acc": float((pred == y).mean()),
        "balanced_acc": float(balanced_accuracy(pred, y, n_classes=4)),
    }


def aggregate_phase_validation_replay(
    model: torch.nn.Module,
    x_raw: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler1D,
    batch_size: int,
    device: torch.device,
    cfg: dict,
) -> dict[str, float | list[float]]:
    clean_metrics = evaluate_phase_case(model, x_raw, y, scaler, batch_size=batch_size, device=device)
    emg_noise_scores = []
    for seed in [410, 411, 412]:
        emg_noise_metrics = evaluate_phase_case(
            model,
            apply_eval_stress_np(x_raw, cfg, "fusion", "emg_noise", seed=int(cfg["seed"]) + seed),
            y,
            scaler,
            batch_size=batch_size,
            device=device,
        )
        emg_noise_scores.append(float(emg_noise_metrics["macro_f1"]))
    drop_eeg_scores = []
    for seed in [413]:
        drop_eeg_metrics = evaluate_phase_case(
            model,
            apply_eval_stress_np(x_raw, cfg, "fusion", "drop_eeg", seed=int(cfg["seed"]) + seed),
            y,
            scaler,
            batch_size=batch_size,
            device=device,
        )
        drop_eeg_scores.append(float(drop_eeg_metrics["macro_f1"]))
    return {
        "clean_macro_f1": float(clean_metrics["macro_f1"]),
        "clean_acc": float(clean_metrics["acc"]),
        "clean_balanced_acc": float(clean_metrics["balanced_acc"]),
        "emg_noise_macro_f1": float(np.mean(emg_noise_scores)),
        "drop_eeg_macro_f1": float(np.mean(drop_eeg_scores)),
        "emg_noise_macro_f1_seeds": emg_noise_scores,
        "drop_eeg_macro_f1_seeds": drop_eeg_scores,
        "selection_score": float(
            phase_selection_score(
                float(clean_metrics["macro_f1"]),
                float(np.mean(emg_noise_scores)),
                float(np.mean(drop_eeg_scores)),
                cfg,
            )
        ),
    }


def select_phase_history(history: list[dict[str, float | list[float]]], cfg: dict) -> dict[str, float | list[float]]:
    clean_margin = float(cfg.get("model", {}).get("phase", {}).get("selection_clean_margin", 0.015))
    best_clean = max(float(item["clean_macro_f1"]) for item in history)
    clean_threshold = best_clean - clean_margin
    candidates = [item for item in history if float(item["clean_macro_f1"]) >= clean_threshold]
    selected = max(
        candidates,
        key=lambda item: (
            float(item["selection_score"]),
            float(item["clean_macro_f1"]),
            float(item.get("emg_noise_macro_f1", 0.0)),
            -int(item["epoch"]),
        ),
    )
    return {
        "selected_epoch": int(selected["epoch"]),
        "selected_summary": copy.deepcopy(selected),
        "best_clean_epoch": int(max(history, key=lambda item: float(item["clean_macro_f1"]))["epoch"]),
        "best_clean_macro_f1": float(best_clean),
        "clean_threshold": float(clean_threshold),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--modality", type=str, default="fusion", choices=["fusion", "eeg", "emg"])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_yaml(Path(args.config)) if args.config else load_yaml(run_dir / "config.yaml")
    set_seed(int(cfg["seed"]))

    seg = make_segments(run_dir, cfg, modality=args.modality)
    x = seg["X"]
    y = seg["y_phase"]
    groups = select_split_groups(seg, cfg)
    tr_idx, va_idx, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 1, groups=groups)

    scaler = StandardScaler1D().fit(x[tr_idx])
    xs = scaler.transform(x)

    device = get_device()
    eeg_ch, emg_ch = get_channel_counts(cfg)
    model = build_phase_model(cfg, args.modality).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    loss_fn, class_weights = build_loss(cfg, y[tr_idx], device)
    phase_cfg = cfg.get("model", {}).get("phase", {})
    consistency_weight = float(phase_cfg.get("consistency_weight", 0.0))

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(SegmentDataset(xs[tr_idx], y[tr_idx]), batch_size=batch_size, shuffle=True, drop_last=False)
    x_val_raw = x[va_idx]
    y_val = y[va_idx]

    best_path = None
    best_summary: dict[str, float | list[float] | str | int | None] = {}
    epochs = int(cfg["train"]["epochs_phase"])
    epoch_states: list[dict[str, torch.Tensor]] = []
    history: list[dict[str, float | list[float]]] = []

    for epoch in range(epochs):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"[phase-{args.modality}] epoch {epoch + 1}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            if args.modality == "fusion":
                xb_corrupt = apply_training_modality_augmentation(
                    xb,
                    args.modality,
                    eeg_ch,
                    emg_ch,
                    cfg,
                    profile="phase_fusion",
                    epoch_index=epoch,
                )
                logits_clean = model(xb)
                logits_corrupt = model(xb_corrupt)
                loss = loss_fn(logits_clean, yb) + loss_fn(logits_corrupt, yb)
                if consistency_weight > 0:
                    loss = loss + consistency_weight * kl_consistency_loss(logits_clean, logits_corrupt)
            else:
                xb_aug = apply_training_modality_augmentation(
                    xb,
                    args.modality,
                    eeg_ch,
                    emg_ch,
                    cfg,
                    profile="single_modality",
                )
                logits_clean = model(xb_aug)
                loss = loss_fn(logits_clean, yb)
            loss.backward()
            opt.step()

        if args.modality == "fusion":
            val_summary = aggregate_phase_validation_replay(
                model,
                x_val_raw,
                y_val,
                scaler,
                batch_size=batch_size,
                device=device,
                cfg=cfg,
            )
        else:
            clean_metrics = evaluate_phase_case(model, x_val_raw, y_val, scaler, batch_size=batch_size, device=device)
            val_summary = {
                "clean_macro_f1": float(clean_metrics["macro_f1"]),
                "clean_acc": float(clean_metrics["acc"]),
                "clean_balanced_acc": float(clean_metrics["balanced_acc"]),
                "selection_score": float(clean_metrics["macro_f1"]),
            }
        val_summary["epoch"] = int(epoch + 1)
        history.append(copy.deepcopy(val_summary))
        epoch_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    if args.modality == "fusion":
        selection_meta = select_phase_history(history, cfg)
        selected_epoch = int(selection_meta["selected_epoch"])
        best_summary = dict(selection_meta["selected_summary"])
        best_summary["history_path"] = str(run_dir / "artifacts" / f"phase_history_{args.modality}.json")
        out = run_dir / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        history_payload = {
            "modality": args.modality,
            "selection_rule": "lexicographic_clean_margin_then_selection_score",
            "selection_clean_margin": float(cfg.get("model", {}).get("phase", {}).get("selection_clean_margin", 0.015)),
            "selected_epoch": selected_epoch,
            "best_clean_epoch": int(selection_meta["best_clean_epoch"]),
            "best_clean_macro_f1": float(selection_meta["best_clean_macro_f1"]),
            "clean_threshold": float(selection_meta["clean_threshold"]),
            "epochs": history,
        }
        save_json(out / f"phase_history_{args.modality}.json", history_payload)
    else:
        selected_epoch = int(max(history, key=lambda item: float(item["clean_macro_f1"]))["epoch"])
        best_summary = dict(max(history, key=lambda item: float(item["clean_macro_f1"])))

    out = run_dir / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / f"phase_model_{args.modality}.pt"
    model.load_state_dict(epoch_states[selected_epoch - 1])
    save_model_checkpoint(
        best_path,
        model,
        cfg,
        task="phase",
        modality=args.modality,
        extra=best_summary | {"selected_epoch": selected_epoch, "best_val_score": float(best_summary["selection_score"])},
    )

    test_loader = DataLoader(SegmentDataset(xs[te_idx], y[te_idx]), batch_size=batch_size, shuffle=False, drop_last=False)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint)

    pred, y_test = evaluate_model(model, test_loader, device)
    smoothing_window = int(cfg.get("eval", {}).get("phase_smoothing_window", cfg.get("model", {}).get("phase", {}).get("smoothing_window", 1)))
    pred_smooth = smooth_predictions_by_group(pred, seg["trial"][te_idx], seg["start"][te_idx], smoothing_window)

    conf = confusion_matrix(pred, y_test, n_classes=4)
    conf_smooth = confusion_matrix(pred_smooth, y_test, n_classes=4)
    meta = {
        "modality": args.modality,
        "split_strategy": cfg["train"].get("split_strategy", "segment"),
        "loss_type": str(cfg.get("model", {}).get("phase", {}).get("loss_type", "weighted_ce")),
        "class_weights": class_weights.tolist(),
        "consistency_weight": consistency_weight,
        "phase_smoothing_window": smoothing_window,
        "selected_epoch": selected_epoch,
        "val_macro_f1_best": float(best_summary.get("clean_macro_f1", 0.0)),
        "phase_selection_score": float(best_summary.get("selection_score", 0.0)),
        "val_emg_noise_macro_f1_best": best_summary.get("emg_noise_macro_f1"),
        "val_drop_eeg_macro_f1_best": best_summary.get("drop_eeg_macro_f1"),
        "phase_history_path": best_summary.get("history_path"),
        "test_macro_f1": float(macro_f1(pred, y_test, n_classes=4)),
        "test_acc": float((pred == y_test).mean()),
        "test_balanced_acc": float(balanced_accuracy(pred, y_test, n_classes=4)),
        "test_macro_f1_smoothed": float(macro_f1(pred_smooth, y_test, n_classes=4)),
        "test_acc_smoothed": float((pred_smooth == y_test).mean()),
        "test_balanced_acc_smoothed": float(balanced_accuracy(pred_smooth, y_test, n_classes=4)),
        "per_class_f1": classwise_f1(pred, y_test, n_classes=4),
        "per_class_f1_smoothed": classwise_f1(pred_smooth, y_test, n_classes=4),
        "confusion_summary": summarize_confusion(conf),
        "confusion_summary_smoothed": summarize_confusion(conf_smooth),
        "n_test_segments": int(len(te_idx)),
        "model_path": str(best_path),
    }

    with open(run_dir / "artifacts" / f"scaler_{args.modality}.json", "w") as handle:
        json.dump(scaler.to_dict(), handle)
    save_json(run_dir / "artifacts" / f"phase_metrics_{args.modality}.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
