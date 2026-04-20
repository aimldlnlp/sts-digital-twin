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
from src.features.dataset import (
    StandardScaler1D,
    StandardScalerTarget,
    get_channel_counts,
    make_segments,
    select_split_groups,
    split_indices,
)
from src.models.nets import build_torque_model, save_model_checkpoint
from src.models.train_utils import (
    SegmentDataset,
    apply_eval_stress_np,
    apply_training_modality_augmentation,
    compute_torque_range_weights,
    corrcoef,
    get_device,
    mae,
    regression_consistency_loss,
    r2_score,
    rmse,
    sequence_smoothness_loss,
    torque_error_by_range,
    torque_selection_score,
    weighted_regression_loss,
)


def build_regression_loss(cfg: dict) -> tuple[str, float]:
    torque_cfg = cfg.get("model", {}).get("torque", {})
    loss_type = str(torque_cfg.get("regression_loss", "huber"))
    return loss_type, float(torque_cfg.get("huber_delta", 1.0))


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    target_scaler: StandardScalerTarget,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    ys = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(target_scaler.inverse_transform(pred))
            ys.append(target_scaler.inverse_transform(yb.numpy()))
    return np.concatenate(preds), np.concatenate(ys)


def predict_model(
    model: torch.nn.Module,
    x_scaled: np.ndarray,
    target_scaler: StandardScalerTarget,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        SegmentDataset(x_scaled.astype(np.float32), np.zeros(len(x_scaled), dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    pred = np.concatenate(preds)
    return target_scaler.inverse_transform(pred)


def summarize_torque(pred: np.ndarray, y_true: np.ndarray) -> dict[str, float | None]:
    return {
        "rmse": float(rmse(pred, y_true)),
        "mae": float(mae(pred, y_true)),
        "corr": corrcoef(pred, y_true),
        "r2": r2_score(pred, y_true),
    }


def evaluate_torque_case(
    model: torch.nn.Module,
    x_raw: np.ndarray,
    y_true: np.ndarray,
    x_scaler: StandardScaler1D,
    y_scaler: StandardScalerTarget,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | None]:
    pred = predict_model(model, x_scaler.transform(x_raw), y_scaler, batch_size=batch_size, device=device)
    return summarize_torque(pred, y_true)


def aggregate_torque_validation_replay(
    model: torch.nn.Module,
    x_raw: np.ndarray,
    y_true: np.ndarray,
    x_scaler: StandardScaler1D,
    y_scaler: StandardScalerTarget,
    batch_size: int,
    device: torch.device,
    cfg: dict,
) -> dict[str, float | list[float] | None]:
    clean_metrics = evaluate_torque_case(model, x_raw, y_true, x_scaler, y_scaler, batch_size=batch_size, device=device)
    emg_noise_scores = []
    for seed in [710, 711]:
        pred = predict_model(
            model,
            x_scaler.transform(apply_eval_stress_np(x_raw, cfg, "fusion", "emg_noise", seed=int(cfg["seed"]) + seed)),
            y_scaler,
            batch_size=batch_size,
            device=device,
        )
        emg_noise_scores.append(float(rmse(pred, y_true)))
    drop_eeg_scores = []
    for seed in [712]:
        pred = predict_model(
            model,
            x_scaler.transform(apply_eval_stress_np(x_raw, cfg, "fusion", "drop_eeg", seed=int(cfg["seed"]) + seed)),
            y_scaler,
            batch_size=batch_size,
            device=device,
        )
        drop_eeg_scores.append(float(rmse(pred, y_true)))
    clean_pred = predict_model(model, x_scaler.transform(x_raw), y_scaler, batch_size=batch_size, device=device)
    high_target_rmse = torque_error_by_range(y_true, clean_pred)["high"]["rmse"]
    high_target_rmse = float(high_target_rmse) if high_target_rmse is not None else float(clean_metrics["rmse"])
    return {
        "clean_rmse": float(clean_metrics["rmse"]),
        "clean_mae": float(clean_metrics["mae"]),
        "clean_corr": clean_metrics["corr"],
        "clean_r2": clean_metrics["r2"],
        "emg_noise_rmse": float(np.mean(emg_noise_scores)),
        "drop_eeg_rmse": float(np.mean(drop_eeg_scores)),
        "emg_noise_rmse_seeds": emg_noise_scores,
        "drop_eeg_rmse_seeds": drop_eeg_scores,
        "high_target_rmse": float(high_target_rmse),
        "selection_score": float(
            torque_selection_score(
                float(clean_metrics["rmse"]),
                float(np.mean(emg_noise_scores)),
                float(np.mean(drop_eeg_scores)),
                high_target_rmse,
                cfg,
            )
        ),
    }


def select_torque_history(history: list[dict[str, float | list[float] | None]], cfg: dict) -> dict[str, float | list[float] | None | int]:
    clean_margin_pct = float(cfg.get("model", {}).get("torque", {}).get("selection_clean_margin_pct", 0.02))
    best_clean = min(float(item["clean_rmse"]) for item in history)
    clean_threshold = best_clean * (1.0 + clean_margin_pct)
    candidates = [item for item in history if float(item["clean_rmse"]) <= clean_threshold]
    selected = min(
        candidates,
        key=lambda item: (
            float(item["selection_score"]),
            float(item["clean_rmse"]),
            float(item.get("high_target_rmse", 9e9) or 9e9),
            int(item["epoch"]),
        ),
    )
    return {
        "selected_epoch": int(selected["epoch"]),
        "selected_summary": copy.deepcopy(selected),
        "best_clean_epoch": int(min(history, key=lambda item: float(item["clean_rmse"]))["epoch"]),
        "best_clean_rmse": float(best_clean),
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
    y = seg["y_tau"]
    groups = select_split_groups(seg, cfg)
    tr_idx, va_idx, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 2, groups=groups)

    x_scaler = StandardScaler1D().fit(x[tr_idx])
    xs = x_scaler.transform(x)
    y_scaler = StandardScalerTarget().fit(y[tr_idx])
    ys = y_scaler.transform(y)
    eeg_ch, emg_ch = get_channel_counts(cfg)

    device = get_device()
    model = build_torque_model(cfg, args.modality).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    loss_type, huber_delta = build_regression_loss(cfg)
    smoothness_weight = float(cfg.get("model", {}).get("torque", {}).get("smoothness_weight", 0.0))
    consistency_weight = float(cfg.get("model", {}).get("torque", {}).get("consistency_weight", 0.0))
    range_weights = dict(cfg.get("model", {}).get("torque", {}).get("range_weights", {"low": 1.0, "mid": 1.0, "high": 1.0}))
    aux_corruption_prob = float(cfg.get("augment", {}).get("train", {}).get("torque_fusion", {}).get("aux_corruption_prob", 1.0))

    batch_size = int(cfg["train"]["batch_size"])
    train_order = np.lexsort((seg["start"][tr_idx], seg["trial"][tr_idx]))
    train_idx_sorted = tr_idx[train_order]
    train_sample_weight = compute_torque_range_weights(y[train_idx_sorted], range_weights=range_weights)
    train_loader = DataLoader(
        SegmentDataset(xs[train_idx_sorted], ys[train_idx_sorted], seg["trial"][train_idx_sorted], train_sample_weight),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    x_val_raw = x[va_idx]
    y_val = y[va_idx]

    best_path = None
    best_summary: dict[str, float | list[float] | str | int | None] = {}
    epochs = int(cfg["train"]["epochs_torque"])
    epoch_states: list[dict[str, torch.Tensor]] = []
    history: list[dict[str, float | list[float] | None]] = []

    for epoch in range(epochs):
        model.train()
        for xb, yb, trial_b, weight_b in tqdm(train_loader, desc=f"[torque-{args.modality}] epoch {epoch + 1}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device).float()
            trial_b = trial_b.to(device)
            weight_b = weight_b.to(device).float()
            opt.zero_grad()
            if args.modality == "fusion":
                pred_clean = model(xb)
                loss_clean = weighted_regression_loss(pred_clean, yb, weight_b, loss_type=loss_type, huber_delta=huber_delta)
                smooth_clean = sequence_smoothness_loss(pred_clean, yb, trial_b)
                loss = loss_clean + smoothness_weight * smooth_clean
                if float(np.random.rand()) < aux_corruption_prob:
                    xb_corrupt = apply_training_modality_augmentation(
                        xb,
                        args.modality,
                        eeg_ch,
                        emg_ch,
                        cfg,
                        profile="torque_fusion",
                        epoch_index=epoch,
                        active_prob=1.0,
                    )
                    pred_corrupt = model(xb_corrupt)
                    loss_corrupt = weighted_regression_loss(pred_corrupt, yb, weight_b, loss_type=loss_type, huber_delta=huber_delta)
                    smooth_corrupt = sequence_smoothness_loss(pred_corrupt, yb, trial_b)
                    loss = loss + 0.65 * loss_corrupt + smoothness_weight * 0.50 * smooth_corrupt
                    if consistency_weight > 0:
                        loss = loss + consistency_weight * regression_consistency_loss(pred_clean, pred_corrupt)
            else:
                xb_aug = apply_training_modality_augmentation(
                    xb,
                    args.modality,
                    eeg_ch,
                    emg_ch,
                    cfg,
                    profile="single_modality",
                )
                pred_clean = model(xb_aug)
                loss_main = weighted_regression_loss(pred_clean, yb, weight_b, loss_type=loss_type, huber_delta=huber_delta)
                loss_smooth = sequence_smoothness_loss(pred_clean, yb, trial_b)
                loss = loss_main + smoothness_weight * loss_smooth
            loss.backward()
            opt.step()

        if args.modality == "fusion":
            val_summary = aggregate_torque_validation_replay(
                model,
                x_val_raw,
                y_val,
                x_scaler,
                y_scaler,
                batch_size=batch_size,
                device=device,
                cfg=cfg,
            )
        else:
            clean_metrics = evaluate_torque_case(model, x_val_raw, y_val, x_scaler, y_scaler, batch_size=batch_size, device=device)
            val_summary = {
                "clean_rmse": float(clean_metrics["rmse"]),
                "clean_mae": float(clean_metrics["mae"]),
                "clean_corr": clean_metrics["corr"],
                "clean_r2": clean_metrics["r2"],
                "selection_score": float(clean_metrics["rmse"]),
            }
        val_summary["epoch"] = int(epoch + 1)
        history.append(copy.deepcopy(val_summary))
        epoch_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    if args.modality == "fusion":
        selection_meta = select_torque_history(history, cfg)
        selected_epoch = int(selection_meta["selected_epoch"])
        best_summary = dict(selection_meta["selected_summary"])
        best_summary["history_path"] = str(run_dir / "artifacts" / f"torque_history_{args.modality}.json")
        out = run_dir / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        history_payload = {
            "modality": args.modality,
            "selection_rule": "lexicographic_clean_margin_then_selection_score",
            "selection_clean_margin_pct": float(cfg.get("model", {}).get("torque", {}).get("selection_clean_margin_pct", 0.02)),
            "selected_epoch": selected_epoch,
            "best_clean_epoch": int(selection_meta["best_clean_epoch"]),
            "best_clean_rmse": float(selection_meta["best_clean_rmse"]),
            "clean_threshold": float(selection_meta["clean_threshold"]),
            "epochs": history,
        }
        save_json(out / f"torque_history_{args.modality}.json", history_payload)
    else:
        selected_epoch = int(min(history, key=lambda item: float(item["clean_rmse"]))["epoch"])
        best_summary = dict(min(history, key=lambda item: float(item["clean_rmse"])))

    out = run_dir / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / f"torque_model_{args.modality}.pt"
    model.load_state_dict(epoch_states[selected_epoch - 1])
    save_model_checkpoint(
        best_path,
        model,
        cfg,
        task="torque",
        modality=args.modality,
        extra=best_summary | {"selected_epoch": selected_epoch, "best_val_score": float(best_summary["selection_score"])},
    )

    test_loader = DataLoader(SegmentDataset(xs[te_idx], ys[te_idx]), batch_size=batch_size, shuffle=False, drop_last=False)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint)
    pred, y_test = evaluate_model(model, test_loader, y_scaler, device)

    mean_baseline = np.full_like(y_test, float(np.mean(y[tr_idx])))
    metrics = {
        "modality": args.modality,
        "split_strategy": cfg["train"].get("split_strategy", "segment"),
        "loss_type": loss_type,
        "selected_epoch": selected_epoch,
        "smoothness_weight": smoothness_weight,
        "consistency_weight": consistency_weight,
        "range_weights": {k: float(v) for k, v in range_weights.items()},
        "val_rmse_best": float(best_summary.get("clean_rmse", 0.0)),
        "torque_selection_score": float(best_summary.get("selection_score", 0.0)),
        "val_emg_noise_rmse_best": best_summary.get("emg_noise_rmse"),
        "val_drop_eeg_rmse_best": best_summary.get("drop_eeg_rmse"),
        "high_target_rmse": torque_error_by_range(y_test, pred)["high"]["rmse"],
        "val_high_target_rmse_best": best_summary.get("high_target_rmse"),
        "torque_history_path": best_summary.get("history_path"),
        "test_rmse": float(rmse(pred, y_test)),
        "test_mae": float(mae(pred, y_test)),
        "test_corr": corrcoef(pred, y_test),
        "test_r2": r2_score(pred, y_test),
        "baseline_mean_rmse": float(rmse(mean_baseline, y_test)),
        "n_test_segments": int(len(te_idx)),
        "model_path": str(best_path),
        "target_scaler_path": str(run_dir / "artifacts" / f"target_scaler_{args.modality}_torque.json"),
    }

    with open(run_dir / "artifacts" / f"scaler_{args.modality}_torque.json", "w") as handle:
        json.dump(x_scaler.to_dict(), handle)
    with open(run_dir / "artifacts" / f"target_scaler_{args.modality}_torque.json", "w") as handle:
        json.dump(y_scaler.to_dict(), handle)
    save_json(run_dir / "artifacts" / f"torque_metrics_{args.modality}.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
