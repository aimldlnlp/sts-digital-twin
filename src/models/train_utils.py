from __future__ import annotations
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.features.dataset import get_channel_counts

class SegmentDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, *extras: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y)
        self.extras = [torch.from_numpy(extra) for extra in extras]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if not self.extras:
            return self.X[i], self.y[i]
        return (self.X[i], self.y[i], *[extra[i] for extra in self.extras])

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())

def macro_f1(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


def classwise_f1(pred: np.ndarray, y: np.ndarray, n_classes: int) -> list[float]:
    out = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        out.append(float(2 * prec * rec / (prec + rec + 1e-9)))
    return out


def confusion_matrix(pred: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, guess in zip(y.astype(np.int64), pred.astype(np.int64)):
        conf[truth, guess] += 1
    return conf


def balanced_accuracy(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    conf = confusion_matrix(pred, y, n_classes)
    recalls = []
    for c in range(n_classes):
        total = conf[c].sum()
        recalls.append(conf[c, c] / (total + 1e-9))
    return float(np.mean(recalls))

def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y)**2)))


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def corrcoef(pred: np.ndarray, y: np.ndarray) -> float | None:
    if len(pred) < 2:
        return None
    pred_std = float(np.std(pred))
    y_std = float(np.std(y))
    if pred_std <= 1e-9 or y_std <= 1e-9:
        return None
    return float(np.corrcoef(pred, y)[0, 1])


def r2_score(pred: np.ndarray, y: np.ndarray) -> float | None:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-9:
        return None
    num = float(np.sum((y - pred) ** 2))
    return float(1.0 - num / denom)


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= np.mean(weights)
    return weights.astype(np.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def smooth_sequence_labels(pred: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(pred) == 0:
        return pred.copy()
    radius = max(0, window // 2)
    smoothed = pred.copy()
    for idx in range(len(pred)):
        left = max(0, idx - radius)
        right = min(len(pred), idx + radius + 1)
        smoothed[idx] = int(np.bincount(pred[left:right]).argmax())
    return smoothed


def smooth_predictions_by_group(pred: np.ndarray, groups: np.ndarray, order: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pred.copy()
    out = pred.copy()
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        idx = idx[np.argsort(order[idx])]
        out[idx] = smooth_sequence_labels(pred[idx], window)
    return out


def summarize_confusion(conf: np.ndarray) -> dict:
    return {
        "matrix": conf.tolist(),
        "per_class_support": conf.sum(axis=1).astype(int).tolist(),
        "per_class_tp": np.diag(conf).astype(int).tolist(),
    }


def sequence_smoothness_loss(pred: torch.Tensor, target: torch.Tensor, trial_ids: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    same_trial = trial_ids[1:] == trial_ids[:-1]
    if not torch.any(same_trial):
        return pred.new_tensor(0.0)
    pred_delta = pred[1:] - pred[:-1]
    target_delta = target[1:] - target[:-1]
    return torch.mean((pred_delta[same_trial] - target_delta[same_trial]) ** 2)


def apply_channel_shift(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        return x
    out = x.clone()
    for idx in range(x.shape[0]):
        s = int(shift[idx].item())
        if s == 0:
            continue
        if s > 0:
            out[idx, :, :s] = x[idx, :, :1]
            out[idx, :, s:] = x[idx, :, :-s]
        else:
            out[idx, :, s:] = x[idx, :, -1:]
            out[idx, :, :s] = x[idx, :, -s:]
    return out


def _float_cfg(d: dict[str, Any], key: str, default: float) -> float:
    return float(d.get(key, default))


def _int_cfg(d: dict[str, Any], key: str, default: int) -> int:
    return int(d.get(key, default))


def get_train_augment_cfg(cfg: dict[str, Any], profile: str, modality: str | None = None) -> dict[str, float | int]:
    aug = cfg.get("augment", {})
    train_aug = aug.get("train", {})
    profile_cfg = dict(train_aug.get(profile, {}))
    if profile_cfg:
        if profile == "single_modality":
            noise_std = profile_cfg.get(
                "noise_std",
                profile_cfg.get("eeg_noise_std" if modality == "eeg" else "emg_noise_std", 0.0),
            )
            return {"noise_std": float(noise_std)}
        result: dict[str, Any] = {
            "drop_eeg_prob": _float_cfg(profile_cfg, "drop_eeg_prob", 0.0),
            "drop_emg_prob": _float_cfg(profile_cfg, "drop_emg_prob", 0.0),
            "eeg_noise_std": _float_cfg(profile_cfg, "eeg_noise_std", 0.0),
            "emg_noise_std": _float_cfg(profile_cfg, "emg_noise_std", 0.0),
            "temporal_jitter_samples": _int_cfg(profile_cfg, "temporal_jitter_samples", 0),
            "channel_mask_prob": _float_cfg(profile_cfg, "channel_mask_prob", 0.0),
            "curriculum_warmup_epochs": _int_cfg(profile_cfg, "curriculum_warmup_epochs", 0),
            "curriculum_peak_epoch": _int_cfg(profile_cfg, "curriculum_peak_epoch", 0),
            "aux_corruption_prob": _float_cfg(profile_cfg, "aux_corruption_prob", 1.0),
        }
        if "emg_noise_levels" in profile_cfg:
            result["emg_noise_levels"] = [float(v) for v in profile_cfg.get("emg_noise_levels", [])]
        if "corruption_mix" in profile_cfg:
            result["corruption_mix"] = {str(k): float(v) for k, v in dict(profile_cfg.get("corruption_mix", {})).items()}
        return result

    if profile == "single_modality":
        legacy_key = "train_eeg_noise_std" if modality == "eeg" else "train_emg_noise_std"
        return {"noise_std": float(aug.get(legacy_key, aug.get("train_single_noise_std", 0.0)))}

    return {
        "drop_eeg_prob": float(aug.get("train_fusion_drop_eeg_prob", 0.0)),
        "drop_emg_prob": float(aug.get("train_fusion_drop_emg_prob", 0.0)),
        "eeg_noise_std": float(aug.get("train_eeg_noise_std", 0.0)),
        "emg_noise_std": float(aug.get("train_emg_noise_std", 0.0)),
        "temporal_jitter_samples": int(aug.get("train_temporal_jitter_samples", 0)),
        "channel_mask_prob": float(aug.get("train_channel_mask_prob", 0.0)),
    }


def curriculum_scale(aug_cfg: dict[str, Any], epoch_index: int | None) -> float:
    if epoch_index is None:
        return 1.0
    warmup_epochs = int(aug_cfg.get("curriculum_warmup_epochs", 0))
    peak_epoch = int(aug_cfg.get("curriculum_peak_epoch", 0))
    if peak_epoch <= 0:
        return 1.0
    denom = max(1, peak_epoch - 1)
    scale = 0.25 + 0.75 * float(epoch_index) / float(denom)
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        scale = min(scale, 0.50)
    return float(np.clip(scale, 0.0, 1.0))


def _normalize_mix(mix: dict[str, float]) -> tuple[list[str], np.ndarray]:
    items = [(name, float(weight)) for name, weight in mix.items() if float(weight) > 0.0]
    if not items:
        return [], np.asarray([], dtype=np.float32)
    names = [name for name, _ in items]
    weights = np.asarray([weight for _, weight in items], dtype=np.float32)
    weights /= float(np.sum(weights))
    return names, weights


def _sample_level(levels: list[float] | np.ndarray | None, fallback: float, size: int, device: torch.device) -> torch.Tensor:
    if levels:
        values = torch.tensor([float(level) for level in levels], dtype=torch.float32, device=device)
        choice = torch.randint(0, len(values), (size,), device=device)
        return values[choice]
    return torch.full((size,), float(fallback), dtype=torch.float32, device=device)


def apply_training_modality_augmentation(
    x: torch.Tensor,
    modality: str,
    eeg_ch: int,
    emg_ch: int,
    cfg: dict,
    profile: str | None = None,
    epoch_index: int | None = None,
    active_prob: float | None = None,
) -> torch.Tensor:
    profile = profile or ("phase_fusion" if modality == "fusion" else "single_modality")
    aug = get_train_augment_cfg(cfg, profile=profile, modality=modality)
    out = x.clone()
    if modality == "fusion":
        eeg = out[:, :eeg_ch, :]
        emg = out[:, eeg_ch:eeg_ch + emg_ch, :]
        mix_names, mix_weights = _normalize_mix(dict(aug.get("corruption_mix", {})))
        corruption_prob = float(active_prob if active_prob is not None else curriculum_scale(aug, epoch_index))
        if mix_names:
            active = torch.rand(out.shape[0], device=out.device) < corruption_prob
            if torch.any(active):
                choice_idx = np.random.choice(len(mix_names), size=int(active.sum().item()), p=mix_weights)
                choice_names = [mix_names[idx] for idx in choice_idx]
                active_idx = torch.where(active)[0]
                emg_noise_levels = aug.get("emg_noise_levels", [])
                for corruption_name in mix_names:
                    local_positions = [pos for pos, name in enumerate(choice_names) if name == corruption_name]
                    if not local_positions:
                        continue
                    idx = active_idx[torch.tensor(local_positions, device=out.device, dtype=torch.long)]
                    if corruption_name == "emg_noise":
                        noise_scale = _sample_level(emg_noise_levels, float(aug.get("emg_noise_std", 0.0)), len(idx), out.device)
                        emg[idx] += torch.randn_like(emg[idx]) * noise_scale.view(-1, 1, 1)
                    elif corruption_name == "drop_eeg":
                        drop_mask = torch.rand(len(idx), device=out.device) < float(aug.get("drop_eeg_prob", 1.0))
                        if torch.any(drop_mask):
                            eeg[idx[drop_mask]] = 0.0
                    elif corruption_name == "drop_emg":
                        drop_mask = torch.rand(len(idx), device=out.device) < float(aug.get("drop_emg_prob", 1.0))
                        if torch.any(drop_mask):
                            emg[idx[drop_mask]] = 0.0
                    elif corruption_name == "channel_mask":
                        mask_prob = float(aug.get("channel_mask_prob", 0.0))
                        if mask_prob > 0:
                            eeg_mask = (torch.rand((len(idx), eeg.shape[1]), device=out.device) < mask_prob).unsqueeze(-1)
                            emg_mask = (torch.rand((len(idx), emg.shape[1]), device=out.device) < mask_prob).unsqueeze(-1)
                            eeg[idx] = eeg[idx].masked_fill(eeg_mask, 0.0)
                            emg[idx] = emg[idx].masked_fill(emg_mask, 0.0)
                    elif corruption_name == "temporal_jitter":
                        max_shift = int(aug.get("temporal_jitter_samples", 0))
                        if max_shift > 0:
                            shift = torch.randint(-max_shift, max_shift + 1, (len(idx),), device=out.device)
                            eeg[idx] = apply_channel_shift(eeg[idx], shift)
        else:
            drop_eeg = torch.rand(out.shape[0], device=out.device) < float(aug.get("drop_eeg_prob", 0.0))
            drop_emg = torch.rand(out.shape[0], device=out.device) < float(aug.get("drop_emg_prob", 0.0))
            both = drop_eeg & drop_emg
            if torch.any(both):
                flip = torch.rand(out.shape[0], device=out.device) < 0.5
                drop_eeg = torch.where(both & flip, torch.zeros_like(drop_eeg), drop_eeg)
                drop_emg = torch.where(both & ~flip, torch.zeros_like(drop_emg), drop_emg)
            if torch.any(drop_eeg):
                eeg[drop_eeg] = 0.0
            if torch.any(drop_emg):
                emg[drop_emg] = 0.0

            eeg_noise = float(aug.get("eeg_noise_std", 0.0))
            emg_noise = float(aug.get("emg_noise_std", 0.0))
            if eeg_noise > 0:
                eeg += torch.randn_like(eeg) * eeg_noise
            if emg_noise > 0:
                emg += torch.randn_like(emg) * emg_noise

            mask_prob = float(aug.get("channel_mask_prob", 0.0))
            if mask_prob > 0:
                eeg_mask = (torch.rand(eeg.shape[:2], device=out.device) < mask_prob).unsqueeze(-1)
                emg_mask = (torch.rand(emg.shape[:2], device=out.device) < mask_prob).unsqueeze(-1)
                eeg = eeg.masked_fill(eeg_mask, 0.0)
                emg = emg.masked_fill(emg_mask, 0.0)

            max_shift = int(aug.get("temporal_jitter_samples", 0))
            if max_shift > 0:
                shift = torch.randint(-max_shift, max_shift + 1, (out.shape[0],), device=out.device)
                eeg = apply_channel_shift(eeg, shift)

        out[:, :eeg_ch, :] = eeg
        out[:, eeg_ch:eeg_ch + emg_ch, :] = emg
        return out

    noise_std = float(aug.get("noise_std", 0.0))
    if noise_std > 0:
        out += torch.randn_like(out) * noise_std
    return out


def kl_consistency_loss(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor) -> torch.Tensor:
    teacher = torch.softmax(clean_logits.detach(), dim=1)
    return F.kl_div(F.log_softmax(corrupted_logits, dim=1), teacher, reduction="batchmean")


def regression_consistency_loss(clean_pred: torch.Tensor, corrupted_pred: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(corrupted_pred, clean_pred.detach())


def phase_selection_score(
    clean_macro_f1: float,
    emg_noise_macro_f1: float,
    drop_eeg_macro_f1: float,
    cfg: dict[str, Any],
) -> float:
    weights = cfg.get("model", {}).get("phase", {}).get("selection_weights", {})
    w_clean = float(weights.get("clean", 0.60))
    w_noise = float(weights.get("emg_noise", 0.25))
    w_drop_eeg = float(weights.get("drop_eeg", 0.15))
    return float(w_clean * clean_macro_f1 + w_noise * emg_noise_macro_f1 + w_drop_eeg * drop_eeg_macro_f1)


def torque_error_by_range(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    q1, q2 = np.quantile(y_true, [1 / 3, 2 / 3])
    bins = {
        "low": y_true <= q1,
        "mid": (y_true > q1) & (y_true <= q2),
        "high": y_true > q2,
    }
    out: dict[str, dict[str, float | int | None]] = {}
    for name, mask in bins.items():
        if not np.any(mask):
            out[name] = {"rmse": None, "mae": None, "count": 0}
            continue
        out[name] = {
            "rmse": float(rmse(y_pred[mask], y_true[mask])),
            "mae": float(mae(y_pred[mask], y_true[mask])),
            "count": int(mask.sum()),
        }
    return out


def torque_selection_score(
    clean_rmse: float,
    emg_noise_rmse: float,
    drop_eeg_rmse: float,
    high_target_rmse: float,
    cfg: dict[str, Any],
) -> float:
    weights = cfg.get("model", {}).get("torque", {}).get("selection_weights", {})
    w_clean = float(weights.get("clean", 0.55))
    w_noise = float(weights.get("emg_noise", 0.20))
    w_drop_eeg = float(weights.get("drop_eeg", 0.15))
    w_high = float(weights.get("high_target", 0.10))
    return float(w_clean * clean_rmse + w_noise * emg_noise_rmse + w_drop_eeg * drop_eeg_rmse + w_high * high_target_rmse)


def _shift_window_np(x: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return x.copy()
    out = np.empty_like(x)
    if shift > 0:
        out[..., :shift] = x[..., :1]
        out[..., shift:] = x[..., :-shift]
    else:
        out[..., shift:] = x[..., -1:]
        out[..., :shift] = x[..., -shift:]
    return out


def apply_eval_stress_np(
    x: np.ndarray,
    cfg: dict[str, Any],
    modality: str,
    stress_case: str,
    seed: int,
) -> np.ndarray:
    if stress_case == "clean":
        return x.copy()

    rng = np.random.default_rng(seed)
    out = x.copy()
    eeg_ch, emg_ch = get_channel_counts(cfg)
    noise_mult = float(cfg.get("eval", {}).get("stress", {}).get("noise_multipliers", [1.5])[-1])
    shift_samples = int(cfg.get("eval", {}).get("stress", {}).get("temporal_shift_samples", [12])[-1])

    if modality == "fusion":
        eeg = out[:, :eeg_ch, :]
        emg = out[:, eeg_ch:eeg_ch + emg_ch, :]
        if stress_case == "drop_eeg":
            eeg[:] = 0.0
        elif stress_case == "drop_emg":
            emg[:] = 0.0
        elif stress_case == "emg_noise":
            emg += rng.normal(0.0, 0.15 * max(0.0, noise_mult - 1.0), size=emg.shape).astype(np.float32)
        elif stress_case == "temporal_shift":
            out[:, :eeg_ch, :] = _shift_window_np(eeg, shift_samples)
        return out

    if modality == "emg":
        if stress_case == "emg_noise":
            out += rng.normal(0.0, 0.15 * max(0.0, noise_mult - 1.0), size=out.shape).astype(np.float32)
        elif stress_case == "temporal_shift":
            out = _shift_window_np(out, shift_samples)
        return out

    if modality == "eeg" and stress_case == "temporal_shift":
        return _shift_window_np(out, shift_samples)
    return out


def compute_torque_sample_weights(
    y: np.ndarray,
    alpha: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    scale = np.max(np.abs(y)) + 1e-6
    magnitude = np.abs(y) / scale
    weights = 1.0 + alpha * np.power(magnitude, gamma)
    weights /= float(np.mean(weights))
    return weights.astype(np.float32)


def compute_torque_range_weights(
    y: np.ndarray,
    range_weights: dict[str, float] | None = None,
) -> np.ndarray:
    if range_weights is None:
        range_weights = {"low": 1.0, "mid": 1.0, "high": 1.0}
    y = np.asarray(y, dtype=np.float32)
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    weights = np.full(len(y), float(range_weights.get("mid", 1.0)), dtype=np.float32)
    weights[y <= q1] = float(range_weights.get("low", 1.0))
    weights[y > q2] = float(range_weights.get("high", 1.0))
    weights /= float(np.mean(weights))
    return weights.astype(np.float32)


def weighted_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_type == "mse":
        loss = (pred - target) ** 2
    else:
        loss = F.huber_loss(pred, target, reduction="none", delta=huber_delta)
    sample_weight = sample_weight.float()
    return torch.mean(loss * sample_weight)
