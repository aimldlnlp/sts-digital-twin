from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.features.dataset import StandardScaler1D, StandardScalerTarget
from src.models.nets import load_model_checkpoint
from src.models.train_utils import get_device, smooth_sequence_labels


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def input_by_modality(eeg: np.ndarray, emg_env: np.ndarray, modality: str) -> np.ndarray:
    if modality == "eeg":
        return eeg
    if modality == "emg":
        return emg_env
    return np.concatenate([eeg, emg_env], axis=1)


def interpolate_probs(probs: np.ndarray) -> np.ndarray:
    output = probs.copy()
    t_idx = np.arange(output.shape[0])
    valid = ~np.isnan(output[:, 0])
    if valid.sum() == 0:
        output[:] = 1.0 / output.shape[1]
        return output
    for c in range(output.shape[1]):
        output[:, c] = np.interp(t_idx, t_idx[valid], output[valid, c])
    output = np.clip(output, 1e-8, None)
    output /= output.sum(axis=1, keepdims=True)
    return output


def phase_segments(t: np.ndarray, phase: np.ndarray) -> list[tuple[float, float, int]]:
    starts = [0]
    starts.extend(list(np.where(np.diff(phase) != 0)[0] + 1))
    starts.append(len(phase))
    spans: list[tuple[float, float, int]] = []
    for i0, i1 in zip(starts[:-1], starts[1:]):
        spans.append((float(t[i0]), float(t[i1 - 1]), int(phase[i0])))
    return spans


def export_gif(video_path: Path, fps: int) -> Path:
    gif_path = video_path.with_suffix(".gif")
    palette_path = gif_path.with_name(f"{gif_path.stem}_palette.png")
    vf = f"fps={max(1, min(20, int(fps)))},scale=1200:-1:flags=lanczos"
    subprocess.check_call(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"{vf},palettegen=stats_mode=diff", str(palette_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(palette_path),
            "-lavfi",
            f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
            str(gif_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    palette_path.unlink(missing_ok=True)
    return gif_path


def apply_trial_stress(
    eeg: np.ndarray,
    emg_env: np.ndarray,
    cfg: dict,
    stress_case: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    eeg_out = eeg.copy()
    emg_out = emg_env.copy()
    if stress_case == "clean":
        return eeg_out, emg_out

    rng = np.random.default_rng(seed)
    noise_mult = float(cfg.get("eval", {}).get("stress", {}).get("noise_multipliers", [1.5])[-1])
    shift_samples = int(cfg.get("eval", {}).get("stress", {}).get("temporal_shift_samples", [12])[-1])

    if stress_case == "drop_eeg":
        eeg_out[:] = 0.0
    elif stress_case == "drop_emg":
        emg_out[:] = 0.0
    elif stress_case == "emg_noise":
        emg_out += rng.normal(0.0, 0.15 * max(0.0, noise_mult - 1.0), size=emg_out.shape).astype(np.float32)
    elif stress_case == "temporal_shift":
        out = np.empty_like(eeg_out)
        if shift_samples > 0:
            out[:shift_samples] = eeg_out[:1]
            out[shift_samples:] = eeg_out[:-shift_samples]
        elif shift_samples < 0:
            out[shift_samples:] = eeg_out[-1:]
            out[:shift_samples] = eeg_out[-shift_samples:]
        else:
            out = eeg_out.copy()
        eeg_out = out
    return eeg_out, emg_out


def predict_phase_track(run_dir: Path, cfg: dict, eeg: np.ndarray, emg_env: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
    scaler_path = run_dir / "artifacts" / f"scaler_{modality}.json"
    model_path = run_dir / "artifacts" / f"phase_model_{modality}.pt"
    inp = input_by_modality(eeg, emg_env, modality)
    total = inp.shape[0]
    seg_len = int(cfg["train"]["segment_len"])
    half = seg_len // 2
    centers = np.arange(half, total - half + 1, dtype=np.int64)
    segments = np.stack([inp[c - half:c + half].T for c in centers], axis=0).astype(np.float32)

    scaler = StandardScaler1D.from_dict(json.loads(scaler_path.read_text()))
    xs = scaler.transform(segments)
    device = get_device()
    model = load_model_checkpoint(model_path, cfg, task="phase", modality=modality, device=device)
    model.eval()

    logits = []
    with torch.no_grad():
        for offset in range(0, len(xs), 512):
            xb = torch.from_numpy(xs[offset:offset + 512]).to(device)
            logits.append(model(xb).cpu().numpy())
    logits_arr = np.concatenate(logits, axis=0)
    logits_arr -= logits_arr.max(axis=1, keepdims=True)
    probs = np.exp(logits_arr)
    probs /= probs.sum(axis=1, keepdims=True)

    track = np.full((total, 4), np.nan, dtype=np.float32)
    track[centers] = probs.astype(np.float32)
    track = interpolate_probs(track)
    pred = track.argmax(axis=1).astype(np.int64)
    window = int(cfg.get("eval", {}).get("phase_smoothing_window", 1))
    pred = smooth_sequence_labels(pred, window)
    return track, pred


def predict_torque_segments(run_dir: Path, cfg: dict, eeg: np.ndarray, emg_env: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
    inp = input_by_modality(eeg, emg_env, modality)
    seg_len = int(cfg["train"]["segment_len"])
    stride = int(cfg["train"]["segment_stride"])
    starts = list(range(0, inp.shape[0] - seg_len + 1, stride))
    if not starts:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    segments = np.stack([inp[start:start + seg_len].T for start in starts], axis=0).astype(np.float32)
    centers = np.array([start + seg_len // 2 for start in starts], dtype=np.int64)

    scaler_path = run_dir / "artifacts" / f"scaler_{modality}_torque.json"
    model_path = run_dir / "artifacts" / f"torque_model_{modality}.pt"
    if not (scaler_path.exists() and model_path.exists()):
        return centers.astype(np.float32), np.full(len(centers), np.nan, dtype=np.float32)

    scaler = StandardScaler1D.from_dict(json.loads(scaler_path.read_text()))
    target_scaler_path = run_dir / "artifacts" / f"target_scaler_{modality}_torque.json"
    target_scaler = StandardScalerTarget.from_dict(json.loads(target_scaler_path.read_text())) if target_scaler_path.exists() else None
    xs = scaler.transform(segments)
    device = get_device()
    model = load_model_checkpoint(model_path, cfg, task="torque", modality=modality, device=device)
    model.eval()

    preds = []
    with torch.no_grad():
        for offset in range(0, len(xs), 512):
            xb = torch.from_numpy(xs[offset:offset + 512]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds).astype(np.float32)
    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred).astype(np.float32)
    return centers.astype(np.float32), pred.astype(np.float32)
