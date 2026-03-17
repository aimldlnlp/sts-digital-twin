from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import torch

from src.common import load_yaml, ensure_dir
from src.features.dataset import load_trials, StandardScaler1D
from src.models.nets import PhaseClassifier
from src.models.train_utils import get_device


MODS = ["eeg", "emg", "fusion"]
PANEL = {"eeg": "A", "emg": "B", "fusion": "C"}
PHASE_NAMES = ["Preparation", "Momentum", "Extension", "Stabilization"]
PHASE_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
MOD_COLORS = {"eeg": "#1f77b4", "emg": "#2ca02c", "fusion": "#d62728"}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    })


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def _input_by_modality(eeg: np.ndarray, emg_env: np.ndarray, modality: str) -> np.ndarray:
    if modality == "eeg":
        return eeg
    if modality == "emg":
        return emg_env
    return np.concatenate([eeg, emg_env], axis=1)


def _interpolate_probs(probs: np.ndarray) -> np.ndarray:
    # probs: (T, C) with NaN at edges
    out = probs.copy()
    T, C = out.shape
    idx = np.arange(T)
    valid = ~np.isnan(out[:, 0])
    if valid.sum() == 0:
        out[:] = 1.0 / C
        return out
    for c in range(C):
        out[:, c] = np.interp(idx, idx[valid], out[valid, c])
    out = np.clip(out, 1e-8, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def predict_phase_track(
    run_dir: Path,
    cfg: Dict[str, Any],
    eeg: np.ndarray,
    emg_env: np.ndarray,
    modality: str,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler_path = run_dir / "artifacts" / f"scaler_{modality}.json"
    model_path = run_dir / "artifacts" / f"phase_model_{modality}.pt"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    inp = _input_by_modality(eeg, emg_env, modality)
    T = inp.shape[0]
    L = int(cfg["train"]["segment_len"])
    half = L // 2
    n_classes = 4

    centers = np.arange(half, T - half + 1, dtype=np.int64)
    segs = np.stack([inp[c - half:c + half].T for c in centers], axis=0).astype(np.float32)

    scaler = StandardScaler1D.from_dict(json.loads(scaler_path.read_text()))
    x = scaler.transform(segs)

    device = get_device()
    model = PhaseClassifier(in_ch=x.shape[1], n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logits_all = []
    bs = 512
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs]).to(device)
            logits_all.append(model(xb).cpu().numpy())
    logits = np.concatenate(logits_all, axis=0)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)

    probs = np.full((T, n_classes), np.nan, dtype=np.float32)
    probs[centers] = p.astype(np.float32)
    probs = _interpolate_probs(probs)
    pred = probs.argmax(axis=1).astype(np.int64)
    return probs, pred


def load_macro_f1(run_dir: Path, modality: str) -> float:
    p = run_dir / "artifacts" / f"phase_metrics_{modality}.json"
    if not p.exists():
        return float("nan")
    d = json.loads(p.read_text())
    return float(d.get("test_macro_f1", float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--trial_index", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame_step", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality (lower is better, typical 16-23).")
    ap.add_argument("--preset", type=str, default="slow",
                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"])
    ap.add_argument("--max_frames", type=int, default=None, help="Optional cap for quick debug.")
    ap.add_argument("--out_path", type=str, default=None)
    args = ap.parse_args()

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("FFmpeg writer is not available. Please install `ffmpeg` first.")

    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    apply_style()

    trials = load_trials(run_dir)
    tp = trials[min(args.trial_index, len(trials) - 1)]
    d = np.load(tp, allow_pickle=True)

    t = d["t"].astype(np.float32)
    phase = d["phase"].astype(np.int64)
    eeg = d["eeg"].astype(np.float32)
    emg_env = d["emg_env"].astype(np.float32)

    sig_eeg = _zscore(eeg[:, 0])
    sig_emg = _zscore(emg_env.mean(axis=1))

    pred_map: Dict[str, np.ndarray] = {}
    prob_map: Dict[str, np.ndarray] = {}
    f1_map: Dict[str, float] = {}
    for m in MODS:
        probs, pred = predict_phase_track(run_dir, cfg, eeg, emg_env, m)
        pred_map[m] = pred
        prob_map[m] = probs
        f1_map[m] = load_macro_f1(run_dir, m)

    # Figure layout: 2 rows x 3 columns
    fig = plt.figure(figsize=(15.5, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1.2])
    ax_sig = {m: fig.add_subplot(gs[0, i]) for i, m in enumerate(MODS)}
    ax_phase = {m: fig.add_subplot(gs[1, i]) for i, m in enumerate(MODS)}

    cmap_phase = ListedColormap(PHASE_COLORS)
    sig_prog = {}
    sig_cursor = {}
    phase_cursor = {}
    info_text = {}

    for m in MODS:
        ax = ax_sig[m]
        if m == "eeg":
            ax.plot(t, sig_eeg, color=MOD_COLORS[m], alpha=0.22, linewidth=1.0)
            ln, = ax.plot([], [], color=MOD_COLORS[m], linewidth=1.7)
            ymin, ymax = sig_eeg.min(), sig_eeg.max()
            ax.set_ylabel("EEG z-score")
        elif m == "emg":
            ax.plot(t, sig_emg, color=MOD_COLORS[m], alpha=0.22, linewidth=1.0)
            ln, = ax.plot([], [], color=MOD_COLORS[m], linewidth=1.7)
            ymin, ymax = sig_emg.min(), sig_emg.max()
            ax.set_ylabel("EMG z-score")
        else:
            ax.plot(t, sig_eeg, color="#1f77b4", alpha=0.18, linewidth=1.0)
            ax.plot(t, sig_emg, color="#2ca02c", alpha=0.18, linewidth=1.0)
            ln, = ax.plot([], [], color=MOD_COLORS[m], linewidth=1.8)
            mix = 0.5 * (sig_eeg + sig_emg)
            ymin, ymax = mix.min(), mix.max()
            ax.set_ylabel("Fusion signal")

        pad = 0.2 * float(ymax - ymin + 1e-6)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(float(ymin - pad), float(ymax + pad))
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{PANEL[m]}. {m.upper()} model  |  test Macro-F1={f1_map[m]:.3f}")
        cur = ax.axvline(t[0], color="#333333", linestyle="--", linewidth=1.0)
        sig_prog[m] = ln
        sig_cursor[m] = cur

        axp = ax_phase[m]
        arr = np.vstack([phase, pred_map[m]]).astype(np.float32)
        axp.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap_phase, vmin=0, vmax=3,
                   extent=[float(t[0]), float(t[-1]), 0.0, 2.0], origin="lower")
        axp.set_yticks([0.5, 1.5])
        axp.set_yticklabels(["GT", "Pred"])
        axp.set_xlim(float(t[0]), float(t[-1]))
        axp.set_xlabel("Time (s)")
        phase_cursor[m] = axp.axvline(t[0], color="#111111", linestyle="--", linewidth=1.0)
        legend = " | ".join([f"{i}:{name}" for i, name in enumerate(PHASE_NAMES)])
        axp.text(0.01, -0.48, legend, transform=axp.transAxes, fontsize=8.8)
        info_text[m] = axp.text(0.01, 1.12, "", transform=axp.transAxes, fontsize=9.2)

    supt = fig.suptitle("", fontsize=12.5)

    frame_indices = list(range(0, len(t), max(1, int(args.frame_step))))
    if args.max_frames is not None:
        frame_indices = frame_indices[:max(1, int(args.max_frames))]

    def init():
        for m in MODS:
            sig_prog[m].set_data([], [])
            sig_cursor[m].set_xdata([t[0], t[0]])
            phase_cursor[m].set_xdata([t[0], t[0]])
            info_text[m].set_text("")
        return [sig_prog[m] for m in MODS]

    def update(fi: int):
        ti = float(t[fi])
        for m in MODS:
            if m == "eeg":
                y = sig_eeg
            elif m == "emg":
                y = sig_emg
            else:
                y = 0.5 * (sig_eeg + sig_emg)
            sig_prog[m].set_data(t[:fi + 1], y[:fi + 1])
            sig_cursor[m].set_xdata([ti, ti])
            phase_cursor[m].set_xdata([ti, ti])

            pred_cls = int(pred_map[m][fi])
            gt_cls = int(phase[fi])
            conf = float(prob_map[m][fi, pred_cls])
            info_text[m].set_text(
                f"t={ti:.2f}s | GT={gt_cls} ({PHASE_NAMES[gt_cls]}) | "
                f"Pred={pred_cls} ({PHASE_NAMES[pred_cls]}) | p={conf:.2f}"
            )

        supt.set_text(f"Phase Ablation Video  |  trial={args.trial_index:03d}")
        return [sig_prog[m] for m in MODS]

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frame_indices,
        interval=1000.0 / max(1, int(args.fps)),
        blit=False,
    )

    if args.out_path is None:
        out_dir = ensure_dir(run_dir / "videos")
        out_path = out_dir / f"trial_{args.trial_index:03d}_ablation_phase.mp4"
    else:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = animation.FFMpegWriter(
        fps=max(1, int(args.fps)),
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", str(int(args.crf)), "-preset", args.preset],
    )
    ani.save(str(out_path), writer=writer, dpi=max(80, int(args.dpi)))
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
