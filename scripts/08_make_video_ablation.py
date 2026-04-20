from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.common import ensure_dir, load_yaml
from src.features.dataset import StandardScaler1D, load_trials
from src.models.nets import load_model_checkpoint
from src.models.train_utils import get_device, smooth_sequence_labels
from src.viz import MODALITY_COLORS, PHASE_COLORS, PHASE_NAMES, add_corner_label, apply_scientific_luxe_style, short_phase_name, style_axes


MODS = ["eeg", "emg", "fusion"]


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


def predict_phase_track(run_dir: Path, cfg: dict[str, Any], eeg: np.ndarray, emg_env: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
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


def load_macro_f1(run_dir: Path, modality: str) -> float:
    path = run_dir / "artifacts" / f"phase_metrics_{modality}.json"
    if not path.exists():
        return float("nan")
    return float(json.loads(path.read_text()).get("test_macro_f1", float("nan")))


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
    subprocess.check_call([
        "ffmpeg", "-y", "-i", str(video_path), "-vf", f"{vf},palettegen=stats_mode=diff", str(palette_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(palette_path), "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", str(gif_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    palette_path.unlink(missing_ok=True)
    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--trial_index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="slow", choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"])
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("FFmpeg writer is not available. Please install `ffmpeg` first.")

    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    style_info = apply_scientific_luxe_style("video")

    trials = load_trials(run_dir)
    trial_path = trials[min(args.trial_index, len(trials) - 1)]
    data = np.load(trial_path, allow_pickle=True)

    t = data["t"].astype(np.float32)
    phase = data["phase"].astype(np.int64)
    eeg = data["eeg"].astype(np.float32)
    emg_env = data["emg_env"].astype(np.float32)
    sig_eeg = zscore(eeg[:, 0])
    sig_emg = zscore(emg_env.mean(axis=1))
    spans = phase_segments(t, phase)

    pred_map: dict[str, np.ndarray] = {}
    prob_map: dict[str, np.ndarray] = {}
    f1_map: dict[str, float] = {}
    for modality in MODS:
        probs, pred = predict_phase_track(run_dir, cfg, eeg, emg_env, modality)
        pred_map[modality] = pred
        prob_map[modality] = probs
        f1_map[modality] = load_macro_f1(run_dir, modality)

    fig = plt.figure(figsize=(15.6, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.1])
    ax_sig = {modality: fig.add_subplot(gs[0, idx]) for idx, modality in enumerate(MODS)}
    ax_phase = {modality: fig.add_subplot(gs[1, idx]) for idx, modality in enumerate(MODS)}

    cmap = ListedColormap(PHASE_COLORS)
    sig_prog: dict[str, Any] = {}
    conf_line: dict[str, Any] = {}
    sig_cursor: dict[str, Any] = {}
    phase_cursor: dict[str, Any] = {}
    info_text: dict[str, Any] = {}

    for modality in MODS:
        ax = ax_sig[modality]
        style_axes(ax)
        for left, right, phase_idx in spans:
            ax.axvspan(left, right, color=PHASE_COLORS[phase_idx], alpha=0.04, linewidth=0, zorder=0)
        if modality == "eeg":
            base_signal = sig_eeg
            accent = MODALITY_COLORS["eeg"]
            accent_text = "EEG"
        elif modality == "emg":
            base_signal = sig_emg
            accent = MODALITY_COLORS["emg"]
            accent_text = "EMG"
        else:
            base_signal = 0.5 * (sig_eeg + sig_emg)
            accent = MODALITY_COLORS["fusion"]
            accent_text = "FUSION"
        ax.plot(t, base_signal, color=accent, alpha=0.18, linewidth=1.2)
        ax.fill_between(t, base_signal, 0.0, color=accent, alpha=0.07)
        sig_prog[modality] = ax.plot([], [], color=accent, linewidth=1.8)[0]
        pad = 0.25 * float(base_signal.max() - base_signal.min() + 1e-6)
        conf = prob_map[modality].max(axis=1)
        ymin = float(base_signal.min() - pad)
        ymax = float(base_signal.max() + pad)
        conf_trace = ymin + 0.12 * (ymax - ymin) * conf
        ax.fill_between(t, ymin, conf_trace, color=accent, alpha=0.05, linewidth=0)
        conf_line[modality] = ax.plot([], [], color=accent, linewidth=1.0, alpha=0.55)[0]
        sig_cursor[modality] = ax.axvline(float(t[0]), color="#312925", linestyle="--", linewidth=1.0)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(ymin, ymax)
        ax.spines["left"].set_color(accent)
        ax.spines["left"].set_linewidth(1.0)
        add_corner_label(ax, accent_text)
        ax.text(0.98, 0.98, f"F1 {f1_map[modality]:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.9, color="#000000")
        strip = inset_axes(ax, width="100%", height="9%", loc="upper center", borderpad=0.55)
        strip.imshow(
            phase[None, :],
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            extent=[float(t[0]), float(t[-1]), 0.0, 1.0],
            origin="lower",
            vmin=0,
            vmax=3,
        )
        style_axes(strip, hide_ticks=True)
        strip.axvline(float(t[0]), color="#181412", linestyle="--", linewidth=0.9)
        ax._phase_strip = strip
        ax._conf_trace = conf_trace

        axp = ax_phase[modality]
        style_axes(axp, hide_ticks=True)
        arr = np.vstack([phase, pred_map[modality]]).astype(np.float32)
        axp.imshow(
            arr,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            extent=[float(t[0]), float(t[-1]), 0.0, 2.0],
            origin="lower",
            vmin=0,
            vmax=3,
        )
        axp.set_xlim(float(t[0]), float(t[-1]))
        axp.set_yticks([0.5, 1.5])
        axp.set_yticklabels(["GT", "PR"])
        axp.set_xlabel("time (s)")
        add_corner_label(axp, "GT / Pred")
        phase_cursor[modality] = axp.axvline(float(t[0]), color="#181412", linestyle="--", linewidth=1.0)
        info_text[modality] = axp.text(0.98, 1.08, "", transform=axp.transAxes, fontsize=8.1, color="#000000", ha="right", va="bottom")
        for x0, x1, phase_idx in spans:
            xmid = 0.5 * (x0 + x1)
            axp.text(xmid, 1.78, short_phase_name(phase_idx), ha="center", va="center", fontsize=7.2, color="#000000")

    title = fig.suptitle(f"phase ablation  |  trial {args.trial_index:03d}", fontsize=11.2, x=0.51, y=0.992)

    frame_indices = list(range(0, len(t), max(1, int(args.frame_step))))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(1, int(args.max_frames))]

    def init():
        for modality in MODS:
            sig_prog[modality].set_data([], [])
            conf_line[modality].set_data([], [])
            sig_cursor[modality].set_xdata([float(t[0]), float(t[0])])
            phase_cursor[modality].set_xdata([float(t[0]), float(t[0])])
            info_text[modality].set_text("")
            ax_sig[modality]._phase_strip.lines[0].set_xdata([float(t[0]), float(t[0])])
        return [sig_prog[modality] for modality in MODS] + [conf_line[modality] for modality in MODS]

    def update(frame_idx: int):
        time_now = float(t[frame_idx])
        for modality in MODS:
            if modality == "eeg":
                y = sig_eeg
            elif modality == "emg":
                y = sig_emg
            else:
                y = 0.5 * (sig_eeg + sig_emg)
            sig_prog[modality].set_data(t[: frame_idx + 1], y[: frame_idx + 1])
            conf_line[modality].set_data(t[: frame_idx + 1], ax_sig[modality]._conf_trace[: frame_idx + 1])
            sig_cursor[modality].set_xdata([time_now, time_now])
            ax_sig[modality]._phase_strip.lines[0].set_xdata([time_now, time_now])
            phase_cursor[modality].set_xdata([time_now, time_now])
            pred_cls = int(pred_map[modality][frame_idx])
            conf = float(prob_map[modality][frame_idx, pred_cls])
            info_text[modality].set_text(f"{short_phase_name(pred_cls)}  p={conf:.2f}")
        title.set_text(f"phase ablation  |  trial {args.trial_index:03d}  |  {time_now:0.2f}s")
        return [sig_prog[modality] for modality in MODS] + [conf_line[modality] for modality in MODS]

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
    gif_path = export_gif(out_path, args.fps)
    print(f"Using font: {style_info['font_primary']}")
    print(f"Saved: {out_path}")
    print(f"Saved: {gif_path}")


if __name__ == "__main__":
    main()
