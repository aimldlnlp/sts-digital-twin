from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.common import ensure_dir, load_yaml
from src.features.dataset import load_trials
from src.viz import MODALITY_COLORS, PHASE_COLORS, add_corner_label, apply_scientific_luxe_style, style_axes
from src.viz.video_support import export_gif, phase_segments, predict_torque_segments


RANGE_COLORS = {
    "low": MODALITY_COLORS["accent"],
    "mid": "#b6845b",
    "high": MODALITY_COLORS["fusion"],
}


def compute_segment_target(signal: np.ndarray, seg_len: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    starts = list(range(0, len(signal) - seg_len + 1, stride))
    centers = np.array([start + seg_len // 2 for start in starts], dtype=np.int64)
    values = np.array([float(np.mean(signal[start:start + seg_len])) for start in starts], dtype=np.float32)
    return centers, values


def range_labels(values: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    out = np.empty(len(values), dtype=object)
    out[values <= q1] = "low"
    out[(values > q1) & (values <= q2)] = "mid"
    out[values > q2] = "high"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--trial_index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="slow")
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
    tau_exo = data["tau_exo"].astype(np.float32)
    spans = phase_segments(t, phase)

    centers, pred_tau = predict_torque_segments(run_dir, cfg, eeg, emg_env, "fusion")
    seg_len = int(cfg["train"]["segment_len"])
    stride = int(cfg["train"]["segment_stride"])
    target_centers, target_tau = compute_segment_target(tau_exo, seg_len=seg_len, stride=stride)
    if len(centers) != len(target_centers):
        n = min(len(centers), len(target_centers))
        centers = centers[:n]
        pred_tau = pred_tau[:n]
        target_centers = target_centers[:n]
        target_tau = target_tau[:n]
    center_t = t[centers.astype(np.int64)]
    abs_err = np.abs(pred_tau - target_tau)
    labels = range_labels(target_tau)
    colors = [RANGE_COLORS[str(label)] for label in labels]

    fig = plt.figure(figsize=(14.0, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0])
    ax_tau = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[1, 0])

    style_axes(ax_tau)
    for left, right, phase_idx in spans:
        ax_tau.axvspan(left, right, color=PHASE_COLORS[phase_idx], alpha=0.04, linewidth=0)
    ax_tau.plot(t, tau_exo, color="#8f8477", linewidth=1.2, alpha=0.8)
    obs_prog = ax_tau.plot([], [], color="#221c18", linewidth=1.8)[0]
    pred_prog = ax_tau.plot([], [], color=MODALITY_COLORS["fusion"], linewidth=2.1)[0]
    range_scatter = ax_tau.scatter(center_t, target_tau, s=60, c=colors, alpha=0.35, edgecolors="#ffffff", linewidths=0.45)
    cursor = ax_tau.axvline(float(t[0]), color="#2d2723", linestyle="--", linewidth=1.0)
    active_point = ax_tau.scatter([], [], s=170, facecolors="none", edgecolors="#1d1a17", linewidths=1.25, zorder=5)
    ax_tau.set_xlim(float(t[0]), float(t[-1]))
    pad = float(0.10 * (tau_exo.max() - tau_exo.min() + 1e-6))
    ax_tau.set_ylim(float(tau_exo.min() - pad), float(tau_exo.max() + pad))
    ax_tau.set_ylabel("torque")
    add_corner_label(ax_tau, "range-aware torque")
    meta_text = ax_tau.text(0.985, 0.97, "", transform=ax_tau.transAxes, ha="right", va="top", fontsize=8.1, color="#000000")

    style_axes(ax_err)
    ax_err.scatter(target_tau, abs_err, s=42, c=colors, alpha=0.42, edgecolors="#ffffff", linewidths=0.4)
    ax_err.set_xlabel("target segment torque")
    ax_err.set_ylabel("|error|")
    add_corner_label(ax_err, "error concentration")
    active_err = ax_err.scatter([], [], s=170, facecolors="none", edgecolors="#1d1a17", linewidths=1.25, zorder=5)

    title = fig.suptitle(f"torque error range  |  trial {args.trial_index:03d}", fontsize=11.6, x=0.51, y=0.995)

    frame_indices = list(range(0, len(t), max(1, int(args.frame_step))))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(1, int(args.max_frames))]

    def update(frame_idx: int):
        time_now = float(t[frame_idx])
        obs_prog.set_data(t[: frame_idx + 1], tau_exo[: frame_idx + 1])
        cursor.set_xdata([time_now, time_now])
        mask = center_t <= time_now
        pred_prog.set_data(center_t[mask], pred_tau[mask])
        if np.any(mask):
            last = int(np.where(mask)[0][-1])
            active_point.set_offsets(np.array([[center_t[last], pred_tau[last]]]))
            active_err.set_offsets(np.array([[target_tau[last], abs_err[last]]]))
            meta_text.set_text(f"{labels[last]}  |error| {abs_err[last]:.3f}")
        title.set_text(f"torque error range  |  trial {args.trial_index:03d}  |  {time_now:0.2f}s")
        return [obs_prog, pred_prog, cursor, active_point, active_err]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000.0 / max(1, int(args.fps)),
        blit=False,
    )

    if args.out_path is None:
        out_dir = ensure_dir(run_dir / "videos")
        out_path = out_dir / f"trial_{args.trial_index:03d}_torque_error_range.mp4"
    else:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = animation.FFMpegWriter(
        fps=max(1, int(args.fps)),
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", str(int(args.crf)), "-preset", str(args.preset)],
    )
    ani.save(str(out_path), writer=writer, dpi=max(80, int(args.dpi)))
    plt.close(fig)
    gif_path = export_gif(out_path, args.fps)
    print(f"Using font: {style_info['font_primary']}")
    print(f"Saved: {out_path}")
    print(f"Saved: {gif_path}")


if __name__ == "__main__":
    main()
