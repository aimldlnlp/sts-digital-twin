from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.common import ensure_dir, load_yaml
from src.features.dataset import load_trials
from src.viz import MODALITY_COLORS, PHASE_COLORS, add_corner_label, apply_scientific_luxe_style, short_phase_name, style_axes
from src.viz.video_support import apply_trial_stress, export_gif, phase_segments, predict_phase_track, zscore


STRESS_CASES = [
    ("clean", "clean"),
    ("emg_noise", "emg noise"),
    ("drop_eeg", "drop eeg"),
    ("drop_emg", "drop emg"),
]


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
    spans = phase_segments(t, phase)

    stress_tracks: dict[str, dict[str, np.ndarray]] = {}
    for offset, (stress_case, _) in enumerate(STRESS_CASES):
        eeg_use, emg_use = apply_trial_stress(eeg, emg_env, cfg, stress_case, seed=int(cfg["seed"]) + 1200 + offset)
        probs, pred = predict_phase_track(run_dir, cfg, eeg_use, emg_use, "fusion")
        stress_tracks[stress_case] = {"probs": probs, "pred": pred}

    sig_eeg = zscore(eeg[:, 0])
    sig_emg = zscore(emg_env.mean(axis=1))
    cmap = ListedColormap(PHASE_COLORS)

    fig = plt.figure(figsize=(14.0, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[1.25, 0.72, 0.72, 0.72, 0.72])
    ax_sig = fig.add_subplot(gs[0, 0])
    phase_axes = [fig.add_subplot(gs[idx + 1, 0]) for idx in range(4)]

    style_axes(ax_sig)
    for left, right, phase_idx in spans:
        ax_sig.axvspan(left, right, color=PHASE_COLORS[phase_idx], alpha=0.045, linewidth=0)
    ax_sig.plot(t, sig_eeg, color=MODALITY_COLORS["eeg"], linewidth=1.35, alpha=0.85)
    ax_sig.plot(t, sig_emg, color=MODALITY_COLORS["emg"], linewidth=1.35, alpha=0.78)
    ax_sig.fill_between(t, sig_eeg, 0.0, color=MODALITY_COLORS["eeg"], alpha=0.06)
    ax_sig.fill_between(t, sig_emg, 0.0, color=MODALITY_COLORS["emg"], alpha=0.05)
    ax_sig.set_xlim(float(t[0]), float(t[-1]))
    ymin = float(min(sig_eeg.min(), sig_emg.min())) - 0.6
    ymax = float(max(sig_eeg.max(), sig_emg.max())) + 0.6
    ax_sig.set_ylim(ymin, ymax)
    ax_sig.set_ylabel("z")
    add_corner_label(ax_sig, "signals")
    sig_cursor = ax_sig.axvline(float(t[0]), color="#2e2722", linestyle="--", linewidth=1.0)
    ax_sig.text(0.98, 0.97, "EEG / EMG", transform=ax_sig.transAxes, ha="right", va="top", fontsize=8.2, color="#000000")

    phase_cursors = []
    info_texts = []
    for ax, (stress_case, label) in zip(phase_axes, STRESS_CASES):
        style_axes(ax, hide_ticks=True)
        arr = np.vstack([phase, stress_tracks[stress_case]["pred"]]).astype(np.float32)
        ax.imshow(
            arr,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            extent=[float(t[0]), float(t[-1]), 0.0, 2.0],
            origin="lower",
            vmin=0,
            vmax=3,
        )
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["GT", "PR"])
        add_corner_label(ax, label)
        phase_cursor = ax.axvline(float(t[0]), color="#1a1512", linestyle="--", linewidth=1.0)
        phase_cursors.append(phase_cursor)
        info_texts.append(ax.text(0.985, 1.06, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, color="#000000"))

    phase_axes[-1].set_xlabel("time (s)")
    title = fig.suptitle(f"stress phase robustness  |  trial {args.trial_index:03d}", fontsize=11.6, x=0.51, y=0.995)

    frame_indices = list(range(0, len(t), max(1, int(args.frame_step))))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(1, int(args.max_frames))]

    def update(frame_idx: int):
        time_now = float(t[frame_idx])
        sig_cursor.set_xdata([time_now, time_now])
        for phase_cursor, info_text, (stress_case, _) in zip(phase_cursors, info_texts, STRESS_CASES):
            phase_cursor.set_xdata([time_now, time_now])
            pred_cls = int(stress_tracks[stress_case]["pred"][frame_idx])
            conf = float(stress_tracks[stress_case]["probs"][frame_idx, pred_cls])
            info_text.set_text(f"{short_phase_name(pred_cls)}  p={conf:.2f}")
        title.set_text(f"stress phase robustness  |  trial {args.trial_index:03d}  |  {time_now:0.2f}s")
        return [sig_cursor, *phase_cursors]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000.0 / max(1, int(args.fps)),
        blit=False,
    )

    if args.out_path is None:
        out_dir = ensure_dir(run_dir / "videos")
        out_path = out_dir / f"trial_{args.trial_index:03d}_stress_phase_robustness.mp4"
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
