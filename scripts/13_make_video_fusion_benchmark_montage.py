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
from src.viz.video_support import apply_trial_stress, export_gif, phase_segments, predict_phase_track, predict_torque_segments


MODS = ["eeg", "emg", "fusion"]


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

    eeg_noise_eeg, eeg_noise_emg = apply_trial_stress(eeg, emg_env, cfg, "emg_noise", seed=int(cfg["seed"]) + 1300)

    phase_clean = {}
    phase_noise = {}
    for modality in MODS:
        phase_clean[modality] = predict_phase_track(run_dir, cfg, eeg, emg_env, modality)
        phase_noise[modality] = predict_phase_track(run_dir, cfg, eeg_noise_eeg, eeg_noise_emg, modality)

    centers_emg, torque_emg = predict_torque_segments(run_dir, cfg, eeg, emg_env, "emg")
    centers_fusion, torque_fusion = predict_torque_segments(run_dir, cfg, eeg, emg_env, "fusion")
    centers_emg_noise, torque_emg_noise = predict_torque_segments(run_dir, cfg, eeg_noise_eeg, eeg_noise_emg, "emg")
    centers_fusion_noise, torque_fusion_noise = predict_torque_segments(run_dir, cfg, eeg_noise_eeg, eeg_noise_emg, "fusion")
    t_emg = t[centers_emg.astype(np.int64)] if len(centers_emg) else np.array([], dtype=np.float32)
    t_fusion = t[centers_fusion.astype(np.int64)] if len(centers_fusion) else np.array([], dtype=np.float32)
    t_emg_noise = t[centers_emg_noise.astype(np.int64)] if len(centers_emg_noise) else np.array([], dtype=np.float32)
    t_fusion_noise = t[centers_fusion_noise.astype(np.int64)] if len(centers_fusion_noise) else np.array([], dtype=np.float32)

    fig = plt.figure(figsize=(14.8, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.0])
    ax_phase_clean = fig.add_subplot(gs[0, 0])
    ax_phase_noise = fig.add_subplot(gs[0, 1])
    ax_tau_clean = fig.add_subplot(gs[1, 0])
    ax_tau_noise = fig.add_subplot(gs[1, 1])

    cmap = ListedColormap(PHASE_COLORS)
    for ax, label, tracks in [
        (ax_phase_clean, "clean phase", phase_clean),
        (ax_phase_noise, "emg-noise phase", phase_noise),
    ]:
        style_axes(ax, hide_ticks=True)
        arr = np.vstack([phase, tracks["eeg"][1], tracks["emg"][1], tracks["fusion"][1]]).astype(np.float32)
        ax.imshow(
            arr,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            extent=[float(t[0]), float(t[-1]), 0.0, 4.0],
            origin="lower",
            vmin=0,
            vmax=3,
        )
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(["GT", "EEG", "EMG", "FUS"])
        add_corner_label(ax, label)

    cursor_clean = ax_phase_clean.axvline(float(t[0]), color="#1e1814", linestyle="--", linewidth=1.0)
    cursor_noise = ax_phase_noise.axvline(float(t[0]), color="#1e1814", linestyle="--", linewidth=1.0)
    info_clean = ax_phase_clean.text(0.985, 1.05, "", transform=ax_phase_clean.transAxes, ha="right", va="bottom", fontsize=8.0, color="#000000")
    info_noise = ax_phase_noise.text(0.985, 1.05, "", transform=ax_phase_noise.transAxes, ha="right", va="bottom", fontsize=8.0, color="#000000")

    for ax, label in [(ax_tau_clean, "clean torque"), (ax_tau_noise, "emg-noise torque")]:
        style_axes(ax)
        for left, right, phase_idx in spans:
            ax.axvspan(left, right, color=PHASE_COLORS[phase_idx], alpha=0.04, linewidth=0)
        ax.plot(t, tau_exo, color="#8e8479", linewidth=1.15, alpha=0.75)
        ax.set_xlim(float(t[0]), float(t[-1]))
        pad = float(0.10 * (tau_exo.max() - tau_exo.min() + 1e-6))
        ax.set_ylim(float(tau_exo.min() - pad), float(tau_exo.max() + pad))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("torque")
        add_corner_label(ax, label)

    obs_clean = ax_tau_clean.plot([], [], color="#221c18", linewidth=1.8)[0]
    emg_clean_line = ax_tau_clean.plot([], [], color=MODALITY_COLORS["emg"], linewidth=1.8)[0]
    fusion_clean_line = ax_tau_clean.plot([], [], color=MODALITY_COLORS["fusion"], linewidth=2.0)[0]
    tau_cursor_clean = ax_tau_clean.axvline(float(t[0]), color="#2e2722", linestyle="--", linewidth=1.0)

    obs_noise = ax_tau_noise.plot([], [], color="#221c18", linewidth=1.8)[0]
    emg_noise_line = ax_tau_noise.plot([], [], color=MODALITY_COLORS["emg"], linewidth=1.8)[0]
    fusion_noise_line = ax_tau_noise.plot([], [], color=MODALITY_COLORS["fusion"], linewidth=2.0)[0]
    tau_cursor_noise = ax_tau_noise.axvline(float(t[0]), color="#2e2722", linestyle="--", linewidth=1.0)

    legend_text_clean = ax_tau_clean.text(0.985, 0.97, "obs / emg / fusion", transform=ax_tau_clean.transAxes, ha="right", va="top", fontsize=7.8, color="#000000")
    legend_text_noise = ax_tau_noise.text(0.985, 0.97, "obs / emg / fusion", transform=ax_tau_noise.transAxes, ha="right", va="top", fontsize=7.8, color="#000000")
    _ = legend_text_clean, legend_text_noise

    title = fig.suptitle(f"fusion benchmark montage  |  trial {args.trial_index:03d}", fontsize=11.8, x=0.51, y=0.995)

    frame_indices = list(range(0, len(t), max(1, int(args.frame_step))))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(1, int(args.max_frames))]

    def update(frame_idx: int):
        time_now = float(t[frame_idx])
        pred_clean_cls = int(phase_clean["fusion"][1][frame_idx])
        pred_noise_cls = int(phase_noise["fusion"][1][frame_idx])
        conf_clean = float(phase_clean["fusion"][0][frame_idx, pred_clean_cls])
        conf_noise = float(phase_noise["fusion"][0][frame_idx, pred_noise_cls])
        info_clean.set_text(f"fusion {short_phase_name(pred_clean_cls)}  p={conf_clean:.2f}")
        info_noise.set_text(f"fusion {short_phase_name(pred_noise_cls)}  p={conf_noise:.2f}")
        cursor_clean.set_xdata([time_now, time_now])
        cursor_noise.set_xdata([time_now, time_now])

        obs_clean.set_data(t[: frame_idx + 1], tau_exo[: frame_idx + 1])
        obs_noise.set_data(t[: frame_idx + 1], tau_exo[: frame_idx + 1])
        tau_cursor_clean.set_xdata([time_now, time_now])
        tau_cursor_noise.set_xdata([time_now, time_now])
        emg_clean_line.set_data(t_emg[t_emg <= time_now], torque_emg[t_emg <= time_now])
        fusion_clean_line.set_data(t_fusion[t_fusion <= time_now], torque_fusion[t_fusion <= time_now])
        emg_noise_line.set_data(t_emg_noise[t_emg_noise <= time_now], torque_emg_noise[t_emg_noise <= time_now])
        fusion_noise_line.set_data(t_fusion_noise[t_fusion_noise <= time_now], torque_fusion_noise[t_fusion_noise <= time_now])
        title.set_text(f"fusion benchmark montage  |  trial {args.trial_index:03d}  |  {time_now:0.2f}s")
        return [
            cursor_clean,
            cursor_noise,
            tau_cursor_clean,
            tau_cursor_noise,
            obs_clean,
            emg_clean_line,
            fusion_clean_line,
            obs_noise,
            emg_noise_line,
            fusion_noise_line,
        ]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000.0 / max(1, int(args.fps)),
        blit=False,
    )

    if args.out_path is None:
        out_dir = ensure_dir(run_dir / "videos")
        out_path = out_dir / f"trial_{args.trial_index:03d}_fusion_benchmark_montage.mp4"
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
