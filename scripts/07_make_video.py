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
from src.features.dataset import StandardScaler1D, StandardScalerTarget, load_trials
from src.models.nets import load_model_checkpoint
from src.models.train_utils import get_device
from src.viz import (
    MODALITY_COLORS,
    PHASE_COLORS,
    PHASE_NAMES,
    add_corner_label,
    apply_scientific_luxe_style,
    short_phase_name,
    style_3d_axes,
    style_axes,
)


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def make_model_input(eeg: np.ndarray, emg_env: np.ndarray, modality: str) -> np.ndarray:
    if modality == "eeg":
        return eeg
    if modality == "emg":
        return emg_env
    return np.concatenate([eeg, emg_env], axis=1)


def predict_torque_segments(run_dir: Path, cfg: dict[str, Any], inp: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
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


def precompute_bones(joint_names: list[str], bones: list[tuple[str, str]]) -> list[tuple[int, int]]:
    index = {name: idx for idx, name in enumerate(joint_names)}
    return [(index[a], index[b]) for a, b in bones]


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
    parser.add_argument("--modality", type=str, default="fusion", choices=["fusion", "eeg", "emg"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame_step", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="slow", choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"])
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_pred", action="store_true")
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
    joints = data["joints"].astype(np.float32)
    eeg = data["eeg"].astype(np.float32)
    emg_env = data["emg_env"].astype(np.float32)
    tau_exo = data["tau_exo"].astype(np.float32)
    joint_names = [str(name) for name in data["joint_names"]]
    bones = [(str(a), str(b)) for a, b in data["bones"]]
    bone_pairs = precompute_bones(joint_names, bones)
    spans = phase_segments(t, phase)
    com = joints.mean(axis=1)

    sig_eeg = zscore(eeg[:, 0])
    sig_emg = zscore(emg_env.mean(axis=1))
    inp = make_model_input(eeg, emg_env, args.modality)
    if args.no_pred:
        pred_centers = np.array([], dtype=np.float32)
        pred_tau = np.array([], dtype=np.float32)
    else:
        pred_centers, pred_tau = predict_torque_segments(run_dir, cfg, inp, args.modality)
        pred_centers = t[pred_centers.astype(np.int64)] if len(pred_centers) else pred_centers

    fig = plt.figure(figsize=(14.6, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.08, 1.35], height_ratios=[1.1, 0.55, 0.75])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_sig = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 1])
    ax_tau = fig.add_subplot(gs[2, 1])

    style_3d_axes(ax3d)
    x_limits = (float(joints[:, :, 0].min()) - 0.2, float(joints[:, :, 0].max()) + 0.2)
    y_limits = (float(joints[:, :, 1].min()) - 0.2, float(joints[:, :, 1].max()) + 0.2)
    z_limits = (float(joints[:, :, 2].min()) - 0.2, float(joints[:, :, 2].max()) + 0.2)
    ax3d.set_xlim(*x_limits)
    ax3d.set_ylim(*y_limits)
    ax3d.set_zlim(*z_limits)
    ax3d.view_init(elev=18, azim=-58)
    add_corner_label(ax3d, "motion")

    bone_lines = [ax3d.plot([], [], [], linewidth=2.4, color="#241d19")[0] for _ in bone_pairs]
    joint_scatter = ax3d.scatter([], [], [], s=20, color=MODALITY_COLORS["accent"], depthshade=False)
    trail_line = ax3d.plot([], [], [], linewidth=1.4, color=MODALITY_COLORS["fusion"], alpha=0.42)[0]
    shadow_line = ax3d.plot([], [], [], linewidth=1.8, color="#d2c4b3", alpha=0.45)[0]

    style_axes(ax_sig)
    for left, right, phase_idx in spans:
        ax_sig.axvspan(left, right, color=PHASE_COLORS[phase_idx], alpha=0.045, linewidth=0, zorder=0)
    ax_sig.plot(t, sig_eeg, color=MODALITY_COLORS["eeg"], alpha=0.18, linewidth=1.1)
    ax_sig.plot(t, sig_emg, color=MODALITY_COLORS["emg"], alpha=0.18, linewidth=1.1)
    sig_line_eeg = ax_sig.plot([], [], color=MODALITY_COLORS["eeg"], linewidth=1.5)[0]
    sig_line_emg = ax_sig.plot([], [], color=MODALITY_COLORS["emg"], linewidth=1.5)[0]
    sig_cursor = ax_sig.axvline(float(t[0]), color="#3b342f", linestyle="--", linewidth=1.0)
    ax_sig.fill_between(t, sig_eeg, 0.0, color=MODALITY_COLORS["eeg"], alpha=0.05)
    ax_sig.fill_between(t, sig_emg, 0.0, color=MODALITY_COLORS["emg"], alpha=0.05)
    ax_sig.set_xlim(float(t[0]), float(t[-1]))
    ymin = float(min(sig_eeg.min(), sig_emg.min())) - 0.6
    ymax = float(max(sig_eeg.max(), sig_emg.max())) + 0.6
    ax_sig.set_ylim(ymin, ymax)
    ax_sig.set_xlabel("time (s)")
    add_corner_label(ax_sig, "EEG / EMG")
    phase_strip = inset_axes(ax_sig, width="100%", height="10%", loc="upper center", borderpad=0.55)
    phase_strip.imshow(
        phase[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(PHASE_COLORS),
        extent=[float(t[0]), float(t[-1]), 0.0, 1.0],
        vmin=0,
        vmax=3,
    )
    style_axes(phase_strip, hide_ticks=True)
    phase_strip_cursor = phase_strip.axvline(float(t[0]), color="#241d19", linestyle="--", linewidth=0.9)

    style_axes(ax_phase, hide_ticks=True)
    ax_phase.imshow(
        phase[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(PHASE_COLORS),
        extent=[float(t[0]), float(t[-1]), 0.0, 1.0],
        vmin=0,
        vmax=3,
    )
    phase_cursor = ax_phase.axvline(float(t[0]), color="#241d19", linestyle="--", linewidth=1.0)
    ax_phase.set_xlim(float(t[0]), float(t[-1]))
    phase_text = ax_phase.text(0.02, 1.08, "Prep", transform=ax_phase.transAxes, fontsize=10.2, color="#000000", ha="left", va="bottom")
    for x0, x1, phase_idx in spans:
        xmid = 0.5 * (x0 + x1)
        ax_phase.text(xmid, 0.5, short_phase_name(phase_idx), ha="center", va="center", fontsize=8.1, color="#000000")

    style_axes(ax_tau)
    for left, right, phase_idx in spans:
        ax_tau.axvspan(left, right, color=PHASE_COLORS[phase_idx], alpha=0.045, linewidth=0, zorder=0)
    ax_tau.plot(t, tau_exo, color="#93887a", alpha=0.35, linewidth=1.1)
    ax_tau.fill_between(t, tau_exo, np.zeros_like(tau_exo), color=MODALITY_COLORS["fusion"], alpha=0.08)
    tau_line = ax_tau.plot([], [], color="#241d19", linewidth=1.8)[0]
    tau_cursor = ax_tau.axvline(float(t[0]), color="#3b342f", linestyle="--", linewidth=1.0)
    pred_line = None
    if len(pred_centers):
        pred_line = ax_tau.plot([], [], color=MODALITY_COLORS["fusion"], linewidth=1.8)[0]
    tau_pad = float(max(0.25, 0.08 * (tau_exo.max() - tau_exo.min() + 1e-6)))
    ax_tau.set_xlim(float(t[0]), float(t[-1]))
    ax_tau.set_ylim(float(tau_exo.min() - tau_pad), float(tau_exo.max() + tau_pad))
    ax_tau.set_xlabel("time (s)")
    ax_tau.set_ylabel("torque")
    add_corner_label(ax_tau, f"torque / {args.modality.upper()}")
    ax_tau.text(0.98, 0.98, "obs", transform=ax_tau.transAxes, ha="right", va="top", fontsize=7.8, color="#93887a")
    if pred_line is not None:
        ax_tau.text(0.98, 0.90, "pred", transform=ax_tau.transAxes, ha="right", va="top", fontsize=7.8, color=MODALITY_COLORS["fusion"])

    meta_text = ax_sig.text(0.98, 1.06, f"trial {args.trial_index:03d}", transform=ax_sig.transAxes, ha="right", va="bottom", fontsize=8.5, color="#000000")

    frame_indices = list(range(0, len(t), max(1, int(args.frame_step))))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(1, int(args.max_frames))]

    def init():
        for line in bone_lines:
            line.set_data_3d([], [], [])
        joint_scatter._offsets3d = ([], [], [])
        trail_line.set_data_3d([], [], [])
        shadow_line.set_data_3d([], [], [])
        sig_line_eeg.set_data([], [])
        sig_line_emg.set_data([], [])
        tau_line.set_data([], [])
        if pred_line is not None:
            pred_line.set_data([], [])
        return bone_lines + [joint_scatter, trail_line, shadow_line, sig_line_eeg, sig_line_emg, tau_line]

    def update(frame_idx: int):
        joints_now = joints[frame_idx]
        for line, (i0, i1) in zip(bone_lines, bone_pairs):
            p0, p1 = joints_now[i0], joints_now[i1]
            line.set_data_3d([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]])
        joint_scatter._offsets3d = (joints_now[:, 0], joints_now[:, 1], joints_now[:, 2])
        trail_start = max(0, frame_idx - 18)
        trail = com[trail_start: frame_idx + 1]
        trail_line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
        shadow_line.set_data_3d(trail[:, 0], trail[:, 1], np.full(len(trail), z_limits[0]))

        time_now = float(t[frame_idx])
        sig_line_eeg.set_data(t[: frame_idx + 1], sig_eeg[: frame_idx + 1])
        sig_line_emg.set_data(t[: frame_idx + 1], sig_emg[: frame_idx + 1])
        sig_cursor.set_xdata([time_now, time_now])
        phase_strip_cursor.set_xdata([time_now, time_now])
        phase_cursor.set_xdata([time_now, time_now])
        tau_line.set_data(t[: frame_idx + 1], tau_exo[: frame_idx + 1])
        tau_cursor.set_xdata([time_now, time_now])
        phase_text.set_text(short_phase_name(int(phase[frame_idx])))
        meta_text.set_text(f"trial {args.trial_index:03d}  |  {time_now:0.2f}s")
        if pred_line is not None:
            mask = pred_centers <= time_now
            pred_line.set_data(pred_centers[mask], pred_tau[mask])
        return bone_lines + [joint_scatter, trail_line, shadow_line, sig_line_eeg, sig_line_emg, tau_line]

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
        out_path = out_dir / f"trial_{args.trial_index:03d}_multimodal.mp4"
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
