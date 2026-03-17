from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import torch

from src.common import load_yaml, ensure_dir
from src.features.dataset import load_trials, StandardScaler1D
from src.models.nets import TorqueRegressor
from src.models.train_utils import get_device


PHASE_NAMES = ["Preparation", "Momentum", "Extension", "Stabilization"]
PHASE_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]


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


def _make_model_input(eeg: np.ndarray, emg_env: np.ndarray, modality: str) -> np.ndarray:
    if modality == "eeg":
        return eeg
    if modality == "emg":
        return emg_env
    return np.concatenate([eeg, emg_env], axis=1)


def _predict_torque_segments(
    run_dir: Path,
    cfg: Dict[str, Any],
    inp: np.ndarray,
    tau_exo: np.ndarray,
    modality: str,
) -> Tuple[np.ndarray, np.ndarray]:
    seg_len = int(cfg["train"]["segment_len"])
    stride = int(cfg["train"]["segment_stride"])

    starts = list(range(0, inp.shape[0] - seg_len + 1, stride))
    if not starts:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    segs = np.stack([inp[s:s + seg_len].T for s in starts], axis=0).astype(np.float32)
    centers = np.array([s + seg_len // 2 for s in starts], dtype=np.int64)
    _ = np.array([float(np.mean(tau_exo[s:s + seg_len])) for s in starts], dtype=np.float32)

    scaler_path = run_dir / "artifacts" / f"scaler_{modality}_torque.json"
    model_path = run_dir / "artifacts" / f"torque_model_{modality}.pt"
    if not (scaler_path.exists() and model_path.exists()):
        return centers.astype(np.float32), np.full(len(centers), np.nan, dtype=np.float32)

    scaler = StandardScaler1D.from_dict(json.loads(scaler_path.read_text()))
    x = scaler.transform(segs)

    device = get_device()
    model = TorqueRegressor(in_ch=x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    bs = 512
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds).astype(np.float32)
    return centers.astype(np.float32), pred


def _precompute_bones(joint_names: List[str], bones: List[Tuple[str, str]]) -> List[Tuple[int, int]]:
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    return [(name_to_idx[a], name_to_idx[b]) for a, b in bones]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--trial_index", type=int, default=0)
    ap.add_argument("--modality", type=str, default="fusion", choices=["fusion", "eeg", "emg"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame_step", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality (lower is better, typical 16-23).")
    ap.add_argument("--preset", type=str, default="slow",
                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"])
    ap.add_argument("--max_frames", type=int, default=None, help="Optional cap for fast debugging.")
    ap.add_argument("--no_pred", action="store_true", help="Disable model prediction overlay.")
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
    joints = d["joints"].astype(np.float32)          # (T, J, 3)
    eeg = d["eeg"].astype(np.float32)                # (T, C_eeg)
    emg_env = d["emg_env"].astype(np.float32)        # (T, M)
    tau_exo = d["tau_exo"].astype(np.float32)        # (T,)

    joint_names = [str(x) for x in d["joint_names"]]
    bones = [(str(a), str(b)) for a, b in d["bones"]]
    bone_pairs = _precompute_bones(joint_names, bones)

    sig_eeg = _zscore(eeg[:, 0])
    sig_emg = _zscore(emg_env.mean(axis=1))

    inp = _make_model_input(eeg, emg_env, modality=args.modality)
    if args.no_pred:
        pred_t_centers = np.array([], dtype=np.float32)
        pred_tau = np.array([], dtype=np.float32)
    else:
        pred_t_centers, pred_tau = _predict_torque_segments(run_dir, cfg, inp, tau_exo, args.modality)
        pred_t_centers = t[pred_t_centers.astype(np.int64)] if len(pred_t_centers) else pred_t_centers

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.4], height_ratios=[1.2, 0.8, 0.8])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_sig = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 1])
    ax_tau = fig.add_subplot(gs[2, 1])

    # 3D skeleton panel
    x_min, x_max = float(joints[:, :, 0].min()) - 0.2, float(joints[:, :, 0].max()) + 0.2
    y_min, y_max = float(joints[:, :, 1].min()) - 0.2, float(joints[:, :, 1].max()) + 0.2
    z_min, z_max = float(joints[:, :, 2].min()) - 0.2, float(joints[:, :, 2].max()) + 0.2
    ax3d.set_xlim(x_min, x_max)
    ax3d.set_ylim(y_min, y_max)
    ax3d.set_zlim(z_min, z_max)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("STS Skeleton (3D)")
    ax3d.view_init(elev=18, azim=-55)
    ax3d.grid(False)

    bone_lines = []
    for _ in bone_pairs:
        line, = ax3d.plot([], [], [], linewidth=2.2, color="#1b1b1b")
        bone_lines.append(line)
    joint_scatter = ax3d.scatter([], [], [], s=18, color="#d62728")

    # Signal panel
    ax_sig.plot(t, sig_eeg, color="#1f77b4", alpha=0.25, linewidth=1.0)
    ax_sig.plot(t, sig_emg, color="#2ca02c", alpha=0.25, linewidth=1.0)
    sig_line_eeg, = ax_sig.plot([], [], color="#1f77b4", linewidth=1.4, label="EEG ch0 (z-score)")
    sig_line_emg, = ax_sig.plot([], [], color="#2ca02c", linewidth=1.4, label="EMG mean envelope (z-score)")
    sig_cursor = ax_sig.axvline(t[0], color="#444444", linestyle="--", linewidth=1.0)
    ax_sig.set_xlim(float(t[0]), float(t[-1]))
    ymin = float(min(sig_eeg.min(), sig_emg.min())) - 0.5
    ymax = float(max(sig_eeg.max(), sig_emg.max())) + 0.5
    ax_sig.set_ylim(ymin, ymax)
    ax_sig.set_ylabel("Normalized amplitude")
    ax_sig.set_title("Synchronized Neural Signals")
    ax_sig.legend(loc="upper right")

    # Phase panel
    cmap = ListedColormap(PHASE_COLORS)
    ax_phase.imshow(phase[None, :], aspect="auto", interpolation="nearest", cmap=cmap,
                    extent=[float(t[0]), float(t[-1]), 0.0, 1.0], vmin=0, vmax=3)
    phase_cursor = ax_phase.axvline(t[0], color="black", linestyle="--", linewidth=1.0)
    ax_phase.set_yticks([])
    ax_phase.set_xlim(float(t[0]), float(t[-1]))
    ax_phase.set_title("STS Phase Timeline")
    ax_phase.set_xlabel("Time (s)")
    phase_text = ax_phase.text(0.01, 1.10, "Phase: Preparation", transform=ax_phase.transAxes, fontsize=10)
    legend_text = " | ".join([f"{i}:{name}" for i, name in enumerate(PHASE_NAMES)])
    ax_phase.text(0.01, -0.45, legend_text, transform=ax_phase.transAxes, fontsize=9)

    # Torque panel
    ax_tau.plot(t, tau_exo, color="#555555", alpha=0.3, linewidth=1.0)
    tau_line, = ax_tau.plot([], [], color="#222222", linewidth=1.8, label="Target exo torque")
    tau_cursor = ax_tau.axvline(t[0], color="#444444", linestyle="--", linewidth=1.0)
    if len(pred_t_centers):
        pred_line, = ax_tau.plot([], [], color="#d62728", linewidth=1.8, label=f"Predicted torque ({args.modality})")
    else:
        pred_line = None
    ax_tau.set_xlim(float(t[0]), float(t[-1]))
    tau_pad = float(max(0.25, 0.08 * (tau_exo.max() - tau_exo.min() + 1e-6)))
    ax_tau.set_ylim(float(tau_exo.min() - tau_pad), float(tau_exo.max() + tau_pad))
    ax_tau.set_xlabel("Time (s)")
    ax_tau.set_ylabel("Torque (a.u.)")
    ax_tau.set_title("Assist Torque")
    ax_tau.legend(loc="upper right")

    suptitle = fig.suptitle("", fontsize=12)

    frame_indices = list(range(0, len(t), max(1, int(args.frame_step))))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(1, int(args.max_frames))]

    def init():
        for ln in bone_lines:
            ln.set_data_3d([], [], [])
        joint_scatter._offsets3d = ([], [], [])
        sig_line_eeg.set_data([], [])
        sig_line_emg.set_data([], [])
        tau_line.set_data([], [])
        if pred_line is not None:
            pred_line.set_data([], [])
        return bone_lines + [joint_scatter, sig_line_eeg, sig_line_emg, tau_line]

    def update(frame_idx: int):
        j = joints[frame_idx]
        for ln, (ia, ib) in zip(bone_lines, bone_pairs):
            pa, pb = j[ia], j[ib]
            ln.set_data_3d([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]])
        joint_scatter._offsets3d = (j[:, 0], j[:, 1], j[:, 2])

        ti = t[frame_idx]
        sig_line_eeg.set_data(t[:frame_idx + 1], sig_eeg[:frame_idx + 1])
        sig_line_emg.set_data(t[:frame_idx + 1], sig_emg[:frame_idx + 1])
        sig_cursor.set_xdata([ti, ti])

        phase_cursor.set_xdata([ti, ti])
        phase_text.set_text(f"Phase: {PHASE_NAMES[int(phase[frame_idx])]}")

        tau_line.set_data(t[:frame_idx + 1], tau_exo[:frame_idx + 1])
        tau_cursor.set_xdata([ti, ti])
        if pred_line is not None and len(pred_t_centers):
            mask = pred_t_centers <= ti
            pred_line.set_data(pred_t_centers[mask], pred_tau[mask])

        suptitle.set_text(
            f"Trial {args.trial_index:03d} | t={ti:.2f}s | phase={int(phase[frame_idx])} ({PHASE_NAMES[int(phase[frame_idx])]})"
        )
        return bone_lines + [joint_scatter, sig_line_eeg, sig_line_emg, tau_line]

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
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
