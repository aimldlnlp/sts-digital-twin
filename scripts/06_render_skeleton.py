from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from src.common import ensure_dir
from src.features.dataset import load_trials
from src.viz import PHASE_COLORS, apply_scientific_luxe_style, save_figure, style_3d_axes


def plot_skeleton(
    ax: plt.Axes,
    joints: np.ndarray,
    joint_names: list[str],
    bones: list[tuple[str, str]],
    *,
    accent: str,
    z_floor: float,
    ghost: list[np.ndarray] | None = None,
) -> None:
    ground = joints.copy()
    ground[:, 2] = z_floor
    if ghost:
        for ghost_joints in ghost:
            for start, end in bones:
                i0 = joint_names.index(start)
                i1 = joint_names.index(end)
                p0, p1 = ghost_joints[i0], ghost_joints[i1]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=1.5, color=accent, alpha=0.12)
    for start, end in bones:
        i0 = joint_names.index(start)
        i1 = joint_names.index(end)
        p0, p1 = joints[i0], joints[i1]
        g0, g1 = ground[i0], ground[i1]
        ax.plot([g0[0], g1[0]], [g0[1], g1[1]], [g0[2], g1[2]], linewidth=3.8, color="#d9cec0", alpha=0.22)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=2.8, color="#2a211d")
    ax.scatter(ground[:, 0], ground[:, 1], ground[:, 2], s=46, color="#d9cec0", alpha=0.18, depthshade=False)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=28, color=accent, depthshade=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--trial_index", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(run_dir / "figures")
    style_info = apply_scientific_luxe_style("figure")

    trials = load_trials(run_dir)
    trial_path = trials[min(args.trial_index, len(trials) - 1)]
    data = np.load(trial_path, allow_pickle=True)
    joints = data["joints"]
    phase = data["phase"]
    joint_names = [str(name) for name in data["joint_names"]]
    bones = [(str(a), str(b)) for a, b in data["bones"]]

    def mid_index(phase_idx: int) -> int:
        idx = np.where(phase == phase_idx)[0]
        return int(idx[len(idx) // 2])

    frame_ids = [mid_index(i) for i in range(4)]
    chosen = joints[frame_ids]
    x_limits = (float(chosen[:, :, 0].min()) - 0.15, float(chosen[:, :, 0].max()) + 0.15)
    y_limits = (float(chosen[:, :, 1].min()) - 0.18, float(chosen[:, :, 1].max()) + 0.18)
    z_limits = (float(chosen[:, :, 2].min()) - 0.10, float(chosen[:, :, 2].max()) + 0.10)
    z_floor = z_limits[0]

    fig = plt.figure(figsize=(10.2, 7.4))
    for idx, frame_id in enumerate(frame_ids):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        style_3d_axes(ax)
        ax.set_facecolor(to_rgba(PHASE_COLORS[idx], 0.08))
        ghost_ids = [max(0, frame_id - 8), min(len(joints) - 1, frame_id + 8)]
        ghosts = [joints[g] for g in ghost_ids if g != frame_id]
        plot_skeleton(ax, joints[frame_id], joint_names, bones, accent=PHASE_COLORS[idx], z_floor=z_floor, ghost=ghosts)
        xs = np.linspace(x_limits[0], x_limits[1], 16)
        ys = np.linspace(y_limits[0], y_limits[1], 16)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, z_floor)
        ax.plot_surface(xx, yy, zz, color=to_rgba(PHASE_COLORS[idx], 0.06), shade=False, linewidth=0, antialiased=False)
        ax.view_init(elev=18, azim=-58)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_zlim(*z_limits)
        ax.text2D(
            0.02,
            0.98,
            ["prep", "momentum", "extend", "stable"][idx],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            color="#000000",
            bbox={"facecolor": "#ffffff", "edgecolor": "#ded4c7", "linewidth": 0.6, "pad": 0.18},
        )
    fig.tight_layout()

    old_path = out_dir / "fig_skeleton_grid_2x2.png"
    old_path.unlink(missing_ok=True)
    out_path = out_dir / "skeleton_grid_2x2.png"
    save_figure(fig, out_path)
    pdf_path = out_path.with_suffix(".pdf")
    pdf_path.unlink(missing_ok=True)
    print(f"Using font: {style_info['font_primary']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
