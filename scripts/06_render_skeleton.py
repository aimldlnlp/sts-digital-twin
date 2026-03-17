\
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.features.dataset import load_trials
from src.common import ensure_dir

def apply_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 9.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

def save_figure(fig: plt.Figure, png_path: Path) -> None:
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _plot_skeleton(ax, joints: np.ndarray, joint_names: list, bones: list):
    # joints: (J,3)
    for a,b in bones:
        ia = joint_names.index(a)
        ib = joint_names.index(b)
        pa, pb = joints[ia], joints[ib]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], linewidth=2.3, color="#1b1b1b")
    ax.scatter(joints[:,0], joints[:,1], joints[:,2], s=16, color="#d62728", zorder=3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--trial_index", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(run_dir/"figures")
    apply_paper_style()

    trials = load_trials(run_dir)
    tp = trials[min(args.trial_index, len(trials)-1)]
    d = np.load(tp, allow_pickle=True)
    joints = d["joints"]  # (T,J,3)
    phase = d["phase"]
    joint_names = [str(x) for x in d["joint_names"]]
    bones = [(str(a), str(b)) for a,b in d["bones"]]

    # keyframes: seated (phase0 mid), momentum (phase1 mid), extension (phase2 mid), standing (phase3 mid)
    def mid_idx(p):
        idx = np.where(phase==p)[0]
        return int(idx[len(idx)//2])
    frames = [mid_idx(0), mid_idx(1), mid_idx(2), mid_idx(3)]
    titles = ["Seated", "Momentum", "Extension", "Standing"]
    panel = ["A", "B", "C", "D"]

    fig = plt.figure(figsize=(10, 7.2))
    chosen = joints[frames]  # (4, J, 3)
    xr = (float(chosen[:, :, 0].min()) - 0.15, float(chosen[:, :, 0].max()) + 0.15)
    yr = (float(chosen[:, :, 1].min()) - 0.18, float(chosen[:, :, 1].max()) + 0.18)
    zr = (float(chosen[:, :, 2].min()) - 0.12, float(chosen[:, :, 2].max()) + 0.12)
    for k,fi in enumerate(frames):
        ax = fig.add_subplot(2,2,k+1, projection="3d")
        _plot_skeleton(ax, joints[fi], joint_names, bones)
        ax.set_title(f"{panel[k]}. {titles[k]}", pad=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-55)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_zlim(*zr)
        ax.grid(False)
    fig.tight_layout()
    out_path = out_dir/"fig_skeleton_grid_2x2.png"
    save_figure(fig, out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
