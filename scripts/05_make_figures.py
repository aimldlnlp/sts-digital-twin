\
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import stft

from src.common import load_yaml, ensure_dir, resample_to_phase
from src.features.dataset import load_trials, split_indices, make_segments, select_split_groups, StandardScaler1D
from src.models.nets import TorqueRegressor
from src.models.train_utils import get_device, rmse
import torch

COLORS = {
    "eeg": "#1f77b4",
    "emg": "#2ca02c",
    "fusion": "#d62728",
    "neutral": "#4d4d4d",
}

def apply_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "legend.frameon": False,
        "figure.dpi": 220,
        "savefig.dpi": 300,
    })

def save_figure(fig: plt.Figure, png_path: Path) -> None:
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def fig1_coupling_heatmap(run_dir: Path, cfg: Dict[str, Any], out_dir: Path) -> None:
    # Trial-level feature vectors
    trials = load_trials(run_dir)
    sr = int(cfg["sample_rate_hz"])

    feats = []
    names = []
    # define names once
    eeg_names = [f"eeg_mu_c{i}" for i in range(cfg["signals"]["eeg"]["n_channels"])] + \
                [f"eeg_beta_c{i}" for i in range(cfg["signals"]["eeg"]["n_channels"])]
    emg_names = [f"emg_{m}" for m in cfg["signals"]["emg"]["muscles"]]
    kin_names = ["knee_rom", "knee_peak_vel", "tau_exo_meanabs"]
    names = eeg_names + emg_names + kin_names

    from scipy.signal import welch
    def bandpower(x, fmin, fmax):
        f, pxx = welch(x, fs=sr, nperseg=min(len(x), 512))
        m = (f>=fmin)&(f<=fmax)
        return float(np.trapezoid(pxx[m], f[m]) + 1e-12)

    for tp in trials:
        d = np.load(tp, allow_pickle=True)
        eeg = d["eeg"]
        emg_env = d["emg_env"]
        ang = d["angles"][:,1]  # knee
        tau = d["tau_exo"]

        row = []
        for c in range(eeg.shape[1]):
            row.append(bandpower(eeg[:,c], 8, 13))
        for c in range(eeg.shape[1]):
            row.append(bandpower(eeg[:,c], 13, 30))
        row.extend(list(emg_env.mean(axis=0)))
        rom = float(ang.max() - ang.min())
        peak_vel = float(np.max(np.abs(np.gradient(ang) * sr)))
        tau_meanabs = float(np.mean(np.abs(tau)))
        row += [rom, peak_vel, tau_meanabs]
        feats.append(row)

    F = np.array(feats, dtype=np.float32)
    C = np.corrcoef(F, rowvar=False)
    C = np.nan_to_num(C)

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    vmax = max(1e-6, float(np.max(np.abs(C))))
    im = ax.imshow(C, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    ax.set_title("Cross-modal Coupling Matrix", pad=8)
    # sparse ticks to keep readable
    step = max(1, len(names)//14)
    ticks = np.arange(0, len(names), step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([names[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([names[i] for i in ticks], fontsize=7)
    # separators: EEG | EMG | Kinematics
    eeg_end = len(eeg_names)
    emg_end = eeg_end + len(emg_names)
    for b in [eeg_end - 0.5, emg_end - 0.5]:
        ax.axhline(b, color="white", linewidth=0.8)
        ax.axvline(b, color="white", linewidth=0.8)
    save_figure(fig, out_dir/"fig1_coupling_heatmap.png")

def fig2_eeg_spectrogram(run_dir: Path, cfg: Dict[str, Any], out_dir: Path) -> None:
    trials = load_trials(run_dir)
    d = np.load(trials[0], allow_pickle=True)
    eeg = d["eeg"][:,0]  # channel 0
    sr = int(cfg["sample_rate_hz"])
    f, tt, Z = stft(eeg, fs=sr, nperseg=256, noverlap=192)
    P_db = 20.0 * np.log10(np.abs(Z) + 1e-6)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    im = ax.imshow(P_db, aspect="auto", origin="lower",
                   extent=[tt.min(), tt.max(), f.min(), f.max()],
                   cmap="magma")
    ax.set_ylim(0, 60)
    cbar = fig.colorbar(im, ax=ax, label="Power (dB)")
    cbar.ax.tick_params(labelsize=8)
    phase_edges = np.cumsum(cfg["data"]["phase_durations_s"])[:-1]
    for x in phase_edges:
        ax.axvline(float(x), color="white", linestyle="--", linewidth=0.9, alpha=0.9)
    ax.set_title("EEG Time-Frequency Map (Channel 0)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    save_figure(fig, out_dir/"fig2_eeg_spectrogram.png")

def fig3_synergy_weights(run_dir: Path, cfg: Dict[str, Any], out_dir: Path) -> None:
    syn = np.load(run_dir/"artifacts"/"nmf_synergy.npz")
    W = syn["W"]  # (K, M)
    muscles = cfg["signals"]["emg"]["muscles"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    im = ax.imshow(W, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Weight")
    ax.set_yticks(np.arange(W.shape[0]))
    ax.set_yticklabels([f"Syn{k+1}" for k in range(W.shape[0])])
    ax.set_xticks(np.arange(W.shape[1]))
    ax.set_xticklabels(muscles, rotation=45, ha="right")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            val = float(W[i, j])
            txt_color = "white" if val > W.max() * 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
    ax.set_title("EMG Synergy Weights (NMF)")
    save_figure(fig, out_dir/"fig3_synergy_weights.png")

def fig4_synergy_activation_heatmap(run_dir: Path, cfg: Dict[str, Any], out_dir: Path) -> None:
    # Project each trial EMG env to synergy activations by nonnegative least squares approximation
    syn = np.load(run_dir/"artifacts"/"nmf_synergy.npz")
    W = syn["W"]  # (K, M)
    K, M = W.shape
    trials = load_trials(run_dir)

    # pseudo-inverse in nonnegative sense (simple): use least squares then clip
    Wt = W.T  # (M,K)
    # average activation over normalized phase (0..100)
    n_bins = 100
    acc = np.zeros((K, n_bins), dtype=np.float32)
    for tp in trials:
        d = np.load(tp, allow_pickle=True)
        env = d["emg_env"].astype(np.float32)  # (T,M)
        # solve H: (T,K)
        H = np.linalg.lstsq(Wt, env.T, rcond=None)[0].T  # (T,K)
        H = np.clip(H, 0.0, None)
        Hn = resample_to_phase(H, d["phase"], n_bins=n_bins)  # (n_bins,K)
        acc += Hn.T
    acc /= max(1, len(trials))

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    im = ax.imshow(acc, aspect="auto", cmap="inferno")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Activation (a.u.)")
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([f"Syn{k+1}" for k in range(K)])
    ax.set_xticks([0,25,50,75,99])
    ax.set_xticklabels(["0%","25%","50%","75%","100%"])
    for x in [25, 50, 75]:
        ax.axvline(x - 0.5, color="white", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Normalized STS phase (%)")
    ax.set_title("Synergy Activation Across Normalized Phase")
    save_figure(fig, out_dir/"fig4_synergy_activation_heatmap.png")

def fig5_fusion_ablation(run_dir: Path, out_dir: Path) -> None:
    # Read phase metrics for eeg/emg/fusion
    mets = {}
    for m in ["eeg","emg","fusion"]:
        p = run_dir/"artifacts"/f"phase_metrics_{m}.json"
        if p.exists():
            mets[m] = json.loads(p.read_text())
    labels = [x for x in ["eeg", "emg", "fusion"] if x in mets]
    vals = [mets[k]["test_macro_f1"] for k in labels]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    bars = ax.bar(labels, vals, color=[COLORS[k] for k in labels], alpha=0.9, width=0.65)
    ax.axhline(0.25, color=COLORS["neutral"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(2.35 if len(labels) >= 3 else len(labels)-0.2, 0.255, "chance", fontsize=8, color=COLORS["neutral"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Phase Classification Ablation")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2.0, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, out_dir/"fig5_fusion_ablation.png")

def fig6_torque_overlay(run_dir: Path, cfg: Dict[str, Any], out_dir: Path) -> None:
    # Use trained torque model (fusion) to predict on test segments.
    device = get_device()

    # segments (fusion)
    seg = make_segments(run_dir, cfg, modality="fusion")
    X, y = seg["X"], seg["y_tau"]
    groups = select_split_groups(seg, cfg)
    tr_idx, va_idx, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"])+2, groups=groups)

    # scaler + model
    import json as _json
    sc_path = run_dir/"artifacts"/"scaler_fusion_torque.json"
    scaler = StandardScaler1D.from_dict(_json.loads(sc_path.read_text()))
    Xs = scaler.transform(X)

    model = TorqueRegressor(in_ch=Xs.shape[1]).to(device)
    model.load_state_dict(torch.load(run_dir/"artifacts"/"torque_model_fusion.pt", map_location=device))
    model.eval()

    # pick first 600 test segments
    idx = te_idx[:600]
    with torch.no_grad():
        pred = model(torch.from_numpy(Xs[idx]).to(device)).cpu().numpy()
    truth = y[idx]
    e = rmse(pred, truth)
    corr = float(np.corrcoef(truth, pred)[0, 1]) if len(truth) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(truth, pred, s=11, alpha=0.45, color=COLORS["fusion"], edgecolors="none")
    lo = float(min(truth.min(), pred.min()))
    hi = float(max(truth.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=COLORS["neutral"], linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target torque (segment mean, a.u.)")
    ax.set_ylabel("Predicted torque (a.u.)")
    ax.set_title("Torque Prediction (Fusion Model)")
    ax.text(0.02, 0.98, f"RMSE = {e:.3f}\nr = {corr:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.25"))
    save_figure(fig, out_dir/"fig6_torque_overlay.png")

def fig7_robustness_curve(run_dir: Path, cfg: Dict[str, Any], out_dir: Path) -> None:
    # robustness: torque RMSE vs EMG noise multiplier (evaluate on fixed test set)
    device = get_device()

    # load trained torque model + scaler
    import json as _json
    scaler = StandardScaler1D.from_dict(_json.loads((run_dir/"artifacts"/"scaler_fusion_torque.json").read_text()))
    model = TorqueRegressor(in_ch=cfg["signals"]["eeg"]["n_channels"] + len(cfg["signals"]["emg"]["muscles"])).to(device)
    model.load_state_dict(torch.load(run_dir/"artifacts"/"torque_model_fusion.pt", map_location=device))
    model.eval()

    # build base segments from raw trials so we can inject noise into EMG part only
    seg_base = make_segments(run_dir, cfg, modality="fusion")
    X, y = seg_base["X"], seg_base["y_tau"]
    groups = select_split_groups(seg_base, cfg)
    _, _, te_idx = split_indices(len(y), cfg, seed=int(cfg["seed"])+2, groups=groups)
    idx = te_idx[:1200]
    X0 = X[idx].copy()
    y0 = y[idx].copy()

    eeg_ch = int(cfg["signals"]["eeg"]["n_channels"])
    mults = cfg["robustness"]["emg_noise_multipliers"]

    rng = np.random.default_rng(int(cfg["seed"]) + 999)
    rmses_mu = []
    rmses_sd = []
    n_repeats = 6
    for m in mults:
        rep = []
        for _ in range(n_repeats):
            Xn = X0.copy()
            # Add noise only to EMG channels
            emg = Xn[:, eeg_ch:, :]
            noise = rng.normal(0, float(m-1.0)*0.15, size=emg.shape).astype(np.float32)
            Xn[:, eeg_ch:, :] = emg + noise
            Xs = scaler.transform(Xn)
            with torch.no_grad():
                pred = model(torch.from_numpy(Xs).to(device)).cpu().numpy()
            rep.append(rmse(pred, y0))
        rmses_mu.append(float(np.mean(rep)))
        rmses_sd.append(float(np.std(rep)))

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.errorbar(mults, rmses_mu, yerr=rmses_sd, marker="o", capsize=3, color=COLORS["fusion"], linewidth=1.8)
    ax.set_xlabel("EMG noise multiplier")
    ax.set_ylabel("Torque RMSE")
    ax.set_title("Robustness to EMG Noise (mean +/- std)")
    save_figure(fig, out_dir/"fig7_robustness_curve.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir/"config.yaml")
    fig_dir = ensure_dir(run_dir/"figures")
    apply_paper_style()

    fig1_coupling_heatmap(run_dir, cfg, fig_dir)
    fig2_eeg_spectrogram(run_dir, cfg, fig_dir)
    fig3_synergy_weights(run_dir, cfg, fig_dir)
    fig4_synergy_activation_heatmap(run_dir, cfg, fig_dir)
    fig5_fusion_ablation(run_dir, fig_dir)
    fig6_torque_overlay(run_dir, cfg, fig_dir)
    fig7_robustness_curve(run_dir, cfg, fig_dir)
    print("Saved figures to:", fig_dir)

if __name__ == "__main__":
    main()
