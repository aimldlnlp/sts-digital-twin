from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import trapezoid
from scipy.signal import stft, welch
from scipy.stats import gaussian_kde

from src.common import ensure_dir, load_yaml, resample_to_phase
from src.features.dataset import StandardScaler1D, StandardScalerTarget, load_trials, make_segments, select_split_groups, split_indices
from src.models.nets import load_model_checkpoint
from src.models.train_utils import get_device, rmse, torque_error_by_range
from src.viz import (
    MODALITY_COLORS,
    PHASE_COLORS,
    abbreviate_muscles,
    add_corner_label,
    add_phase_bands,
    apply_scientific_luxe_style,
    quantile_limits,
    save_figure,
    style_axes,
    style_colorbar,
)


def fig1_coupling_heatmap(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    trials = load_trials(run_dir)
    sr = int(cfg["sample_rate_hz"])

    eeg_names = [f"mu{i+1}" for i in range(cfg["signals"]["eeg"]["n_channels"])]
    eeg_names += [f"b{i+1}" for i in range(cfg["signals"]["eeg"]["n_channels"])]
    emg_names = abbreviate_muscles(cfg["signals"]["emg"]["muscles"])
    kin_names = ["ROM", "VEL", "TAU"]
    names = eeg_names + emg_names + kin_names
    blocks = [
        ("EEG", 0, len(eeg_names), MODALITY_COLORS["eeg"]),
        ("EMG", len(eeg_names), len(emg_names), MODALITY_COLORS["emg"]),
        ("KIN", len(eeg_names) + len(emg_names), len(kin_names), MODALITY_COLORS["accent"]),
    ]

    def bandpower(x: np.ndarray, fmin: float, fmax: float) -> float:
        f, pxx = welch(x, fs=sr, nperseg=min(len(x), 512))
        mask = (f >= fmin) & (f <= fmax)
        return float(trapezoid(pxx[mask], f[mask]) + 1e-12)

    rows = []
    for tp in trials:
        data = np.load(tp, allow_pickle=True)
        eeg = data["eeg"]
        emg_env = data["emg_env"]
        knee = data["angles"][:, 1]
        tau = data["tau_exo"]

        row = [bandpower(eeg[:, c], 8.0, 13.0) for c in range(eeg.shape[1])]
        row += [bandpower(eeg[:, c], 13.0, 30.0) for c in range(eeg.shape[1])]
        row += list(emg_env.mean(axis=0))
        row += [
            float(knee.max() - knee.min()),
            float(np.max(np.abs(np.gradient(knee) * sr))),
            float(np.mean(np.abs(tau))),
        ]
        rows.append(row)

    corr = np.corrcoef(np.asarray(rows, dtype=np.float32), rowvar=False)
    corr = np.nan_to_num(corr)

    fig = plt.figure(figsize=(9.1, 8.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.18, 1.0], height_ratios=[0.18, 1.0], wspace=0.04, hspace=0.04)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 0])
    ax = fig.add_subplot(gs[1, 1])
    vmax = float(np.max(np.abs(corr)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    style_axes(ax)
    focus = ["mu1", "mu4", "b1", "b4", emg_names[0], emg_names[-1], "ROM", "TAU"]
    tick_positions = [names.index(name) for name in focus if name in names]
    tick_labels = [names[idx] for idx in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(length=0)
    eeg_end = len(eeg_names)
    emg_end = eeg_end + len(emg_names)
    for boundary in [eeg_end - 0.5, emg_end - 0.5]:
        ax.axhline(boundary, color="white", linewidth=1.0, alpha=0.85)
        ax.axvline(boundary, color="white", linewidth=1.0, alpha=0.85)
    add_corner_label(ax, "reduced labels")

    for block_name, start, count, color in blocks:
        ax_top.add_patch(Rectangle((start - 0.5, 0.0), count, 1.0, facecolor=color, alpha=0.92, linewidth=0))
        ax_top.text(start + count / 2.0 - 0.5, 0.5, block_name, ha="center", va="center", fontsize=10, color="#000000")
        ax_side.add_patch(Rectangle((0.0, start - 0.5), 1.0, count, facecolor=color, alpha=0.92, linewidth=0))
        ax_side.text(0.5, start + count / 2.0 - 0.5, block_name, ha="center", va="center", rotation=90, fontsize=10, color="#000000")

    for aux in [ax_top, ax_side]:
        aux.set_facecolor("#f6f1e8")
        aux.set_xticks([])
        aux.set_yticks([])
        for spine in aux.spines.values():
            spine.set_visible(False)
    ax_top.set_xlim(-0.5, len(names) - 0.5)
    ax_top.set_ylim(0.0, 1.0)
    ax_side.set_xlim(0.0, 1.0)
    ax_side.set_ylim(len(names) - 0.5, -0.5)

    mask = np.ones_like(corr, dtype=bool)
    mask[:eeg_end, :eeg_end] = False
    mask[eeg_end:emg_end, eeg_end:emg_end] = False
    mask[emg_end:, emg_end:] = False
    score = np.abs(corr) * mask
    peak_indices = np.dstack(np.unravel_index(np.argsort(score.ravel())[-6:], score.shape))[0]
    seen: set[tuple[int, int]] = set()
    lead_pair = None
    for row, col in peak_indices:
        if (int(col), int(row)) in seen or row == col:
            continue
        seen.add((int(row), int(col)))
        ax.scatter(col, row, s=18, facecolor="#fbf8f2", edgecolor="#1d1a17", linewidth=0.7, zorder=3)
        if lead_pair is None:
            lead_pair = (int(row), int(col))

    if lead_pair is not None:
        row, col = lead_pair
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1.0, 1.0, fill=False, edgecolor="#1d1a17", linewidth=1.1, zorder=4))
        pair_label = f"peak {names[row]} / {names[col]}"
        ax.text(
            0.98,
            0.04,
            pair_label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.1,
            color="#000000",
            bbox={"facecolor": "#ffffff", "edgecolor": "#ded4c7", "linewidth": 0.6, "pad": 0.20},
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    style_colorbar(cbar, "Pearson r")
    save_figure(fig, out_dir / "coupling_heatmap.png")


def fig2_eeg_spectrogram(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    trials = load_trials(run_dir)
    data = np.load(trials[0], allow_pickle=True)
    eeg = data["eeg"][:, 0]
    sr = int(cfg["sample_rate_hz"])
    freqs, times, zxx = stft(eeg, fs=sr, nperseg=256, noverlap=192)
    power_db = 20.0 * np.log10(np.abs(zxx) + 1e-6)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    im = ax.imshow(
        power_db,
        aspect="auto",
        origin="lower",
        extent=[float(times.min()), float(times.max()), float(freqs.min()), float(freqs.max())],
        cmap="magma",
    )
    style_axes(ax)
    ax.set_xlim(float(times.min()), float(times.max()))
    ax.set_ylim(0, 60)
    add_phase_bands(ax, [float(x) for x in np.cumsum(cfg["data"]["phase_durations_s"])[:-1]])
    for y_band, label in [(13, "alpha"), (30, "beta")]:
        ax.axhline(y_band, color="#ffffff", linewidth=0.8, alpha=0.55)
        ax.text(float(times.max()) - 0.03 * float(times.max() - times.min()), y_band + 1.4, label, ha="right", va="bottom", fontsize=7.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    add_corner_label(ax, "channel 01")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    style_colorbar(cbar, "dB")
    save_figure(fig, out_dir / "eeg_spectrogram.png")


def fig3_synergy_weights(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    syn = np.load(run_dir / "artifacts" / "nmf_synergy.npz")
    weights = syn["W"]
    muscles = abbreviate_muscles(cfg["signals"]["emg"]["muscles"])

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    im = ax.imshow(weights, aspect="auto", cmap="viridis")
    style_axes(ax)
    ax.set_yticks(np.arange(weights.shape[0]))
    ax.set_yticklabels([f"S{i+1}" for i in range(weights.shape[0])])
    ax.set_xticks(np.arange(weights.shape[1]))
    ax.set_xticklabels(muscles)
    add_corner_label(ax, "NMF basis")
    peak_cols = np.argmax(weights, axis=1)
    ax.scatter(peak_cols, np.arange(weights.shape[0]), s=30, facecolor="#ffffff", edgecolor="#1d1a17", linewidth=0.8, zorder=3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    style_colorbar(cbar, "weight")
    save_figure(fig, out_dir / "synergy_weights.png")


def fig4_synergy_activation_heatmap(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    syn = np.load(run_dir / "artifacts" / "nmf_synergy.npz")
    weights = syn["W"]
    wt = weights.T
    trials = load_trials(run_dir)

    n_bins = 100
    activations = np.zeros((weights.shape[0], n_bins), dtype=np.float32)
    for tp in trials:
        data = np.load(tp, allow_pickle=True)
        env = data["emg_env"].astype(np.float32)
        basis = np.linalg.lstsq(wt, env.T, rcond=None)[0].T
        basis = np.clip(basis, 0.0, None)
        activations += resample_to_phase(basis, data["phase"], n_bins=n_bins).T
    activations /= max(1, len(trials))

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    im = ax.imshow(activations, aspect="auto", cmap="inferno")
    style_axes(ax)
    ax.set_yticks(np.arange(weights.shape[0]))
    ax.set_yticklabels([f"S{i+1}" for i in range(weights.shape[0])])
    ax.set_xticks([0, 25, 50, 75, 99])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    for x in [25, 50, 75]:
        ax.axvline(x - 0.5, color="white", linewidth=0.8, alpha=0.55)
    ax.set_xlabel("phase (%)")
    peak_bins = np.argmax(activations, axis=1)
    ax.scatter(peak_bins, np.arange(weights.shape[0]), s=20, facecolor="#ffffff", edgecolor="#1d1a17", linewidth=0.7, zorder=3)
    ax_phase = inset_axes(ax, width="100%", height="14%", loc="upper center", borderpad=0.8)
    phase_strip = np.arange(4, dtype=np.float32)[None, :]
    ax_phase.imshow(
        phase_strip,
        aspect="auto",
        interpolation="nearest",
        cmap=mcolors.ListedColormap(PHASE_COLORS),
        extent=[0.0, 100.0, 0.0, 1.0],
        vmin=0,
        vmax=3,
    )
    style_axes(ax_phase, hide_ticks=True)
    for x, label in zip([12.5, 37.5, 62.5, 87.5], ["prep", "momentum", "extend", "stable"]):
        ax_phase.text(x, 0.5, label, ha="center", va="center", fontsize=7.6, color="#000000")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    style_colorbar(cbar, "a.u.")
    save_figure(fig, out_dir / "synergy_activation_heatmap.png")


def fig5_fusion_ablation(run_dir: Path, out_dir: Path) -> None:
    metrics: dict[str, dict[str, Any]] = {}
    for modality in ["eeg", "emg", "fusion"]:
        path = run_dir / "artifacts" / f"phase_metrics_{modality}.json"
        if path.exists():
            metrics[modality] = json.loads(path.read_text())

    labels = [modality.upper() for modality in ["eeg", "emg", "fusion"] if modality in metrics]
    values = [metrics[modality].get("test_macro_f1_smoothed", metrics[modality]["test_macro_f1"]) for modality in ["eeg", "emg", "fusion"] if modality in metrics]
    colors = [MODALITY_COLORS[modality] for modality in ["eeg", "emg", "fusion"] if modality in metrics]

    y = np.arange(len(labels), dtype=np.float32)[::-1]
    chance = 0.25
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    style_axes(ax)
    ax.axvline(chance, color="#6d6a67", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
    ax.axvspan(0.0, chance, color=MODALITY_COLORS["neutral"], alpha=0.05, linewidth=0, zorder=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#ddd2c4", linewidth=0.7, alpha=0.45)

    eeg_val = values[0] if values else 0.0
    for yi, value, color, label in zip(y, values, colors, labels):
        ax.hlines(yi, chance, value, color=color, linewidth=2.8, alpha=0.85, zorder=1)
        ax.scatter([value], [yi], s=170, color="#ffffff", edgecolor=color, linewidth=2.1, zorder=3)
        ax.scatter([value], [yi], s=34, color=color, edgecolor="none", zorder=4)
        ax.text(0.02, yi, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=10.6, color="#000000")
        ax.text(value + 0.018, yi, f"{value:.3f}", ha="left", va="center", fontsize=10.0, color="#000000")
        if label != "EEG":
            ax.text(
                min(value + 0.13, 0.965),
                yi,
                f"+{value - eeg_val:.3f}",
                ha="left",
                va="center",
                fontsize=8.1,
                color="#000000",
            )

    ax.text(chance, y[0] + 0.58, "chance", ha="center", va="bottom", fontsize=8.1, color="#000000")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("Macro-F1")
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_corner_label(ax, "phase decoding")
    save_figure(fig, out_dir / "fusion_ablation.png")


def fig6_torque_overlay(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    device = get_device()
    segments = make_segments(run_dir, cfg, modality="fusion")
    x, y = segments["X"], segments["y_tau"]
    groups = select_split_groups(segments, cfg)
    _, _, test_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 2, groups=groups)

    scaler = StandardScaler1D.from_dict(json.loads((run_dir / "artifacts" / "scaler_fusion_torque.json").read_text()))
    target_scaler_path = run_dir / "artifacts" / "target_scaler_fusion_torque.json"
    target_scaler = StandardScalerTarget.from_dict(json.loads(target_scaler_path.read_text())) if target_scaler_path.exists() else None
    xs = scaler.transform(x)
    model = load_model_checkpoint(run_dir / "artifacts" / "torque_model_fusion.pt", cfg, task="torque", modality="fusion", device=device)
    model.eval()

    idx = test_idx[:600]
    with torch.no_grad():
        pred = model(torch.from_numpy(xs[idx]).to(device)).cpu().numpy()
    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred)
    truth = y[idx]
    err = np.abs(pred - truth)
    resid = pred - truth
    score_rmse = rmse(pred, truth)
    corr = float(np.corrcoef(truth, pred)[0, 1]) if len(truth) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    lo, hi = quantile_limits(np.concatenate([truth, pred]))
    pad = 0.05 * (hi - lo + 1e-6)
    low_err = err <= np.quantile(err, 0.35)
    high_err = err >= np.quantile(err, 0.8)

    style_axes(ax)
    band_x = np.linspace(lo - pad, hi + pad, 200)
    band_w = 0.10 * (hi - lo + 1e-6)
    ax.fill_between(band_x, band_x - band_w, band_x + band_w, color=MODALITY_COLORS["accent"], alpha=0.12, linewidth=0, zorder=0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="-", color="#b69d82", linewidth=1.6, alpha=0.95, zorder=1)
    ax.scatter(
        truth,
        pred,
        s=60,
        color="#342821",
        alpha=0.84,
        edgecolors="#ffffff",
        linewidths=0.65,
        zorder=2,
    )
    ax.scatter(
        truth[low_err],
        pred[low_err],
        s=86,
        color="#efb57e",
        alpha=0.55,
        edgecolors="#ffffff",
        linewidths=0.55,
        zorder=3,
    )
    ax.scatter(
        truth[high_err],
        pred[high_err],
        s=92,
        facecolors="none",
        edgecolors="#7a4f3f",
        linewidths=1.0,
        alpha=0.7,
        zorder=4,
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("target")
    ax.set_ylabel("pred")
    ax.set_xticks(np.round(np.linspace(lo, hi, 4), 2))
    ax.set_yticks(np.round(np.linspace(lo, hi, 4), 2))
    ax.text(0.03, 0.96, "fusion fit", transform=ax.transAxes, ha="left", va="top", fontsize=9.0, color="#000000")
    ax.text(0.98, 0.03, f"rmse {score_rmse:.3f}  |  r {corr:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2, color="#000000")
    save_figure(fig, out_dir / "torque_overlay.png")


def fig7_robustness_curve(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    device = get_device()
    scaler = StandardScaler1D.from_dict(json.loads((run_dir / "artifacts" / "scaler_fusion_torque.json").read_text()))
    target_scaler_path = run_dir / "artifacts" / "target_scaler_fusion_torque.json"
    target_scaler = StandardScalerTarget.from_dict(json.loads(target_scaler_path.read_text())) if target_scaler_path.exists() else None
    model = load_model_checkpoint(run_dir / "artifacts" / "torque_model_fusion.pt", cfg, task="torque", modality="fusion", device=device)
    model.eval()

    seg = make_segments(run_dir, cfg, modality="fusion")
    x, y = seg["X"], seg["y_tau"]
    groups = select_split_groups(seg, cfg)
    _, _, test_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 2, groups=groups)
    base = x[test_idx[:1200]].copy()
    target = y[test_idx[:1200]].copy()

    eeg_ch = int(cfg["signals"]["eeg"]["n_channels"])
    multipliers = np.asarray(cfg["robustness"]["emg_noise_multipliers"], dtype=np.float32)
    rng = np.random.default_rng(int(cfg["seed"]) + 999)
    rmse_mean: list[float] = []
    rmse_std: list[float] = []

    for multiplier in multipliers:
        samples = []
        for _ in range(6):
            noisy = base.copy()
            emg = noisy[:, eeg_ch:, :]
            noise = rng.normal(0.0, float(multiplier - 1.0) * 0.15, size=emg.shape).astype(np.float32)
            noisy[:, eeg_ch:, :] = emg + noise
            xs = scaler.transform(noisy)
            with torch.no_grad():
                pred = model(torch.from_numpy(xs).to(device)).cpu().numpy()
            if target_scaler is not None:
                pred = target_scaler.inverse_transform(pred)
            samples.append(rmse(pred, target))
        rmse_mean.append(float(np.mean(samples)))
        rmse_std.append(float(np.std(samples)))

    y_low = np.asarray(rmse_mean) - np.asarray(rmse_std)
    y_high = np.asarray(rmse_mean) + np.asarray(rmse_std)

    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    style_axes(ax)
    baseline = rmse_mean[0]
    y_min = float(min(y_low.min(), baseline) - 0.02)
    y_max = float(y_high.max() + 0.02)
    gradient = np.linspace(0.0, 1.0, 256)[:, None]
    amber = mcolors.LinearSegmentedColormap.from_list("amber_fade", ["#ffffff", "#f0d7bd", "#c7834d"])
    ax.imshow(
        gradient,
        aspect="auto",
        origin="lower",
        extent=[float(multipliers.min()) - 0.18, float(multipliers.max()) + 0.18, y_min, y_max],
        cmap=amber,
        alpha=0.22,
        zorder=0,
    )
    for left, right in zip(multipliers[:-1], multipliers[1:]):
        ax.axvspan(left, right, color=MODALITY_COLORS["accent"], alpha=0.028, linewidth=0)
    ax.fill_between(multipliers, y_low, y_high, color=MODALITY_COLORS["fusion"], alpha=0.22, linewidth=0, zorder=1)
    ax.fill_between(multipliers, y_min, rmse_mean, color=MODALITY_COLORS["fusion"], alpha=0.08, linewidth=0, zorder=1)
    ax.plot(multipliers, rmse_mean, color=MODALITY_COLORS["fusion"], linewidth=2.8, zorder=3)
    ax.axhline(baseline, color="#8e7660", linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)
    bubble = 280.0 * (np.asarray(rmse_mean) - baseline + 0.08)
    bubble = np.clip(bubble, 70.0, None)
    ax.scatter(multipliers, rmse_mean, s=bubble, color=MODALITY_COLORS["fusion"], alpha=0.14, zorder=2, edgecolors="none")
    ax.scatter(multipliers, rmse_mean, s=54, color="#fbf8f2", edgecolor=MODALITY_COLORS["fusion"], linewidth=1.8, zorder=4)
    ax.text(multipliers[0], baseline - 0.015, "base", ha="center", va="top", fontsize=8.2, color="#000000")
    ax.scatter([multipliers[-1]], [rmse_mean[-1]], s=160, color="#ffffff", edgecolor=MODALITY_COLORS["fusion"], linewidth=2.2, zorder=5)
    ax.annotate(
        f"+{rmse_mean[-1] - baseline:.2f}",
        xy=(multipliers[-1], rmse_mean[-1]),
        xytext=(multipliers[-1] - 0.10, rmse_mean[-1] + 0.018),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#000000",
        arrowprops={"arrowstyle": "-", "color": "#9b7d67", "linewidth": 0.9},
    )
    ax.set_xlabel("EMG noise x")
    ax.set_ylabel("RMSE")
    ax.set_xticks(multipliers)
    ax.set_xlim(float(multipliers.min()) - 0.15, float(multipliers.max()) + 0.15)
    ax.set_ylim(y_min, y_max)
    ax.text(0.03, 0.96, "fusion robustness", transform=ax.transAxes, ha="left", va="top", fontsize=9.0, color="#000000")
    save_figure(fig, out_dir / "robustness_curve.png")


def fig8_eeg_band_profile(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    trials = load_trials(run_dir)
    sr = int(cfg["sample_rate_hz"])
    n_channels = int(cfg["signals"]["eeg"]["n_channels"])
    spectra = []
    band_rows = []
    for tp in trials:
        data = np.load(tp, allow_pickle=True)
        eeg = data["eeg"].astype(np.float32)
        for ch in range(n_channels):
            freqs, pxx = welch(eeg[:, ch], fs=sr, nperseg=min(len(eeg), 512))
            spectra.append(pxx)
            mu_mask = (freqs >= 8.0) & (freqs <= 13.0)
            beta_mask = (freqs >= 13.0) & (freqs <= 30.0)
            band_rows.append([
                float(trapezoid(pxx[mu_mask], freqs[mu_mask]) + 1e-12),
                float(trapezoid(pxx[beta_mask], freqs[beta_mask]) + 1e-12),
            ])
    spectra_arr = np.asarray(spectra, dtype=np.float32)
    band_arr = np.asarray(band_rows, dtype=np.float32)
    mean_spec = spectra_arr.mean(axis=0)
    std_spec = spectra_arr.std(axis=0)

    fig = plt.figure(figsize=(8.6, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.85], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax_band = fig.add_subplot(gs[0, 1])

    style_axes(ax)
    ax.axvspan(8.0, 13.0, color=MODALITY_COLORS["eeg"], alpha=0.08, linewidth=0)
    ax.axvspan(13.0, 30.0, color=MODALITY_COLORS["accent"], alpha=0.10, linewidth=0)
    ax.plot(freqs, mean_spec, color=MODALITY_COLORS["eeg"], linewidth=2.0)
    ax.fill_between(freqs, np.maximum(mean_spec - std_spec, 0.0), mean_spec + std_spec, color=MODALITY_COLORS["eeg"], alpha=0.15, linewidth=0)
    ax.set_xlim(4.0, 40.0)
    ax.set_xlabel("Hz")
    ax.set_ylabel("PSD")
    add_corner_label(ax, "band profile")

    style_axes(ax_band)
    mu_vals = band_arr[:, 0]
    beta_vals = band_arr[:, 1]
    positions = np.arange(2, dtype=np.float32)
    viol = ax_band.violinplot([mu_vals, beta_vals], positions=positions, widths=0.82, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(viol["bodies"], [MODALITY_COLORS["eeg"], MODALITY_COLORS["accent"]]):
        body.set_facecolor(color)
        body.set_alpha(0.22)
        body.set_edgecolor("none")
    ax_band.scatter(np.full_like(mu_vals, positions[0]), mu_vals, s=12, color=MODALITY_COLORS["eeg"], alpha=0.18, edgecolors="none")
    ax_band.scatter(np.full_like(beta_vals, positions[1]), beta_vals, s=12, color=MODALITY_COLORS["accent"], alpha=0.18, edgecolors="none")
    ax_band.hlines([mu_vals.mean(), beta_vals.mean()], positions - 0.18, positions + 0.18, color="#1d1a17", linewidth=1.4)
    ax_band.set_xticks(positions)
    ax_band.set_xticklabels(["mu", "beta"])
    ax_band.set_ylabel("band power")
    add_corner_label(ax_band, "distribution")
    save_figure(fig, out_dir / "eeg_band_profile.png")


def fig9_robustness_delta_panels(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    device = get_device()
    scaler = StandardScaler1D.from_dict(json.loads((run_dir / "artifacts" / "scaler_fusion_torque.json").read_text()))
    target_scaler_path = run_dir / "artifacts" / "target_scaler_fusion_torque.json"
    target_scaler = StandardScalerTarget.from_dict(json.loads(target_scaler_path.read_text())) if target_scaler_path.exists() else None
    model = load_model_checkpoint(run_dir / "artifacts" / "torque_model_fusion.pt", cfg, task="torque", modality="fusion", device=device)
    model.eval()

    seg = make_segments(run_dir, cfg, modality="fusion")
    x, y = seg["X"], seg["y_tau"]
    groups = select_split_groups(seg, cfg)
    _, _, test_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 2, groups=groups)
    base = x[test_idx[:1200]].copy()
    target = y[test_idx[:1200]].copy()
    eeg_ch = int(cfg["signals"]["eeg"]["n_channels"])
    multipliers = np.asarray(cfg["robustness"]["emg_noise_multipliers"], dtype=np.float32)
    rng = np.random.default_rng(int(cfg["seed"]) + 1001)
    deltas = []
    rmse_mean = []
    baseline = None
    for multiplier in multipliers:
        noisy = base.copy()
        if multiplier > 1.0:
            noise = rng.normal(0.0, float(multiplier - 1.0) * 0.15, size=noisy[:, eeg_ch:, :].shape).astype(np.float32)
            noisy[:, eeg_ch:, :] += noise
        xs = scaler.transform(noisy)
        with torch.no_grad():
            pred = model(torch.from_numpy(xs).to(device)).cpu().numpy()
        if target_scaler is not None:
            pred = target_scaler.inverse_transform(pred)
        value = rmse(pred, target)
        rmse_mean.append(value)
        if baseline is None:
            baseline = value
        deltas.append(100.0 * (value - baseline) / max(baseline, 1e-6))

    metrics = {}
    for modality in ["eeg", "emg", "fusion"]:
        path = run_dir / "artifacts" / f"phase_metrics_{modality}.json"
        metrics[modality] = json.loads(path.read_text()) if path.exists() else {}
    phase_values = [float(metrics[m].get("test_macro_f1", 0.0)) for m in ["eeg", "emg", "fusion"]]
    phase_delta = np.asarray(phase_values) - phase_values[1]

    fig = plt.figure(figsize=(8.6, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.9], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    style_axes(ax, grid_axis="y")
    style_axes(ax_phase, grid_axis="y")

    ax.bar(np.arange(len(multipliers)), deltas, color=MODALITY_COLORS["fusion"], alpha=0.82, width=0.66)
    ax.axhline(0.0, color="#b79f86", linewidth=1.0)
    ax.set_xticks(np.arange(len(multipliers)))
    ax.set_xticklabels([f"x{m:.1f}" for m in multipliers])
    ax.set_ylabel("delta %")
    add_corner_label(ax, "torque noise drift")

    y = np.arange(3)[::-1]
    ax_phase.barh(y, phase_delta, color=[MODALITY_COLORS["eeg"], MODALITY_COLORS["emg"], MODALITY_COLORS["fusion"]], alpha=0.78, height=0.58)
    ax_phase.axvline(0.0, color="#b79f86", linewidth=1.0)
    ax_phase.set_yticks(y)
    ax_phase.set_yticklabels(["EEG", "EMG", "FUSION"])
    ax_phase.set_xlabel("delta vs EMG")
    add_corner_label(ax_phase, "phase gain")
    save_figure(fig, out_dir / "robustness_delta_panels.png")


def fig10_synergy_summary_grid(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    syn = np.load(run_dir / "artifacts" / "nmf_synergy.npz")
    weights = syn["W"]
    muscles = abbreviate_muscles(cfg["signals"]["emg"]["muscles"])
    totals = weights.sum(axis=1)
    peaks = np.argmax(weights, axis=1)
    ranked = np.argsort(weights, axis=1)[:, ::-1]

    fig = plt.figure(figsize=(8.4, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[0.9, 1.1], hspace=0.22, wspace=0.28)
    ax_tot = fig.add_subplot(gs[:, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_peak = fig.add_subplot(gs[1, 1])
    style_axes(ax_tot, grid_axis="x")
    style_axes(ax_rank)
    style_axes(ax_peak)

    order = np.arange(weights.shape[0])[::-1]
    ax_tot.barh(order, totals, color=MODALITY_COLORS["accent"], alpha=0.82, height=0.58)
    ax_tot.set_yticks(order)
    ax_tot.set_yticklabels([f"S{i+1}" for i in order[::-1]][::-1])
    ax_tot.set_xlabel("total weight")
    add_corner_label(ax_tot, "synergy mass")

    rank_plot = np.take_along_axis(weights, ranked[:, :3], axis=1)
    for row_idx in range(rank_plot.shape[0]):
        ax_rank.plot(np.arange(1, 4), rank_plot[row_idx], marker="o", linewidth=1.6, color=PHASE_COLORS[row_idx % len(PHASE_COLORS)], alpha=0.9)
    ax_rank.set_xticks([1, 2, 3])
    ax_rank.set_xticklabels(["1", "2", "3"])
    ax_rank.set_ylabel("weight")
    add_corner_label(ax_rank, "top ranks")

    peak_matrix = np.zeros((weights.shape[0], weights.shape[1]), dtype=np.float32)
    for row_idx, peak in enumerate(peaks):
        peak_matrix[row_idx, peak] = weights[row_idx, peak]
    im = ax_peak.imshow(peak_matrix, aspect="auto", cmap="cividis")
    ax_peak.set_xticks(np.arange(len(muscles)))
    ax_peak.set_xticklabels(muscles)
    ax_peak.set_yticks(np.arange(weights.shape[0]))
    ax_peak.set_yticklabels([f"S{i+1}" for i in range(weights.shape[0])])
    add_corner_label(ax_peak, "dominant muscle")
    cbar = fig.colorbar(im, ax=ax_peak, fraction=0.046, pad=0.03)
    style_colorbar(cbar, "peak")
    save_figure(fig, out_dir / "synergy_summary_grid.png")


def fig11_torque_error_distribution(run_dir: Path, cfg: dict[str, Any], out_dir: Path) -> None:
    device = get_device()
    segments = make_segments(run_dir, cfg, modality="fusion")
    x, y = segments["X"], segments["y_tau"]
    groups = select_split_groups(segments, cfg)
    _, _, test_idx = split_indices(len(y), cfg, seed=int(cfg["seed"]) + 2, groups=groups)
    idx = test_idx[:800]

    scaler = StandardScaler1D.from_dict(json.loads((run_dir / "artifacts" / "scaler_fusion_torque.json").read_text()))
    target_scaler_path = run_dir / "artifacts" / "target_scaler_fusion_torque.json"
    target_scaler = StandardScalerTarget.from_dict(json.loads(target_scaler_path.read_text())) if target_scaler_path.exists() else None
    xs = scaler.transform(x[idx])
    model = load_model_checkpoint(run_dir / "artifacts" / "torque_model_fusion.pt", cfg, task="torque", modality="fusion", device=device)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xs).to(device)).cpu().numpy()
    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred)
    truth = y[idx]
    resid = pred - truth
    abs_err = np.abs(resid)
    range_stats = torque_error_by_range(truth, pred)

    fig = plt.figure(figsize=(8.8, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.24)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_range = fig.add_subplot(gs[0, 1])
    style_axes(ax_hist, grid_axis="y")
    style_axes(ax_range, grid_axis="x")

    bins = np.linspace(float(resid.min()), float(resid.max()), 28)
    ax_hist.hist(resid, bins=bins, color=MODALITY_COLORS["fusion"], alpha=0.22, edgecolor="none", density=True)
    if len(resid) > 4:
        support = np.linspace(float(resid.min()), float(resid.max()), 200)
        kde = gaussian_kde(resid)
        ax_hist.plot(support, kde(support), color=MODALITY_COLORS["fusion"], linewidth=2.0)
    ax_hist.axvline(0.0, color="#1d1a17", linestyle="--", linewidth=1.0)
    add_corner_label(ax_hist, "residual density")
    ax_hist.set_xlabel("pred - target")
    ax_hist.set_ylabel("density")

    names = ["low", "mid", "high"]
    values = [range_stats[name]["rmse"] or 0.0 for name in names]
    counts = [range_stats[name]["count"] for name in names]
    ax_range.barh(np.arange(len(names))[::-1], values, color=[MODALITY_COLORS["accent"], "#b48a61", MODALITY_COLORS["fusion"]], alpha=0.82, height=0.56)
    ax_range.set_yticks(np.arange(len(names))[::-1])
    ax_range.set_yticklabels([f"{name} ({count})" for name, count in zip(names, counts)][::-1])
    ax_range.set_xlabel("RMSE")
    add_corner_label(ax_range, "target bands")
    save_figure(fig, out_dir / "torque_error_distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    out_dir = ensure_dir(run_dir / "figures")
    style_info = apply_scientific_luxe_style("figure")

    for old_name in [
        "fig1_coupling_heatmap.png",
        "fig2_eeg_spectrogram.png",
        "fig3_synergy_weights.png",
        "fig4_synergy_activation_heatmap.png",
        "fig5_fusion_ablation.png",
        "fig6_torque_overlay.png",
        "fig7_robustness_curve.png",
        "coupling_heatmap.png",
        "eeg_spectrogram.png",
        "synergy_weights.png",
        "synergy_activation_heatmap.png",
        "fusion_ablation.png",
        "torque_overlay.png",
        "robustness_curve.png",
        "eeg_band_profile.png",
        "robustness_delta_panels.png",
        "synergy_summary_grid.png",
        "torque_error_distribution.png",
    ]:
        (out_dir / old_name).unlink(missing_ok=True)

    fig1_coupling_heatmap(run_dir, cfg, out_dir)
    fig2_eeg_spectrogram(run_dir, cfg, out_dir)
    fig3_synergy_weights(run_dir, cfg, out_dir)
    fig4_synergy_activation_heatmap(run_dir, cfg, out_dir)
    fig5_fusion_ablation(run_dir, out_dir)
    fig6_torque_overlay(run_dir, cfg, out_dir)
    fig7_robustness_curve(run_dir, cfg, out_dir)
    fig8_eeg_band_profile(run_dir, cfg, out_dir)
    fig9_robustness_delta_panels(run_dir, cfg, out_dir)
    fig10_synergy_summary_grid(run_dir, cfg, out_dir)
    fig11_torque_error_distribution(run_dir, cfg, out_dir)
    for pdf in out_dir.glob("*.pdf"):
        pdf.unlink(missing_ok=True)
    print(f"Using font: {style_info['font_primary']}")
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
