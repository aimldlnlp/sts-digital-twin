# STS Digital Twin Pipeline

> A paper-style multimodal sit-to-stand benchmark for phase decoding, synergy analysis, torque prediction, and stress-tested fusion behavior.

<p align="center">
  <img src="assets/showcase/v5/videos/trial_000_multimodal.gif" alt="Multimodal STS benchmark overview" width="100%" />
</p>

This repository packages a full offline R&D bench around a synthetic sit-to-stand task: synchronized kinematics, EEG, EMG, phase labels, assistive torque targets, model training, validation reporting, and a dense visual suite. The published snapshot uses `v5` (`sts_20260420_215231`) as the canonical result because it is the best overall clean/tradeoff run to date.

Reference report: [`docs/validation_subject.md`](docs/validation_subject.md)
Curated showcase media: [`assets/showcase/v5`](assets/showcase/v5)

## Benchmark Snapshot

| Task | Model | Result |
| --- | --- | ---: |
| Phase decoding | EEG | Macro-F1 `0.451`, Acc `0.464`, Balanced Acc `0.457` |
| Phase decoding | EMG | Macro-F1 `0.533`, Acc `0.556`, Balanced Acc `0.588` |
| Phase decoding | Fusion | Macro-F1 `0.714`, Acc `0.689`, Balanced Acc `0.703` |
| Torque prediction | EMG | RMSE `0.925`, R2 `0.159`, Corr `0.533` |
| Torque prediction | Fusion | RMSE `0.894`, R2 `0.215`, Corr `0.544` |
| Synergy decomposition | NMF | VAF `0.695` with `K=3` synergies |

Published `v5` highlights:
- `best_clean: true`
- `best_tradeoff: true`
- `best_robust_torque: true`
- fusion phase gain over EMG: `+0.180` Macro-F1 and `+0.133` accuracy
- fusion torque gain over EMG: `-0.031` RMSE

What still fails:
- phase fusion under `emg_noise`: `0.210` Macro-F1
- phase fusion under `drop_eeg`: `0.510` Macro-F1
- phase fusion under `drop_emg`: `0.070` Macro-F1
- torque fusion high-target RMSE: `1.419`
- torque fusion under `drop_emg`: `1.642` RMSE

Interpretation:
the public snapshot favors the best overall clean/scientific tradeoff, not the single strongest robustness-only run. Earlier experiments pushed phase robustness harder, but `v5` is the most balanced benchmark result to publish.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --config configs/default.yaml
```

Generated runs are written to `outputs/<run_id>/` with:
- `data_raw/`
- `artifacts/`
- `figures/`
- `videos/`
- `reports/`

## Pipeline

1. Generate heterogeneous synthetic STS trials with synchronized kinematics, EEG, EMG, phase labels, and assistive torque targets.
2. Fit EMG synergies with NMF.
3. Train phase decoders for `eeg`, `emg`, and `fusion`.
4. Train torque regressors for `emg` and `fusion`.
5. Render paper-style figures, videos, and a validation report.

## Animated Gallery

<p align="center">
  <img src="assets/showcase/v5/videos/trial_000_ablation_phase.gif" alt="Phase ablation animation" width="49%" />
  <img src="assets/showcase/v5/videos/trial_000_stress_phase_robustness.gif" alt="Stress phase robustness animation" width="49%" />
</p>

<p align="center">
  <img src="assets/showcase/v5/videos/trial_000_torque_error_range.gif" alt="Torque error range animation" width="49%" />
  <img src="assets/showcase/v5/videos/trial_000_fusion_benchmark_montage.gif" alt="Fusion benchmark montage animation" width="49%" />
</p>

## Figure Gallery

<p align="center">
  <img src="assets/showcase/v5/figures/coupling_heatmap.png" alt="Cross-modal coupling heatmap" width="49%" />
  <img src="assets/showcase/v5/figures/eeg_spectrogram.png" alt="EEG spectrogram" width="49%" />
</p>

<p align="center">
  <img src="assets/showcase/v5/figures/eeg_band_profile.png" alt="EEG band profile" width="49%" />
  <img src="assets/showcase/v5/figures/fusion_ablation.png" alt="Fusion ablation figure" width="49%" />
</p>

<p align="center">
  <img src="assets/showcase/v5/figures/robustness_curve.png" alt="Robustness curve" width="49%" />
  <img src="assets/showcase/v5/figures/robustness_delta_panels.png" alt="Robustness delta panels" width="49%" />
</p>

<p align="center">
  <img src="assets/showcase/v5/figures/skeleton_grid_2x2.png" alt="Skeleton grid" width="49%" />
  <img src="assets/showcase/v5/figures/synergy_activation_heatmap.png" alt="Synergy activation heatmap" width="49%" />
</p>

<p align="center">
  <img src="assets/showcase/v5/figures/synergy_summary_grid.png" alt="Synergy summary grid" width="49%" />
  <img src="assets/showcase/v5/figures/synergy_weights.png" alt="Synergy weights" width="49%" />
</p>

<p align="center">
  <img src="assets/showcase/v5/figures/torque_error_distribution.png" alt="Torque error distribution" width="49%" />
  <img src="assets/showcase/v5/figures/torque_overlay.png" alt="Torque overlay" width="49%" />
</p>

## Why This Repo Exists

- serious offline benchmark framing, even though the data is synthetic
- reproducible subject-wise evaluation with explicit stress cases
- modality-aware fusion models instead of a toy early-fusion baseline only
- reportable artifacts that are usable for portfolio, benchmark, and ablation storytelling

## Tech Stack

- Python for orchestration and experiment scripting
- NumPy and SciPy for signal generation and feature processing
- PyTorch for phase decoding and torque regression
- Matplotlib for paper-style figure rendering
- ffmpeg for MP4 and GIF export
- YAML configs for reproducible benchmark settings

## Intended Use

Use this repo for:
- multimodal benchmark development
- offline controller prototyping
- ablation and robustness studies
- technical demo and portfolio presentation

Do not use it to claim validated clinical performance on real human recordings without external data collection and evaluation.
