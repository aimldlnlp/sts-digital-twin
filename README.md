# STS Digital Twin Pipeline

> A multimodal sit-to-stand digital twin for offline phase decoding, synergy analysis, and assistive torque benchmarking.

<p align="center">
  <img src="assets/showcase/fig_skeleton_grid_2x2.png" alt="STS skeleton snapshots" width="48%" />
  <img src="assets/showcase/fig3_synergy_weights.png" alt="EMG synergy weights" width="48%" />
</p>

This repository packages an end-to-end STS virtual-lab workflow:

- generate synchronized 3D stick-skeleton kinematics, EEG, EMG, phase labels, and exoskeleton torque targets
- fit interpretable EMG synergies with NMF
- train EEG-only, EMG-only, and fusion phase decoders plus a fusion torque regressor
- export paper-style figures, demo videos, and a validation report for each run

The data source is in-silico. The value of the project is a reproducible offline development bench for multimodal controller R&D, ablation studies, and reporting discipline before deployment on real recordings.

## Why This Repo Reads Like A Serious Project

- Subject-wise evaluation is the default protocol in [`configs/default.yaml`](configs/default.yaml), so headline metrics are not inflated by overlapping windows from the same subject.
- Every run emits a structured artifact set: raw trials, trained models, scalers, figures, videos, and a Markdown/JSON validation report.
- The pipeline is opinionated around benchmark hygiene: ablations, robustness curves, synergy summaries, and a dedicated report generator.

## Latest Benchmark Snapshot

Reference report: [`docs/validation_subject.md`](docs/validation_subject.md)

Subject-wise protocol:
- 30 subjects
- 240 trials
- 2160 segments
- 22 train / 3 val / 5 test subjects
- zero subject overlap across splits

| Task | Model | Result |
| --- | --- | ---: |
| Phase decoding | EEG | Macro-F1 `0.375`, Acc `0.633` |
| Phase decoding | EMG | Macro-F1 `0.785`, Acc `0.806` |
| Phase decoding | Fusion | Macro-F1 `0.784`, Acc `0.803` |
| Torque prediction | Fusion | RMSE `0.840`, approx. R2 `0.314`, corr `0.587` |
| Synergy decomposition | NMF | VAF `0.645` with `K=3` synergies |

These numbers reflect an offline in-silico benchmark, not a clinical claim.

## Pipeline

1. Generate parametric STS trials with synchronized kinematics, EEG, EMG, phase labels, and torque targets.
2. Factorize muscle activity into interpretable synergies.
3. Train multimodal phase and torque models under a configurable split strategy.
4. Render figures, videos, and a validation report for the run.

## Run Everything

```bash
python run_all.py --config configs/default.yaml
```

Outputs are written to `outputs/<run_id>/`:

- `data_raw/` generated trial `.npz` files
- `artifacts/` models, scalers, and NMF outputs
- `figures/` paper-style figures
- `videos/` multimodal and ablation MP4s
- `reports/` validation report in Markdown and JSON

## Step-By-Step

```bash
python scripts/01_generate_data.py --config configs/default.yaml
python scripts/02_fit_synergy.py --run_dir outputs/<run_id>
python scripts/03_train_phase.py --run_dir outputs/<run_id> --modality fusion
python scripts/03_train_phase.py --run_dir outputs/<run_id> --modality eeg
python scripts/03_train_phase.py --run_dir outputs/<run_id> --modality emg
python scripts/04_train_torque.py --run_dir outputs/<run_id>
python scripts/05_make_figures.py --run_dir outputs/<run_id>
python scripts/06_render_skeleton.py --run_dir outputs/<run_id>
python scripts/07_make_video.py --run_dir outputs/<run_id> --trial_index 0
python scripts/08_make_video_ablation.py --run_dir outputs/<run_id> --trial_index 0
python scripts/09_make_validation_report.py --run_dir outputs/<run_id>
```

To recompute a fair benchmark for an existing run under a specific protocol:

```bash
python scripts/09_make_validation_report.py \
  --run_dir outputs/<run_id> \
  --split_strategy subject \
  --recompute
```

## Evaluation Protocol

The default config uses:

- `split_strategy: subject`
- `segment_len: 128` samples
- `segment_stride: 64` samples
- `train / val / test: 0.75 / 0.10 / 0.15`

This keeps the reported metrics closer to a real generalization test than segment-level random splitting.

## Example Artifacts

- Validation report: [`docs/validation_subject.md`](docs/validation_subject.md)
- Skeleton showcase: [`assets/showcase/fig_skeleton_grid_2x2.png`](assets/showcase/fig_skeleton_grid_2x2.png)
- Synergy showcase: [`assets/showcase/fig3_synergy_weights.png`](assets/showcase/fig3_synergy_weights.png)
- Coupling heatmap: [`assets/showcase/fig1_coupling_heatmap.png`](assets/showcase/fig1_coupling_heatmap.png)
- Full figures, reports, and videos are generated under `outputs/<run_id>/` during local runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch works on CPU and can use CUDA if you install the matching wheel for your system. MP4 rendering requires `ffmpeg` on `PATH`.

## Intended Use

Use this project for:

- offline algorithm development
- multimodal ablation studies
- digital twin demos and internal technical showcases
- preclinical benchmarking before real-data integration

Do not use this repository to imply validated performance on real human sensor recordings without an external calibration and evaluation stage.
