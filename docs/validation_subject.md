# STS Digital Twin Validation Report

This run is positioned as an in-silico STS development bench for multimodal phase decoding and assistive torque prediction.

## Snapshot

- Split strategy: `subject`
- Subjects: `30`
- Trials: `240`
- Segments: `2160`
- Segment window: `0.50s` with `0.25s` stride
- Split counts: train `1584`, val `216`, test `360`
- Group counts: train `22`, val `3`, test `5`
- Group overlap: train/val `0`, train/test `0`, val/test `0`
- Synergy VAF: `0.645` with `K=3` synergies over `6` muscles

## Phase Decoding

| Modality | Test Macro-F1 | Test Acc | Majority F1 | Majority Acc | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| EEG | 0.375 | 0.633 | 0.154 | 0.444 | recomputed |
| EMG | 0.785 | 0.806 | 0.154 | 0.444 | recomputed |
| FUSION | 0.784 | 0.803 | 0.154 | 0.444 | recomputed |

## Torque Prediction

| Model | Test RMSE | Mean Baseline RMSE | Approx. R2 | Corr | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Fusion | 0.840 | 1.022 | 0.314 | 0.587 | recomputed |

## Interpretation

- This benchmark reflects controller-development conditions, not a clinical deployment claim.
- Subject-wise or trial-wise splits are the preferred protocol for any headline number.
- The data source remains in-silico, but the evaluation and reporting are structured to match a serious offline R&D workflow.
