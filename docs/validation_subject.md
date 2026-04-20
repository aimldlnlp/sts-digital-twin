# STS Digital Twin Validation Report

This run is positioned as an in-silico STS development bench for multimodal phase decoding and assistive torque prediction.

## Snapshot

- Split strategy: `subject`
- Subjects: `30`
- Trials: `240`
- Segments: `2160`
- Segment window: `0.50s` with `0.25s` stride
- Split counts: train `1584`, val `216`, test `360`
- Flags: clean `True`, robust-phase `False`, robust-torque `True`, tradeoff `True`
- Group counts: train `22`, val `3`, test `5`
- Group overlap: train/val `0`, train/test `0`, val/test `0`
- Synergy VAF: `0.695` with `K=3` synergies over `6` muscles

## Phase Decoding

| Modality | Test Macro-F1 | Smoothed Macro-F1 | Test Acc | Balanced Acc | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| EEG | 0.451 | 0.451 | 0.464 | 0.457 | artifacts |
| EMG | 0.533 | 0.533 | 0.556 | 0.588 | artifacts |
| FUSION | 0.714 | 0.714 | 0.689 | 0.703 | artifacts |

## Fusion Gain

- `delta_over_emg_macro_f1`: `0.180`
- `delta_over_emg_acc`: `0.133`
- `delta_over_emg_torque_rmse`: `-0.031`
- `phase_selection_score`: `0.614`
- `torque_selection_score`: `0.710`
- `phase_history_path`: `-`
- `torque_history_path`: `-`

## Stress Phase

| Model | Clean F1 | EMG Noise | Temporal Shift | Drop EEG | Drop EMG |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMG | 0.533 | 0.161 | 0.501 | - | - |
| FUSION | 0.714 | 0.210 | 0.675 | 0.510 | 0.070 |

## Torque Prediction

| Model | Test RMSE | MAE | Mean Baseline RMSE | Approx. R2 | Corr | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Fusion | 0.894 | 0.612 | 1.018 | 0.215 | 0.544 | artifacts |
| EMG | 0.925 | 0.658 | 1.018 | 0.159 | 0.533 | artifacts |

## Stress Torque

| Model | Clean RMSE | EMG Noise | Temporal Shift | Drop EEG | Drop EMG |
| --- | ---: | ---: | ---: | ---: | ---: |
| FUSION | 0.894 | 0.890 | 0.893 | 0.884 | 1.642 |
| EMG | 0.925 | 0.940 | 0.921 | - | - |

## Torque Error Range

- Low target RMSE: `0.503` over `126` segments
- Mid target RMSE: `0.352` over `114` segments
- High target RMSE: `1.419` over `120` segments

## Generalization Gap

- Phase clean gap (val-test): `-0.096`
- Phase EMG-noise gap (val-test): `0.437`
- Torque clean gap (test-val): `0.222`
- Torque EMG-noise gap (test-val): `0.219`

## Reference Comparison

| Reference | dPhase Clean | dPhase EMG Noise | dPhase Drop EEG | dTorque Clean RMSE | dHigh-Target RMSE | dPhase Selection | dTorque Selection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.011 | 0.067 | 0.162 | 0.018 | 0.034 | 0.105 | -0.233 |
| v4 | 0.026 | -0.293 | -0.013 | 0.033 | 0.084 | -0.002 | -0.222 |
| v5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Acceptance

- Phase clean >= 0.67: `True`
- Phase gain over EMG >= 0.10: `True`
- Phase EMG noise >= 0.40: `False`
- Phase drop EEG >= 0.50: `True`
- Torque clean RMSE <= 0.87: `False`
- Torque gain over EMG RMSE <= -0.05: `False`
- High-target RMSE <= 1.30: `False`

## Interpretation

- This benchmark reflects controller-development conditions, not a clinical deployment claim.
- The headline scientific question is whether fusion beats EMG under subject split and stays competitive under stress.
- The data source remains in-silico, but the evaluation and reporting are structured to match a serious offline R&D workflow.
