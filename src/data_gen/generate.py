\
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .sts_kinematics import generate_sts_kinematics, JOINT_NAMES, BONES
from .signals import generate_eeg, generate_emg
from .torque import compute_knee_torque_proxy

def _sample_phase_durations(cfg: Dict[str, Any], rng: np.random.Generator) -> list[float]:
    dur = float(cfg["data"]["duration_s"])
    base = np.asarray(cfg["data"]["phase_durations_s"], dtype=np.float32)
    jitter = float(cfg.get("augment", {}).get("phase_duration_jitter_s", 0.0))
    if jitter <= 0:
        return base.tolist()
    noisy = base + rng.normal(0.0, jitter, size=base.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.18, None)
    noisy *= dur / float(noisy.sum())
    return noisy.tolist()


def _subject_profile(cfg: Dict[str, Any], rng: np.random.Generator, sr: int) -> dict[str, float]:
    aug = cfg.get("augment", {})
    gain_std = float(aug.get("subject_gain_std", 0.0))
    lat_std_ms = float(aug.get("modality_latency_jitter_ms", 0.0))
    torque_gain_std = float(aug.get("torque_gain_std", 0.0))
    torque_bias_std = float(aug.get("torque_bias_std", 0.0))
    lat_scale = sr / 1000.0
    return {
        "eeg_gain": float(np.clip(rng.normal(1.0, gain_std), 0.55, 1.8)),
        "emg_gain": float(np.clip(rng.normal(1.0, gain_std), 0.55, 1.8)),
        "eeg_latency_samples": int(np.round(rng.normal(0.0, lat_std_ms) * lat_scale)),
        "emg_latency_samples": int(np.round(rng.normal(0.0, lat_std_ms) * lat_scale)),
        "torque_gain": float(np.clip(rng.normal(1.0, torque_gain_std), 0.6, 1.6)),
        "torque_bias": float(rng.normal(0.0, torque_bias_std)),
    }


def _trial_profile(rng: np.random.Generator) -> dict[str, float]:
    return {
        "eeg_trial_gain": float(np.clip(rng.normal(1.0, 0.08), 0.75, 1.25)),
        "emg_trial_gain": float(np.clip(rng.normal(1.0, 0.10), 0.70, 1.30)),
    }


def generate_dataset(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "data_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sr = int(cfg["sample_rate_hz"])
    n_subjects = int(cfg["data"]["n_subjects"])
    tps = int(cfg["data"]["trials_per_subject"])
    anth = cfg["digital_twin"]["anthropometry"]

    index = []
    trial_id = 0
    for s in range(n_subjects):
        subj_dir = raw_dir / f"subject_{s:03d}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        subject_rng = np.random.default_rng(int(cfg["seed"]) + 5000 + s)
        subject_profile = _subject_profile(cfg, subject_rng, sr)
        for tr in range(tps):
            seed = int(cfg["seed"]) + 1000*s + tr
            trial_rng = np.random.default_rng(seed + 77)
            trial_profile = _trial_profile(trial_rng)
            phase_durs = _sample_phase_durations(cfg, trial_rng)
            profile = subject_profile | trial_profile
            kin = generate_sts_kinematics(cfg, sr, seed=seed, anthropometry=anth, phase_durs_s=phase_durs)

            eeg = generate_eeg(cfg, kin["t"], kin["phase"], seed=seed+7, profile=profile)
            emg_raw, emg_env, W_true = generate_emg(cfg, kin["t"], kin["phase"], seed=seed+13, profile=profile)

            # knee angle is angles[:,1]
            tau_h, tau_exo, dyn_meta = compute_knee_torque_proxy(cfg, kin["angles"][:,1], kin["phase"], seed=seed+31, profile=profile)

            trial_path = subj_dir / f"trial_{tr:03d}.npz"
            np.savez_compressed(
                trial_path,
                t=kin["t"],
                phase=kin["phase"],
                joints=kin["joints"],
                angles=kin["angles"],
                eeg=eeg,
                emg_raw=emg_raw,
                emg_env=emg_env,
                tau_h=tau_h,
                tau_exo=tau_exo,
                joint_names=np.array(JOINT_NAMES),
                bones=np.array(BONES, dtype=object),
                meta=np.array([kin["meta"], dyn_meta, {"profile": profile, "phase_durations_s": phase_durs}], dtype=object),
                W_true=W_true,
            )
            index.append({"trial_id": trial_id, "subject": s, "trial": tr, "path": str(trial_path.relative_to(out_dir))})
            trial_id += 1

    # Save index
    import json
    with open(out_dir/"index.json", "w") as f:
        json.dump(index, f, indent=2)

    return {"n_trials": len(index), "raw_dir": str(raw_dir), "index_path": str(out_dir/"index.json")}
