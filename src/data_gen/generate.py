\
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .sts_kinematics import generate_sts_kinematics, JOINT_NAMES, BONES
from .signals import generate_eeg, generate_emg
from .torque import compute_knee_torque_proxy

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
        for tr in range(tps):
            seed = int(cfg["seed"]) + 1000*s + tr
            kin = generate_sts_kinematics(cfg, sr, seed=seed, anthropometry=anth)

            eeg = generate_eeg(cfg, kin["t"], kin["phase"], seed=seed+7)
            emg_raw, emg_env, W_true = generate_emg(cfg, kin["t"], kin["phase"], seed=seed+13)

            # knee angle is angles[:,1]
            tau_h, tau_exo, dyn_meta = compute_knee_torque_proxy(cfg, kin["angles"][:,1], kin["phase"], seed=seed+31)

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
                meta=np.array([kin["meta"], dyn_meta], dtype=object),
                W_true=W_true,
            )
            index.append({"trial_id": trial_id, "subject": s, "trial": tr, "path": str(trial_path.relative_to(out_dir))})
            trial_id += 1

    # Save index
    import json
    with open(out_dir/"index.json", "w") as f:
        json.dump(index, f, indent=2)

    return {"n_trials": len(index), "raw_dir": str(raw_dir), "index_path": str(out_dir/"index.json")}
