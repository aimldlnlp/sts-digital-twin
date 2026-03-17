\
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

def compute_knee_torque_proxy(cfg: Dict[str, Any], knee_angle_rad: np.ndarray, phase: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    dyn = cfg["digital_twin"]["dynamics"]
    I = max(0.05, rng.normal(*dyn["knee_I"]))
    b = max(0.01, rng.normal(*dyn["knee_b"]))
    k = max(0.5, rng.normal(*dyn["knee_k"]))

    sr = int(cfg["sample_rate_hz"])
    # derivatives
    th = knee_angle_rad.astype(np.float32)
    thd = np.gradient(th) * sr
    thdd = np.gradient(thd) * sr

    th_rest = 0.0
    tau_h = I*thdd + b*thd + k*(th - th_rest)

    # Assist target: proportion of |human torque| during extension
    assist = cfg["digital_twin"]["assistance"]
    ratio = float(rng.normal(assist["assist_ratio_mean"], assist["assist_ratio_std"]))
    ratio = float(np.clip(ratio, 0.1, 0.7))
    max_tau = float(assist["max_exo_torque"])

    tau_exo = np.zeros_like(tau_h, dtype=np.float32)
    ext = (phase == 2)
    tau_exo[ext] = np.clip(ratio * tau_h[ext], -max_tau, max_tau)

    meta = {"knee_I": float(I), "knee_b": float(b), "knee_k": float(k), "assist_ratio": float(ratio), "max_exo_torque": float(max_tau)}
    return tau_h.astype(np.float32), tau_exo.astype(np.float32), meta
