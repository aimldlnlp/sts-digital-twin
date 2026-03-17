\
from __future__ import annotations
import os, json, time, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def now_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def humanize_dict(d: Dict[str, Any], indent: int = 0) -> str:
    lines = []
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(humanize_dict(v, indent + 2))
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)

def cosine_interp(y0: float, y1: float, n: int) -> np.ndarray:
    # smooth step using cosine
    t = np.linspace(0, 1, n, endpoint=False)
    w = (1 - np.cos(np.pi * t)) / 2
    return y0 * (1 - w) + y1 * w

def resample_to_phase(x: np.ndarray, phase: np.ndarray, n_bins: int = 100) -> np.ndarray:
    # x: (T, ...) ; phase is int labels 0..3, but we treat time index normalized 0..1
    T = x.shape[0]
    t = np.linspace(0, 1, T)
    tb = np.linspace(0, 1, n_bins)
    # interpolate along first axis for each feature dim
    if x.ndim == 1:
        return np.interp(tb, t, x)
    else:
        flat = x.reshape(T, -1)
        out = np.vstack([np.interp(tb, t, flat[:, i]) for i in range(flat.shape[1])]).T
        return out.reshape(n_bins, *x.shape[1:])
