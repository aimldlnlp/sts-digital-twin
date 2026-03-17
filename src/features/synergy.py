\
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.decomposition import NMF

def fit_nmf_synergy(run_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Stack EMG envelopes across all trials
    idx = json.loads((run_dir/"index.json").read_text())
    M = None
    X = []
    for item in idx:
        data = np.load(run_dir/item["path"], allow_pickle=True)
        env = data["emg_env"].astype(np.float32)  # (T,M)
        if M is None:
            M = env.shape[1]
        X.append(env)
    X = np.vstack(X)  # (sumT, M)
    X = np.clip(X, 0.0, None)

    K = int(cfg["synergy"]["n_synergies"])
    nmf = NMF(n_components=K, init="nndsvda", max_iter=int(cfg["synergy"]["nmf_max_iter"]), random_state=int(cfg["seed"]))
    H = nmf.fit_transform(X)   # (sumT, K)
    W = nmf.components_        # (K, M)
    recon = H @ W
    vaf = 1.0 - (np.var(X - recon) / (np.var(X) + 1e-9))

    art = run_dir/"artifacts"
    art.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art/"nmf_synergy.npz", W=W.astype(np.float32), vaf=float(vaf))
    with open(art/"nmf_synergy_meta.json", "w") as f:
        json.dump({"vaf": float(vaf), "K": K, "M": int(M)}, f, indent=2)
    return {"vaf": float(vaf), "K": K, "M": int(M)}
