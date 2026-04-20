from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
from scipy.signal import butter, filtfilt, welch

def load_index(run_dir: Path) -> List[dict]:
    return json.loads((run_dir/"index.json").read_text())

def load_trials(run_dir: Path) -> List[Path]:
    return [run_dir / item["path"] for item in load_index(run_dir)]

def bandpower(x: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    f, pxx = welch(x, fs=sr, nperseg=min(len(x), 512))
    mask = (f >= fmin) & (f <= fmax)
    return float(np.trapz(pxx[mask], f[mask]) + 1e-12)

def compute_eeg_bandpower_features(eeg_seg: np.ndarray, sr: int) -> np.ndarray:
    # eeg_seg: (T, C)
    # return (C*2,) mu and beta bandpower
    C = eeg_seg.shape[1]
    feats = []
    for c in range(C):
        feats.append(bandpower(eeg_seg[:, c], sr, 8.0, 13.0))   # mu
        feats.append(bandpower(eeg_seg[:, c], sr, 13.0, 30.0))  # beta
    return np.array(feats, dtype=np.float32)


def get_channel_counts(cfg: Dict[str, Any]) -> tuple[int, int]:
    eeg_ch = int(cfg["signals"]["eeg"]["n_channels"])
    emg_ch = len(cfg["signals"]["emg"]["muscles"])
    return eeg_ch, emg_ch

def make_segments(run_dir: Path, cfg: Dict[str, Any], modality: str = "fusion") -> Dict[str, Any]:
    sr = int(cfg["sample_rate_hz"])
    seg_len = int(cfg["train"]["segment_len"])
    stride = int(cfg["train"]["segment_stride"])

    index = load_index(run_dir)
    X_list = []
    y_phase = []
    y_tau = []
    subj_ids = []
    trial_ids = []
    starts = []
    ends = []

    for item in index:
        tp = run_dir / item["path"]
        data = np.load(tp, allow_pickle=True)
        eeg = data["eeg"].astype(np.float32)  # (T,C_eeg)
        emg_env = data["emg_env"].astype(np.float32)  # (T,M)
        phase = data["phase"].astype(np.int64)
        tau_exo = data["tau_exo"].astype(np.float32)
        T = eeg.shape[0]

        # input channels: raw EEG + EMG envelope (more stable for learning)
        if modality == "eeg":
            inp = eeg
        elif modality == "emg":
            inp = emg_env
        else:
            inp = np.concatenate([eeg, emg_env], axis=1)

        # segment
        for start in range(0, T - seg_len + 1, stride):
            end = start + seg_len
            seg = inp[start:end]  # (L, C)
            # label = majority phase
            ph = int(np.bincount(phase[start:end]).argmax())
            # regression target = mean torque over segment
            tau = float(np.mean(tau_exo[start:end]))
            X_list.append(seg.T)  # (C, L) for Conv1d
            y_phase.append(ph)
            y_tau.append(tau)
            starts.append(start)
            ends.append(end)
        n_segments = (T - seg_len)//stride + 1
        subj_ids.extend([int(item["subject"])] * n_segments)
        trial_ids.extend([int(item["trial_id"])] * n_segments)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_phase = np.array(y_phase, dtype=np.int64)
    y_tau = np.array(y_tau, dtype=np.float32)
    subj_ids = np.array(subj_ids, dtype=np.int64)
    trial_ids = np.array(trial_ids, dtype=np.int64)
    starts = np.array(starts, dtype=np.int64)
    ends = np.array(ends, dtype=np.int64)
    return {
        "X": X,
        "y_phase": y_phase,
        "y_tau": y_tau,
        "subj": subj_ids,
        "trial": trial_ids,
        "start": starts,
        "end": ends,
    }

def select_split_groups(seg: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray | None:
    strategy = cfg.get("train", {}).get("split_strategy", "segment")
    if strategy == "subject":
        return seg["subj"]
    if strategy == "trial":
        return seg["trial"]
    return None

def split_indices(n: int, cfg: Dict[str, Any], seed: int, groups: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr = float(cfg["train"]["train_split"])
    va = float(cfg["train"]["val_split"])
    strategy = cfg.get("train", {}).get("split_strategy", "segment")

    if strategy == "segment" or groups is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        ntr = int(tr*n)
        nva = int(va*n)
        train_idx = idx[:ntr]
        val_idx = idx[ntr:ntr+nva]
        test_idx = idx[ntr+nva:]
        return train_idx, val_idx, test_idx

    if len(groups) != n:
        raise ValueError(f"Expected {n} groups, got {len(groups)}")

    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_groups = len(uniq)
    ntr = int(tr*n_groups)
    nva = int(va*n_groups)
    train_groups = uniq[:ntr]
    val_groups = uniq[ntr:ntr+nva]
    test_groups = uniq[ntr+nva:]

    train_idx = np.where(np.isin(groups, train_groups))[0]
    val_idx = np.where(np.isin(groups, val_groups))[0]
    test_idx = np.where(np.isin(groups, test_groups))[0]
    return train_idx, val_idx, test_idx

class StandardScaler1D:
    # per-channel scaler for (N, C, L)
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "StandardScaler1D":
        # mean/std over N and L
        self.mean = X.mean(axis=(0,2), keepdims=True)
        self.std = X.std(axis=(0,2), keepdims=True) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "StandardScaler1D":
        sc = StandardScaler1D()
        sc.mean = np.array(d["mean"], dtype=np.float32)
        sc.std = np.array(d["std"], dtype=np.float32)
        return sc


class StandardScalerTarget:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y: np.ndarray) -> "StandardScalerTarget":
        y = np.asarray(y, dtype=np.float32)
        self.mean = float(np.mean(y))
        self.std = float(np.std(y) + 1e-6)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float32) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": float(self.mean), "std": float(self.std)}

    @staticmethod
    def from_dict(d: dict) -> "StandardScalerTarget":
        sc = StandardScalerTarget()
        sc.mean = float(d["mean"])
        sc.std = float(d["std"])
        return sc
