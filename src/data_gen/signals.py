\
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from scipy.signal import butter, filtfilt

def _band_limited_noise(rng: np.random.Generator, n: int, sr: int, low: float, high: float) -> np.ndarray:
    x = rng.normal(0, 1, n)
    nyq = sr / 2.0
    # clamp to valid digital filter range
    low = max(0.1, min(low, nyq * 0.99))
    high = max(low + 0.1, min(high, nyq * 0.99))

    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)

def generate_eeg(cfg: Dict[str, Any], t: np.ndarray, phase: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eeg_cfg = cfg["signals"]["eeg"]
    C = int(eeg_cfg["n_channels"])
    mu = float(eeg_cfg["mu_hz"])
    beta = float(eeg_cfg["beta_hz"])
    erd = float(eeg_cfg["erd_factor"])
    rebound = float(eeg_cfg["rebound_factor"])
    noise_std = float(eeg_cfg["noise_std"])
    artifact_prob = float(eeg_cfg["artifact_prob"])

    T = t.shape[0]
    X = np.zeros((T, C), dtype=np.float32)

    # amplitude envelope by phase: ERD during momentum+extension
    amp = np.ones(T, dtype=np.float32)
    amp[(phase == 1) | (phase == 2)] *= erd
    amp[(phase == 3)] *= rebound
    # smooth envelope
    amp = np.convolve(amp, np.ones(33)/33, mode="same").astype(np.float32)

    for c in range(C):
        ph1 = rng.uniform(0, 2*np.pi)
        ph2 = rng.uniform(0, 2*np.pi)
        base = (np.sin(2*np.pi*mu*t + ph1) + 0.6*np.sin(2*np.pi*beta*t + ph2))
        # add mild 1/f-ish noise via lowpass filtered white noise
        colored = _band_limited_noise(rng, T, int(cfg["sample_rate_hz"]), 1.0, 45.0)
        sig = amp * base + 0.25*colored
        sig += rng.normal(0, noise_std, T)

        # artifacts: random spikes
        spikes = rng.random(T) < artifact_prob
        if spikes.any():
            sig[spikes] += rng.normal(0, 4.0, spikes.sum())
        X[:, c] = sig.astype(np.float32)
    return X

def generate_emg(cfg: Dict[str, Any], t: np.ndarray, phase: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    emg_cfg = cfg["signals"]["emg"]
    muscles = emg_cfg["muscles"]
    M = len(muscles)
    sr = int(cfg["sample_rate_hz"])
    T = t.shape[0]

    # 3 underlying synergies -> muscles
    K = int(cfg["synergy"]["n_synergies"])
    W_true = rng.uniform(0.1, 1.0, size=(M, K))
    W_true = W_true / (W_true.sum(axis=0, keepdims=True) + 1e-6)

    # synergy activations over time (phase-shaped)
    H_true = np.zeros((K, T), dtype=np.float32)
    # Syn0: extension (quad/glute)
    H_true[0] = 0.2 + 1.2*((phase == 2).astype(np.float32))
    # Syn1: momentum transfer (TA + trunk)
    H_true[1] = 0.15 + 1.0*((phase == 1).astype(np.float32))
    # Syn2: stabilization (co-contraction)
    H_true[2] = 0.10 + 0.6*((phase == 3).astype(np.float32))
    # smooth
    for k in range(K):
        H_true[k] = np.convolve(H_true[k], np.ones(41)/41, mode="same")

    # muscle envelopes
    env = (W_true @ H_true).astype(np.float32)
    # add small muscle-specific modulation
    env *= (0.9 + 0.2*rng.random((M, 1))).astype(np.float32)

    # crosstalk mixing
    crosstalk = float(emg_cfg["crosstalk"])
    A = np.eye(M) + crosstalk * (rng.normal(0, 0.5, size=(M, M)))
    env_mix = np.clip(A @ env, 0.0, None).astype(np.float32)

    # raw EMG: band-limited noise scaled by envelope
    raw = np.zeros((T, M), dtype=np.float32)
    base_noise_std = float(emg_cfg["base_noise_std"])
    for m in range(M):
        n = _band_limited_noise(rng, T, sr, 20.0, 150.0)
        sig = env_mix[m] * n
        sig += rng.normal(0, base_noise_std, T)
        raw[:, m] = sig.astype(np.float32)

    # envelope (rectify + lowpass)
    b, a = butter(4, 5.0/(sr/2), btype="low")
    envelope = filtfilt(b, a, np.abs(raw), axis=0).astype(np.float32)
    return raw, envelope, W_true.astype(np.float32)
