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


def _shift_signal(x: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return x.copy()
    out = np.empty_like(x)
    if shift > 0:
        out[:shift] = x[0]
        out[shift:] = x[:-shift]
    else:
        out[shift:] = x[-1]
        out[:shift] = x[-shift:]
    return out


def _trial_drift(rng: np.random.Generator, n: int, strength: float) -> np.ndarray:
    if strength <= 0:
        return np.ones(n, dtype=np.float32)
    slope = rng.normal(0.0, strength)
    curve = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    return np.clip(1.0 + slope * curve, 0.5, 1.8).astype(np.float32)


def _corrupt_channel(
    rng: np.random.Generator,
    signal: np.ndarray,
    dropout_prob: float,
    corruption_strength: float,
) -> np.ndarray:
    out = signal.copy()
    if rng.random() >= dropout_prob:
        return out
    n = len(out)
    span = max(8, int(rng.uniform(0.08, 0.22) * n))
    start = int(rng.integers(0, max(1, n - span)))
    end = min(n, start + span)
    if rng.random() < 0.5:
        out[start:end] *= rng.uniform(0.0, 0.15)
    else:
        out[start:end] += rng.normal(0.0, corruption_strength * max(1e-3, float(np.std(out))), end - start)
    return out


def generate_eeg(cfg: Dict[str, Any], t: np.ndarray, phase: np.ndarray, seed: int, profile: Dict[str, Any] | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eeg_cfg = cfg["signals"]["eeg"]
    aug_cfg = cfg.get("augment", {})
    profile = profile or {}
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
    amp = _shift_signal(amp, int(profile.get("eeg_latency_samples", 0)))
    amp *= float(profile.get("eeg_gain", 1.0)) * float(profile.get("eeg_trial_gain", 1.0))
    amp *= _trial_drift(rng, T, float(aug_cfg.get("eeg_drift_strength", 0.0)))

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
        sig = _corrupt_channel(
            rng,
            sig,
            dropout_prob=float(aug_cfg.get("channel_dropout_prob", 0.0)),
            corruption_strength=float(aug_cfg.get("corruption_strength", 0.0)),
        )
        X[:, c] = sig.astype(np.float32)
    return X


def generate_emg(cfg: Dict[str, Any], t: np.ndarray, phase: np.ndarray, seed: int, profile: Dict[str, Any] | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    emg_cfg = cfg["signals"]["emg"]
    aug_cfg = cfg.get("augment", {})
    profile = profile or {}
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
        H_true[k] = _shift_signal(H_true[k], int(profile.get("emg_latency_samples", 0)))

    # muscle envelopes
    env = (W_true @ H_true).astype(np.float32)
    # add small muscle-specific modulation
    env *= (0.9 + 0.2*rng.random((M, 1))).astype(np.float32)
    env *= float(profile.get("emg_gain", 1.0)) * float(profile.get("emg_trial_gain", 1.0))
    env *= _trial_drift(rng, T, float(aug_cfg.get("emg_drift_strength", 0.0)))[None, :]

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
        sig = _corrupt_channel(
            rng,
            sig,
            dropout_prob=float(aug_cfg.get("channel_dropout_prob", 0.0)),
            corruption_strength=float(aug_cfg.get("corruption_strength", 0.0)),
        )
        raw[:, m] = sig.astype(np.float32)

    # envelope (rectify + lowpass)
    b, a = butter(4, 5.0/(sr/2), btype="low")
    envelope = filtfilt(b, a, np.abs(raw), axis=0).astype(np.float32)
    return raw, envelope, W_true.astype(np.float32)
