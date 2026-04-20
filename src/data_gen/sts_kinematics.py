\
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from ..common import cosine_interp

JOINT_NAMES = [
    "pelvis","spine","neck","head",
    "lhip","lknee","lankle","ltoe",
    "rhip","rknee","rankle","rtoe",
    "lsho","lelb","lwri",
    "rsho","relb","rwri",
]

BONES = [
    ("pelvis","spine"), ("spine","neck"), ("neck","head"),
    ("pelvis","lhip"), ("lhip","lknee"), ("lknee","lankle"), ("lankle","ltoe"),
    ("pelvis","rhip"), ("rhip","rknee"), ("rknee","rankle"), ("rankle","rtoe"),
    ("neck","lsho"), ("lsho","lelb"), ("lelb","lwri"),
    ("neck","rsho"), ("rsho","relb"), ("relb","rwri"),
]

def _fk_leg(pelvis: np.ndarray, hip_flex: float, knee_flex: float, ankle: float,
            thigh: float, shank: float, foot: float, side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Planar in x-z, add small y offset for left/right
    y = 0.08 if side == "L" else -0.08
    # angles in radians; hip_flex is flexion from upright (0=standing), positive flex forward
    # define thigh direction angle from vertical downward:
    # upright standing: thigh points down (angle 0). flexion rotates thigh forward => +hip_flex
    a_thigh = hip_flex
    knee_rel = knee_flex  # 0 straight, positive flex
    a_shank = a_thigh - knee_rel  # knee flex brings shank back relative to thigh (simple model)
    a_foot = a_shank + ankle

    hip = pelvis + np.array([0.0, y, 0.0])
    knee = hip + np.array([thigh*np.sin(a_thigh), 0.0, -thigh*np.cos(a_thigh)])
    ankle = knee + np.array([shank*np.sin(a_shank), 0.0, -shank*np.cos(a_shank)])
    toe = ankle + np.array([foot*np.cos(a_foot), 0.0, foot*np.sin(a_foot)])  # toe forward
    return hip, knee, ankle, toe

def generate_sts_kinematics(cfg: Dict[str, Any], sr: int, seed: int,
                            anthropometry: Dict[str, Any], phase_durs_s: list[float] | None = None) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    dur = float(cfg["data"]["duration_s"])
    phase_durs = phase_durs_s if phase_durs_s is not None else cfg["data"]["phase_durations_s"]
    assert abs(sum(phase_durs) - dur) < 1e-6, "phase durations must sum to duration"

    T = int(dur * sr)
    t = np.arange(T) / sr

    # Phase labels
    phase = np.zeros(T, dtype=np.int64)
    idx = 0
    phase_bounds = []
    for p, pd in enumerate(phase_durs):
        n = int(pd * sr)
        phase[idx:idx+n] = p
        phase_bounds.append((idx, idx+n))
        idx += n
    phase[-1] = 3

    # Anthropometry sampled per subject/trial
    thigh = max(0.25, rng.normal(*anthropometry["thigh_m"]))
    shank = max(0.25, rng.normal(*anthropometry["shank_m"]))
    foot  = max(0.15, rng.normal(*anthropometry["foot_m"]))
    trunk = max(0.35, rng.normal(*anthropometry["trunk_m"]))

    # Angles (deg) sitting -> standing with smooth transitions
    # Sitting: hip ~80 flex, knee ~90 flex, ankle ~10 dorsiflex
    hip0, hip1 = np.deg2rad(80), np.deg2rad(0)
    knee0, knee1 = np.deg2rad(92), np.deg2rad(0)
    ank0, ank1 = np.deg2rad(10), np.deg2rad(0)

    # build piecewise smooth trajectories per phase
    hip = np.zeros(T); knee = np.zeros(T); ank = np.zeros(T)

    # Phase 0: preparation (small rocking)
    s0, e0 = phase_bounds[0]
    n0 = e0 - s0
    hip[s0:e0] = hip0 + 0.10*np.sin(np.linspace(0, np.pi, n0, endpoint=False))
    knee[s0:e0] = knee0 + 0.05*np.sin(np.linspace(0, np.pi, n0, endpoint=False))
    ank[s0:e0] = ank0

    # Phase 1: momentum transfer (lean forward, slight knee flex increase)
    s1, e1 = phase_bounds[1]
    n1 = e1 - s1
    hip[s1:e1] = cosine_interp(hip[s0:e0][-1], np.deg2rad(95), n1)
    knee[s1:e1] = cosine_interp(knee[s0:e0][-1], np.deg2rad(100), n1)
    ank[s1:e1] = cosine_interp(ank[s0:e0][-1], np.deg2rad(15), n1)

    # Phase 2: extension (stand up)
    s2, e2 = phase_bounds[2]
    n2 = e2 - s2
    hip[s2:e2] = cosine_interp(hip[s1:e1][-1], hip1, n2)
    knee[s2:e2] = cosine_interp(knee[s1:e1][-1], knee1, n2)
    ank[s2:e2] = cosine_interp(ank[s1:e1][-1], ank1, n2)

    # Phase 3: stabilization (settle)
    s3, e3 = phase_bounds[3]
    n3 = e3 - s3
    hip[s3:e3] = hip1 + 0.03*np.sin(np.linspace(0, 2*np.pi, n3, endpoint=False))
    knee[s3:e3] = knee1 + 0.02*np.sin(np.linspace(0, 2*np.pi, n3, endpoint=False))
    ank[s3:e3] = ank1 + 0.01*np.sin(np.linspace(0, 2*np.pi, n3, endpoint=False))

    # Pelvis trajectory (x forward + z up)
    # approximate COM rises during extension
    pelvis_x = 0.05*np.sin(np.linspace(0, np.pi, T, endpoint=False))
    pelvis_z = 0.85*trunk + 0.10*np.sin(np.linspace(0, np.pi, T, endpoint=False))  # baseline
    # add extra lift during extension
    ext_mask = (phase == 2).astype(float)
    # smooth ext mask
    ext_smooth = np.convolve(ext_mask, np.ones(21)/21, mode="same")
    pelvis_z += 0.20*ext_smooth
    pelvis = np.stack([pelvis_x, np.zeros(T), pelvis_z], axis=1)

    # Upper body joints
    spine = pelvis + np.stack([0.0*np.ones(T), np.zeros(T), 0.40*trunk*np.ones(T)], axis=1)
    neck  = pelvis + np.stack([0.02*np.sin(hip), np.zeros(T), 0.62*trunk*np.ones(T)], axis=1)
    head  = pelvis + np.stack([0.03*np.sin(hip), np.zeros(T), 0.80*trunk*np.ones(T)], axis=1)

    # Arms (simple)
    lsho = neck + np.array([0.05, 0.12, 0.02])
    rsho = neck + np.array([0.05,-0.12, 0.02])
    # elbows/wrists hang down a bit; add small motion
    arm_swing = 0.03*np.sin(np.linspace(0, 2*np.pi, T, endpoint=False))
    lelb = lsho + np.stack([0.02+arm_swing, 0.00*np.ones(T), -0.20*np.ones(T)], axis=1)
    lwri = lelb + np.stack([0.02+arm_swing, 0.00*np.ones(T), -0.20*np.ones(T)], axis=1)
    relb = rsho + np.stack([0.02+arm_swing, 0.00*np.ones(T), -0.20*np.ones(T)], axis=1)
    rwri = relb + np.stack([0.02+arm_swing, 0.00*np.ones(T), -0.20*np.ones(T)], axis=1)

    # Legs FK per time
    lhip = np.zeros((T,3)); lknee = np.zeros((T,3)); lankle = np.zeros((T,3)); ltoe = np.zeros((T,3))
    rhip = np.zeros((T,3)); rknee = np.zeros((T,3)); rankle = np.zeros((T,3)); rtoe = np.zeros((T,3))
    for i in range(T):
        hipL, kneeL, ankleL, toeL = _fk_leg(pelvis[i], hip[i], knee[i], ank[i], thigh, shank, foot, "L")
        hipR, kneeR, ankleR, toeR = _fk_leg(pelvis[i], hip[i], knee[i], ank[i], thigh, shank, foot, "R")
        lhip[i], lknee[i], lankle[i], ltoe[i] = hipL, kneeL, ankleL, toeL
        rhip[i], rknee[i], rankle[i], rtoe[i] = hipR, kneeR, ankleR, toeR

    # Compose joints in fixed order
    joints = np.zeros((T, len(JOINT_NAMES), 3), dtype=np.float32)
    name_to_arr = {
        "pelvis": pelvis, "spine": spine, "neck": neck, "head": head,
        "lhip": lhip, "lknee": lknee, "lankle": lankle, "ltoe": ltoe,
        "rhip": rhip, "rknee": rknee, "rankle": rankle, "rtoe": rtoe,
        "lsho": lsho, "lelb": lelb, "lwri": lwri,
        "rsho": rsho, "relb": relb, "rwri": rwri,
    }
    for j, n in enumerate(JOINT_NAMES):
        arr = name_to_arr[n]
        joints[:, j, :] = arr.astype(np.float32)

    angles = np.stack([hip, knee, ank], axis=1).astype(np.float32)  # rad
    meta = {
        "thigh_m": float(thigh), "shank_m": float(shank), "foot_m": float(foot), "trunk_m": float(trunk),
    }
    return {
        "t": t.astype(np.float32),
        "phase": phase,
        "joints": joints,
        "angles": angles,
        "meta": meta,
    }
