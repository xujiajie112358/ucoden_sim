"""Per-epoch detection of the target pulse and the (collaborative or not) trigger decision."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .environment import Acoustics, RayLUT, SoundSpeedPerturbation
from .network import Network


@dataclass
class EpochDetection:
    detected: np.ndarray    # bool (N,)  node reported a detection (true or false alarm)
    is_false: np.ndarray    # bool (N,)  detection is a false alarm
    toa_true: np.ndarray    # (N,) noiseless time of arrival of the target pulse [s]
    direct: np.ndarray      # (N,) direct eigenray exists target -> node
    eta_true: np.ndarray    # (N,) curve-to-line factor of the heard arrival
    cbar_true: np.ndarray   # (N,) path-mean sound speed incl. spatial perturbation
    d_true: np.ndarray      # (N,) straight-line distance target -> node


def detect_epoch(cfg: SimConfig, net: Network, ac: Acoustics, lut: RayLUT,
                 pert: SoundSpeedPerturbation, x_t: np.ndarray, rng: np.random.Generator) -> EpochDetection:
    P = net.true_pos
    dx = P[:, 0] - x_t[0]
    dy = P[:, 1] - x_t[1]
    r_h = np.sqrt(dx**2 + dy**2)
    d = np.sqrt(r_h**2 + (P[:, 2] - x_t[2]) ** 2)
    q = lut.query(np.full(net.n, x_t[2]), P[:, 2], r_h)
    cbar = q.cbar + pert(0.5 * (P[:, 0] + x_t[0]), 0.5 * (P[:, 1] + x_t[1]))
    path_len = q.eta * d
    toa = path_len / cbar
    pd = ac.pd(path_len, q.direct, net.pfa)
    awake = rng.random(net.n) < net.duty
    det_true = awake & (rng.random(net.n) < pd)
    false = awake & ~det_true & (rng.random(net.n) < net.pfa)
    return EpochDetection(detected=det_true | false, is_false=false, toa_true=toa, direct=q.direct,
                          eta_true=q.eta, cbar_true=cbar, d_true=d)


@dataclass
class Trigger:
    positive: bool
    members: np.ndarray     # node indices forming the detection cluster
    all_false: bool         # cluster contains no true detection


def decide_trigger(cfg: SimConfig, net: Network, det: EpochDetection) -> Trigger:
    """Collaborative k-out-of-n decision (C2 on) or 'any single report triggers' (C2 off)."""
    idx = np.where(det.detected)[0]
    if idx.size == 0:
        return Trigger(False, idx, True)
    P = net.est_pos[idx, 0:2]
    dd = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    if cfg.c2_det_loc_track:
        counts = (dd <= cfg.k_pair_dist_m).sum(1)          # includes itself
        best = int(np.argmax(counts))
        if counts[best] < cfg.k_of_n:
            return Trigger(False, idx[:0], True)
        members = idx[dd[best] <= cfg.k_pair_dist_m]
    else:
        # uncoordinated: the first report defines the group; anything within 2*PR joins
        members = idx[dd[0] <= cfg.group_radius_m]
    all_false = bool(det.is_false[members].all())
    return Trigger(True, members, all_false)
