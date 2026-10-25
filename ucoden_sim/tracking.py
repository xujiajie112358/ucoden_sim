"""Constant-velocity Kalman tracker and the detection policy it drives (duty cycle, Pfa)."""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .network import Network


class Tracker:
    def __init__(self, cfg: SimConfig, dt_s: float, sigma_acc: float = 0.02):
        self.cfg = cfg
        self.dt = dt_s
        self.alive = False
        self.epochs_since_fix = 0
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt_s
        self.F = F
        q = sigma_acc**2
        G = np.vstack([np.eye(3) * 0.5 * dt_s**2, np.eye(3) * dt_s])
        self.Q = G @ G.T * q
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e6

    def start(self, z: np.ndarray, R: np.ndarray):
        self.x = np.concatenate([z, np.zeros(3)])
        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = R
        self.P[3:6, 3:6] = np.eye(3) * 2.0**2       # ~2 m/s velocity uncertainty
        self.alive = True
        self.epochs_since_fix = 0

    def predict(self):
        if not self.alive:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray, R: np.ndarray):
        if not self.alive:
            self.start(z, R)
            return
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.epochs_since_fix = 0

    def miss(self):
        if self.alive:
            self.epochs_since_fix += 1
            if self.epochs_since_fix > self.cfg.track_lost_epochs:
                self.alive = False

    @property
    def pos(self) -> np.ndarray:
        return self.x[0:3]

    def gate_radius(self) -> float:
        lam = np.linalg.eigvalsh(self.P[0:3, 0:3]).max()
        return self.cfg.perception_radius_m + self.cfg.gate_sigma_mult * np.sqrt(max(lam, 0.0))


def apply_policy(cfg: SimConfig, net: Network, trk: Tracker) -> np.ndarray:
    """Set per-node duty cycle and Pfa. Returns the boolean mask of gated (track-mode) nodes."""
    if not cfg.c2_det_loc_track or not trk.alive:
        net.duty[:] = cfg.duty_search
        net.pfa[:] = cfg.pfa_search
        return np.zeros(net.n, bool)
    d = np.linalg.norm(net.est_pos - trk.pos[None, :], axis=1)
    gated = d <= trk.gate_radius()
    net.duty[:] = np.where(gated, cfg.duty_track_in, cfg.duty_track_out)
    net.pfa[:] = np.where(gated, cfg.pfa_track, cfg.pfa_search)
    return gated
