"""Range measurements (with NPE / SVE / CPE), target fix, GDOP, node selection and node calibration."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse

from .config import SimConfig
from .environment import RayLUT
from .network import Network


# ----------------------------------------------------------------------------- geometry
def gdop(anchors: np.ndarray, x: np.ndarray) -> float:
    v = anchors - x[None, :]
    nrm = np.linalg.norm(v, axis=1)
    if (nrm < 1e-6).any() or len(anchors) < 4:
        return float("inf")
    H = v / nrm[:, None]
    try:
        tr = float(np.trace(np.linalg.inv(H.T @ H)))
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.sqrt(tr)) if tr > 0 else float("inf")


@dataclass
class Fix:
    x: np.ndarray
    cov: np.ndarray
    gdop: float
    n_used: int


def wls_fix(anchors: np.ndarray, d_hat: np.ndarray, sigma: np.ndarray, z_prior: float,
            x0: Optional[np.ndarray] = None, iters: int = 12) -> Optional[Fix]:
    """Weighted least-squares range fix. 3-D with >= 4 ranges; 2-D at z_prior with 3 ranges."""
    n = len(d_hat)
    if n < 3:
        return None
    w = 1.0 / np.maximum(sigma, 1e-3)
    if x0 is None:
        x = np.array([anchors[:, 0].mean(), anchors[:, 1].mean(), z_prior])
    else:
        x = x0.astype(float).copy()
    three_d = n >= 4
    for _ in range(iters):
        v = x[None, :] - anchors
        rng_ = np.linalg.norm(v, axis=1)
        rng_ = np.maximum(rng_, 1.0)
        res = (d_hat - rng_) * w
        J = (v / rng_[:, None]) * w[:, None]
        if not three_d:
            J = J[:, :2]
        try:
            dx, *_ = np.linalg.lstsq(J, res, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not three_d:
            dx = np.array([dx[0], dx[1], 0.0])
        x = x + dx
        if np.linalg.norm(dx) < 0.01:
            break
    v = x[None, :] - anchors
    rng_ = np.maximum(np.linalg.norm(v, axis=1), 1.0)
    J = (v / rng_[:, None]) * w[:, None]
    try:
        cov = np.linalg.inv(J.T @ J) if three_d else np.diag([*np.diag(np.linalg.inv(J[:, :2].T @ J[:, :2])), 1e6])
    except np.linalg.LinAlgError:
        cov = np.eye(3) * 1e6
    g = gdop(anchors, x) if three_d else float("inf")
    return Fix(x=x, cov=cov, gdop=g, n_used=n)


def select_nodes(anchors_all: np.ndarray, cand: np.ndarray, x_ref: np.ndarray, k_poll: int) -> np.ndarray:
    """Pick up to k_poll candidate nodes with good geometry around x_ref.

    Brute force over the 8 nearest candidates for the best 4-node GDOP, then greedily add nodes
    that reduce GDOP most until k_poll nodes are selected.
    """
    if cand.size <= 4:
        return cand
    d = np.linalg.norm(anchors_all[cand] - x_ref[None, :], axis=1)
    near = cand[np.argsort(d)[:8]]
    best, best_g = None, float("inf")
    for comb in combinations(range(len(near)), 4):
        sel = near[list(comb)]
        g = gdop(anchors_all[sel], x_ref)
        if g < best_g:
            best, best_g = sel, g
    chosen = list(best)
    remaining = [c for c in cand if c not in chosen]
    while len(chosen) < k_poll and remaining:
        gains = [gdop(anchors_all[chosen + [c]], x_ref) for c in remaining]
        j = int(np.argmin(gains))
        chosen.append(remaining.pop(j))
    return np.array(chosen)


# ----------------------------------------------------------------------------- range model
def measured_toa(cfg: SimConfig, toa_true: np.ndarray, is_false: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """Noisy time of arrival; false alarms carry a meaningless 'arrival' inside the perception radius."""
    toa = toa_true + rng.normal(0, cfg.toa_sigma_s, toa_true.shape)
    if is_false.any():
        toa[is_false] = rng.uniform(0.0, cfg.perception_radius_m, is_false.sum()) / cfg.nominal_c_mps
    return toa


def ranges_from_toa(cfg: SimConfig, lut: RayLUT, toa: np.ndarray, node_depth_est: np.ndarray,
                    r_h_est: np.ndarray, zs_guess: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Convert times of arrival to straight-line distance estimates and their 1-sigma.

    C3 off : d = c_nominal * toa                   (straight path, nominal 1500 m/s)
    C3 on  : d = c_est * toa / eta_est              (locally calibrated sound speed, ray-table
                                                     curve-to-line and direct/reflected knowledge)
    """
    if cfg.c3_env:
        q = lut.query(np.full(toa.shape, zs_guess), node_depth_est, r_h_est)
        c_est = q.cbar + rng.normal(0, cfg.c_est_sigma_mps, toa.shape)
        eta_est = 1.0 + (q.eta - 1.0) * (1.0 + rng.normal(0, cfg.eta_est_rel_sigma, toa.shape))
        d = c_est * toa / np.maximum(eta_est, 1.0)
    else:
        d = cfg.nominal_c_mps * toa
    sigma = cfg.nominal_c_mps * cfg.toa_sigma_s + cfg.range_model_frac * d
    return d, sigma


# ----------------------------------------------------------------------------- node calibration
class NodeCalibrator:
    """Re-estimates horizontal node positions from inter-node ranges (dedicated pings or
    opportunistic packet-derived ranges) plus GNSS-referenced chains and a soft design prior."""

    def __init__(self, cfg: SimConfig, net: Network):
        self.cfg = cfg
        self.net = net
        self.links: Dict[Tuple[int, int], list] = {}   # (i, j) -> [(t, d_hat), ...] within the window

    def add(self, i: int, j: int, d_hat: float, t_s: float):
        key = (i, j) if i < j else (j, i)
        rec = self.links.get(key)
        if rec is None:
            self.links[key] = [(t_s, d_hat)]
        else:
            rec.append((t_s, d_hat))
            if len(rec) > 60:
                del rec[:-60]

    def prune(self, t_s: float):
        """Keep only the samples inside the sliding window; drop links with none left."""
        horizon = t_s - self.cfg.calib_window_h * 3600.0
        for key in list(self.links.keys()):
            rec = [x for x in self.links[key] if x[0] >= horizon]
            if rec:
                self.links[key] = rec
            else:
                del self.links[key]

    def solve(self, t_s: float, iters: int = 4):
        """Recursive weighted least squares on the horizontal node positions.

        Prior per node: its previous estimate with a random-walk sigma (``calib_process_sigma_m``),
        or the GNSS reference for GNSS chains. Nodes without a link in the window therefore keep
        their last estimate instead of reverting to the design position. Links whose normalised
        residual exceeds ``calib_outlier_sigma`` after the first iteration (a reflected arrival
        taken for a direct one, or vice versa) are dropped.
        """
        cfg, net = self.cfg, self.net
        self.prune(t_s)
        if not self.links:
            return
        keys = list(self.links.keys())
        I = np.array([k[0] for k in keys]); J = np.array([k[1] for k in keys])
        d_meas = np.array([np.mean([x[1] for x in v]) for v in self.links.values()])
        n_meas = np.array([len(v) for v in self.links.values()])
        sig_link = (cfg.nominal_c_mps * cfg.toa_sigma_s + cfg.range_model_frac * d_meas) / np.sqrt(n_meas)
        z = net.est_pos[:, 2]
        N = net.n
        xy = net.est_pos[:, 0:2].copy()
        gnss_ref = net.gnss_reference()
        gidx = np.where(net.gnss_node)[0]
        prior_sig = np.where(net.gnss_node, cfg.gnss_sigma_m, cfg.calib_process_sigma_m)
        prior_xy = net.est_pos[:, 0:2].copy()
        prior_xy[gidx] = gnss_ref
        keep = np.ones(len(keys), bool)
        for it in range(iters):
            Ik, Jk, dk, sk = I[keep], J[keep], d_meas[keep], sig_link[keep]
            L = len(Ik)
            r = np.arange(L)
            pr = L + np.arange(N)
            dxy = xy[Ik] - xy[Jk]
            dz = z[Ik] - z[Jk]
            rng_ = np.sqrt((dxy**2).sum(1) + dz**2)
            rng_ = np.maximum(rng_, 1.0)
            res_link = (dk - rng_) / sk
            if it == 1:
                # outlier rejection on the residuals of the first corrected geometry
                bad = np.abs(res_link) > cfg.calib_outlier_sigma
                if bad.any() and (~bad).sum() >= 4:
                    idx_keep = np.where(keep)[0]
                    keep[idx_keep[bad]] = False
                    continue
            u = dxy / rng_[:, None]
            # sparse Jacobian: 4 entries per link row, 1 per prior row  -> normal equations
            rows = np.concatenate([r, r, r, r, pr, pr + N])
            cols = np.concatenate([Ik, N + Ik, Jk, N + Jk, np.arange(N), N + np.arange(N)])
            vals = np.concatenate([u[:, 0] / sk, u[:, 1] / sk, -u[:, 0] / sk, -u[:, 1] / sk,
                                   1.0 / prior_sig, 1.0 / prior_sig])
            A = sparse.csr_matrix((vals, (rows, cols)), shape=(L + 2 * N, 2 * N))
            b = np.concatenate([res_link, (prior_xy[:, 0] - xy[:, 0]) / prior_sig,
                                (prior_xy[:, 1] - xy[:, 1]) / prior_sig])
            AtA = (A.T @ A).toarray()
            Atb = A.T @ b
            try:
                delta = np.linalg.solve(AtA + 1e-9 * np.eye(2 * N), Atb)
            except np.linalg.LinAlgError:
                break
            xy[:, 0] += delta[:N]; xy[:, 1] += delta[N:]
            if np.abs(delta).max() < 0.05 and it >= 1:
                break
        net.est_pos[:, 0:2] = xy
