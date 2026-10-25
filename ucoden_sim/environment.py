"""Ocean environment: sound-speed profile, ray-tracing lookup table and the passive sonar equation.

Design choices:

* Sound speed from the Medwin formula, as in the UCODEN paper (IEEE OJCOMS, 2025, Eq. (1)).
* A synthetic winter north-west Pacific profile standing in for the Argo profile
  E143N33 (2020-01): mixed layer ~18 C, thermocline to ~3.5 C at 1 km, SOFAR axis near 1 km.
* Range-independent ray tracing (Snell's law) once per source depth, turned into a
  lookup table over (source depth, receiver depth, horizontal range) that returns the
  curve-to-line factor eta, the path-mean sound speed and whether a direct eigenray exists.
  Where no direct eigenray exists the receiver hears a surface-reflected path, modelled
  with the image source.
* Passive sonar equation with Thorp absorption and Coates (Wenz-type) ambient noise,
  energy detector with time-bandwidth product T*B.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import norm

from .config import SimConfig


# ----------------------------------------------------------------------------- sound speed
def medwin_c(T: np.ndarray, S: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Medwin (1975) sound speed [m/s]; T in C, S in ppt, z depth in m."""
    T = np.asarray(T, float)
    S = np.asarray(S, float)
    z = np.asarray(z, float)
    return 1449.2 + 4.6 * T - 0.055 * T**2 + 0.00029 * T**3 + (1.34 - 0.010 * T) * (S - 35.0) + 0.016 * z


def synthetic_profile(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Temperature and salinity versus depth for a winter NW-Pacific-like water column."""
    z = np.asarray(z, float)
    T = np.where(
        z <= 100.0,
        18.0,
        np.where(
            z <= 1000.0,
            3.5 + (18.0 - 3.5) * np.exp(-(z - 100.0) / 300.0),
            3.5 - (3.5 - 1.8) * (z - 1000.0) / 1000.0,
        ),
    )
    S = 34.6 - 0.4 * np.exp(-((z - 600.0) / 250.0) ** 2)
    return T, S


class SoundSpeedProfile:
    """c(z) on a fine grid with linear interpolation."""

    def __init__(self, seabed_depth_m: float, dz: float = 5.0):
        self.z = np.arange(0.0, seabed_depth_m + dz, dz)
        T, S = synthetic_profile(self.z)
        self.c = medwin_c(T, S, self.z)
        self.c_mean = float(np.mean(self.c))

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(z, self.z[0], self.z[-1]), self.z, self.c)

    def depth_mean(self, z1: float, z2: float) -> float:
        lo, hi = sorted((max(z1, 0.0), max(z2, 0.0)))
        if hi - lo < 1e-6:
            return float(self(np.array([lo]))[0])
        mask = (self.z >= lo) & (self.z <= hi)
        if mask.sum() < 2:
            return float(self(np.array([0.5 * (lo + hi)]))[0])
        return float(np.mean(self.c[mask]))


# ----------------------------------------------------------------------------- ray tracing
@dataclass
class RayQuery:
    eta: np.ndarray      # curve-to-line factor of the first arrival (>= 1)
    cbar: np.ndarray     # path-mean sound speed along that arrival [m/s]
    direct: np.ndarray   # True where a direct (unreflected) eigenray exists


class RayLUT:
    """Lookup table of first-arrival eigenray properties for a range-independent profile.

    Axes: source depth (``zs_grid``), receiver depth (``zr_grid``), horizontal range (``r_grid``).
    Built once and cached as .npz next to this file (keyed by the relevant config values).
    """

    def __init__(self, cfg: SimConfig, ssp: SoundSpeedProfile, cache_dir: str | None = None):
        self.cfg = cfg
        self.ssp = ssp
        zs = sorted(set(list(cfg.target_depth_grid_m) + list(cfg.tier_depths_m)))
        self.zs_grid = np.array(zs, float)
        self.zr_grid = np.array(sorted(set(cfg.tier_depths_m)), float)
        self.r_grid = np.arange(cfg.ray_dr_m, cfg.ray_r_max_m + cfg.ray_dr_m, cfg.ray_dr_m)
        cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "_cache")
        os.makedirs(cache_dir, exist_ok=True)
        key = hashlib.md5(
            repr((tuple(self.zs_grid), tuple(self.zr_grid), cfg.ray_r_max_m, cfg.ray_dr_m,
                  cfg.ray_ds_m, cfg.ray_angle_max_deg, cfg.ray_angle_step_deg,
                  cfg.ray_fine_angle_deg, cfg.ray_fine_step_deg, cfg.ray_ultra_angle_deg,
                  cfg.ray_ultra_step_deg, cfg.seabed_depth_m, "v3")).encode()
        ).hexdigest()[:12]
        path = os.path.join(cache_dir, f"raylut_{key}.npz")
        if os.path.exists(path):
            d = np.load(path)
            self.eta, self.cbar, self.direct = d["eta"], d["cbar"], d["direct"]
        else:
            self.eta, self.cbar, self.direct = self._build()
            np.savez_compressed(path, eta=self.eta, cbar=self.cbar, direct=self.direct)
        # depth-mean sound speed used for the surface-reflected fallback, per (zs, zr)
        self.cref = np.array([[ssp.depth_mean(0.0, max(a, b)) for b in self.zr_grid] for a in self.zs_grid])

    # -- construction --------------------------------------------------------------------
    def _trace_fan(self, zs: float):
        """Trace a fan of rays from depth zs with the arc-length ray equations (Jensen et al.).

        State (r, z, zeta) with the Snell invariant xi = cos(theta)/c fixed per ray:
            dr/ds = c*xi,  dz/ds = c*zeta,  dzeta/ds = -c'(z)/c^2,  dtau/ds = 1/c.
        Turning points need no special handling. All rays are integrated together (RK4);
        a ray stops when it hits the surface, the seabed or the maximum range.
        Returns list of (r, z, s, tau) arrays, one per ray, boundary-free portion only.
        """
        cfg, ssp = self.cfg, self.ssp
        coarse = np.arange(-cfg.ray_angle_max_deg, cfg.ray_angle_max_deg + 1e-9, cfg.ray_angle_step_deg)
        fine = np.arange(-cfg.ray_fine_angle_deg, cfg.ray_fine_angle_deg + 1e-9, cfg.ray_fine_step_deg)
        ultra = np.arange(-cfg.ray_ultra_angle_deg, cfg.ray_ultra_angle_deg + 1e-9, cfg.ray_ultra_step_deg)
        angles = np.deg2rad(np.unique(np.round(np.concatenate([coarse, fine, ultra]), 5)))
        ds = cfg.ray_ds_m
        n_steps = int(np.ceil(cfg.ray_r_max_m / (ds * np.cos(np.deg2rad(cfg.ray_angle_max_deg))))) + 2
        zg, cg = ssp.z, ssp.c
        dcdz = np.gradient(cg, zg)

        def c_of(z):
            return np.interp(np.clip(z, zg[0], zg[-1]), zg, cg)

        def dc_of(z):
            return np.interp(np.clip(z, zg[0], zg[-1]), zg, dcdz)

        c0 = float(c_of(np.array([zs]))[0])
        xi = np.cos(angles) / c0
        zeta = np.sin(angles) / c0          # positive = downward
        r = np.zeros_like(angles); z = np.full_like(angles, zs); tau = np.zeros_like(angles)
        active = np.ones(angles.shape, bool)
        n_rays = angles.size
        R = np.full((n_rays, n_steps), np.nan); Z = np.full((n_rays, n_steps), np.nan)
        TAU = np.full((n_rays, n_steps), np.nan)

        def rhs(z_, zeta_):
            c = c_of(z_)
            return c * xi, c * zeta_, -dc_of(z_) / c**2, 1.0 / c

        for k in range(n_steps):
            k1 = rhs(z, zeta)
            k2 = rhs(z + 0.5 * ds * k1[1], zeta + 0.5 * ds * k1[2])
            k3 = rhs(z + 0.5 * ds * k2[1], zeta + 0.5 * ds * k2[2])
            k4 = rhs(z + ds * k3[1], zeta + ds * k3[2])
            r = r + ds / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            z = z + ds / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            zeta = zeta + ds / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            tau = tau + ds / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            active &= (z > 0.0) & (z < cfg.seabed_depth_m) & (r <= cfg.ray_r_max_m)
            if not active.any():
                break
            R[active, k] = r[active]; Z[active, k] = z[active]; TAU[active, k] = tau[active]
        rays = []
        for i in range(n_rays):
            m = ~np.isnan(R[i])
            n = int(m.sum())
            if n > 2:
                rays.append((R[i, :n], Z[i, :n], ds * np.arange(1, n + 1), TAU[i, :n]))
        return rays

    def _build(self):
        nzs, nzr, nr = len(self.zs_grid), len(self.zr_grid), len(self.r_grid)
        eta = np.ones((nzs, nzr, nr))
        cbar = np.full((nzs, nzr, nr), self.ssp.c_mean)
        direct = np.zeros((nzs, nzr, nr), bool)
        tol = 0.5 * self.cfg.ray_dr_m
        for i, zs in enumerate(self.zs_grid):
            rays = self._trace_fan(zs)
            for j, zr in enumerate(self.zr_grid):
                rc_all, sc_all, tc_all = [], [], []
                for (R, Z, S, TAU) in rays:
                    dzr = Z - zr
                    idx = np.where(np.sign(dzr[:-1]) * np.sign(dzr[1:]) < 0)[0]
                    if abs(zs - zr) < 1e-6:
                        # same depth: the launch point itself is a crossing; keep later ones only
                        idx = idx[idx > 2]
                    if idx.size == 0:
                        continue
                    w = dzr[idx] / (dzr[idx] - dzr[idx + 1])
                    rc_all.append(R[idx] + w * (R[idx + 1] - R[idx]))
                    sc_all.append(S[idx] + w * (S[idx + 1] - S[idx]))
                    tc_all.append(TAU[idx] + w * (TAU[idx + 1] - TAU[idx]))
                d_straight = np.sqrt(self.r_grid**2 + (zs - zr) ** 2)
                if rc_all:
                    rc = np.concatenate(rc_all); sc = np.concatenate(sc_all); tc = np.concatenate(tc_all)
                    order = np.argsort(rc)
                    rc, sc, tc = rc[order], sc[order], tc[order]
                    for k, rk in enumerate(self.r_grid):
                        lo = np.searchsorted(rc, rk - tol)
                        hi = np.searchsorted(rc, rk + tol)
                        if hi > lo:
                            m = lo + int(np.argmin(tc[lo:hi]))   # first arrival
                            eta[i, j, k] = max(1.0, sc[m] / d_straight[k])
                            cbar[i, j, k] = sc[m] / tc[m]
                            direct[i, j, k] = True
                if abs(zs - zr) < 1e-6:
                    # very short same-depth paths fall between launch angles: treat as straight
                    cz = float(self.ssp(np.array([zs]))[0])
                    short = self.r_grid < 300.0
                    eta[i, j, short] = 1.0
                    cbar[i, j, short] = cz
                    direct[i, j, short] = True
                # de-alias: an isolated non-direct bin between two direct bins is a fan gap, not a shadow
                d = direct[i, j]
                gap = np.zeros_like(d)
                gap[1:-1] = (~d[1:-1]) & d[:-2] & d[2:]
                if gap.any():
                    k = np.where(gap)[0]
                    eta[i, j, k] = 0.5 * (eta[i, j, k - 1] + eta[i, j, k + 1])
                    cbar[i, j, k] = 0.5 * (cbar[i, j, k - 1] + cbar[i, j, k + 1])
                    direct[i, j, k] = True
        return eta, cbar, direct

    # -- queries --------------------------------------------------------------------------
    def _idx(self, zs, zr, r):
        izs = np.abs(self.zs_grid[None, :] - np.asarray(zs, float)[:, None]).argmin(axis=1)
        izr = np.abs(self.zr_grid[None, :] - np.asarray(zr, float)[:, None]).argmin(axis=1)
        ir = np.clip(np.rint(np.asarray(r, float) / self.cfg.ray_dr_m).astype(int) - 1, 0, len(self.r_grid) - 1)
        return izs, izr, ir

    def query(self, zs, zr, r_h) -> RayQuery:
        """First-arrival properties for source depths zs, receiver depths zr, horizontal ranges r_h.

        Where no direct eigenray exists, eta/cbar describe the surface-reflected (image) path.
        All arguments broadcast to 1-D arrays of equal length.
        """
        zs = np.atleast_1d(np.asarray(zs, float))
        zr = np.atleast_1d(np.asarray(zr, float))
        r_h = np.atleast_1d(np.asarray(r_h, float))
        n = max(zs.size, zr.size, r_h.size)
        zs = np.broadcast_to(zs, (n,)); zr = np.broadcast_to(zr, (n,)); r_h = np.broadcast_to(r_h, (n,))
        izs, izr, ir = self._idx(zs, zr, r_h)
        eta = self.eta[izs, izr, ir].copy()
        cbar = self.cbar[izs, izr, ir].copy()
        direct = self.direct[izs, izr, ir].copy()
        # reflected fallback
        if (~direct).any():
            d_str = np.sqrt(r_h**2 + (zs - zr) ** 2)
            d_ref = np.sqrt(r_h**2 + (zs + zr) ** 2)
            eta_ref = np.maximum(1.0, d_ref / np.maximum(d_str, 1.0))
            eta[~direct] = eta_ref[~direct]
            cbar[~direct] = self.cref[izs[~direct], izr[~direct]]
        return RayQuery(eta=eta, cbar=cbar, direct=direct)


# ----------------------------------------------------------------------------- spatial Δc field
class SoundSpeedPerturbation:
    """Smooth random horizontal perturbation of the path-mean sound speed (m/s)."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        n = 21
        self.xg = np.linspace(-cfg.half_size_m * 1.2, cfg.half_size_m * 1.2, n)
        X, Y = np.meshgrid(self.xg, self.xg, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        K = np.exp(-0.5 * d2 / cfg.ssp_perturb_corr_m**2) + 1e-6 * np.eye(len(pts))
        L = np.linalg.cholesky(K)
        self.field = (L @ rng.standard_normal(len(pts))).reshape(n, n) * cfg.ssp_perturb_sigma_mps

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # bilinear interpolation
        xg = self.xg
        fx = np.clip((np.asarray(x) - xg[0]) / (xg[1] - xg[0]), 0, len(xg) - 1.001)
        fy = np.clip((np.asarray(y) - xg[0]) / (xg[1] - xg[0]), 0, len(xg) - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        wx, wy = fx - ix, fy - iy
        f = self.field
        return ((1 - wx) * (1 - wy) * f[ix, iy] + wx * (1 - wy) * f[ix + 1, iy]
                + (1 - wx) * wy * f[ix, iy + 1] + wx * wy * f[ix + 1, iy + 1])


# ----------------------------------------------------------------------------- sonar equation
def thorp_alpha_db_per_km(f_khz: float) -> float:
    f2 = f_khz**2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def transmission_loss_db(path_len_m: np.ndarray, f_khz: float) -> np.ndarray:
    r = np.maximum(np.asarray(path_len_m, float), 1.0)
    return 15.0 * np.log10(r) + thorp_alpha_db_per_km(f_khz) * r / 1000.0


def coates_noise_spectrum_db(f_khz: float, wind_speed_mps: float, shipping: float) -> float:
    """Ambient noise spectrum level [dB re 1 uPa^2/Hz] (Coates 1989 / Stojanovic 2007)."""
    f = f_khz
    nt = 17 - 30 * np.log10(f)
    ns = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f) - 60 * np.log10(f + 0.03)
    nw = 50 + 7.5 * np.sqrt(wind_speed_mps) + 20 * np.log10(f) - 40 * np.log10(f + 0.4)
    nth = -15 + 20 * np.log10(f)
    return float(10 * np.log10(10 ** (nt / 10) + 10 ** (ns / 10) + 10 ** (nw / 10) + 10 ** (nth / 10)))


def pd_energy_detector(snr_db: np.ndarray, pfa: np.ndarray | float, tb: float) -> np.ndarray:
    """Energy detector, Gaussian approximation: Pd = Q(Q^-1(Pfa) - sqrt(TB) * SNR_lin)."""
    snr_lin = 10 ** (np.asarray(snr_db, float) / 10.0)
    thr = norm.isf(np.asarray(pfa, float))
    return norm.sf(thr - np.sqrt(tb) * snr_lin)


def required_snr_db(pd: float, pfa: float, tb: float) -> float:
    snr_lin = (norm.isf(pfa) - norm.isf(pd)) / np.sqrt(tb)
    return float(10 * np.log10(max(snr_lin, 1e-12)))


class Acoustics:
    """Bundles NL, SL calibration and detection probability for the configured target pulse."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.nl_band_db = coates_noise_spectrum_db(cfg.f_khz, cfg.wind_speed_mps, cfg.shipping_level) \
            + 10 * np.log10(cfg.bandwidth_hz)
        if cfg.source_level_db is None:
            # calibrate SL so that a single node has Pd = 0.5 at PR in search mode (direct path)
            self.sl_db = self.nl_band_db + required_snr_db(0.5, cfg.pfa_search, cfg.time_bandwidth) \
                + float(transmission_loss_db(cfg.perception_radius_m, cfg.f_khz))
        else:
            self.sl_db = cfg.source_level_db

    def snr_db(self, path_len_m: np.ndarray, direct: np.ndarray) -> np.ndarray:
        tl = transmission_loss_db(path_len_m, self.cfg.f_khz)
        tl = tl + np.where(direct, 0.0, self.cfg.reflected_tl_penalty_db)
        return self.sl_db - tl - self.nl_band_db

    def pd(self, path_len_m: np.ndarray, direct: np.ndarray, pfa: np.ndarray) -> np.ndarray:
        return pd_energy_detector(self.snr_db(path_len_m, direct), pfa, self.cfg.time_bandwidth)
