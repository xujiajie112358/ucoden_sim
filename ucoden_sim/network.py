"""Fixed two-tier sensor network: layouts, mooring drift, node state and energy bookkeeping."""
from __future__ import annotations

import numpy as np

from .config import SimConfig


# ----------------------------------------------------------------------------- layouts
def make_layout(layout: str, spacing: float, half: float) -> np.ndarray:
    """Horizontal positions (M x 2) of the chains for one tier, clipped to the square area.

    * square   : square grid, nearest-neighbour distance = spacing
    * triangle : triangular lattice, nearest-neighbour distance = spacing
    * hexagon  : honeycomb (triangular lattice with one third of the sites removed),
                 nearest-neighbour distance = spacing
    """
    eps = 1e-6
    if layout == "square":
        g = np.arange(-half, half + eps, spacing)
        X, Y = np.meshgrid(g, g, indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], axis=1)
    if layout in ("triangle", "hexagon"):
        a = np.array([spacing, 0.0])
        b = np.array([0.5 * spacing, 0.5 * np.sqrt(3.0) * spacing])
        n = int(np.ceil(2 * half / spacing)) + 4
        pts, keep = [], []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                p = i * a + j * b
                if abs(p[0]) <= half + eps and abs(p[1]) <= half + eps:
                    if layout == "hexagon" and (i - j) % 3 == 0:
                        continue
                    pts.append(p)
        return np.array(pts)
    raise ValueError(f"unknown layout {layout!r}")


# ----------------------------------------------------------------------------- network
class Network:
    """Two-tier moored sensor network with drifting chains.

    Node i sits on chain ``chain[i]`` at tier ``tier[i]``. The chain top (surface buoy)
    excursion D_c(t) is a rotary tide plus a spatially correlated bounded random walk;
    a node at height h above the seabed is displaced by D_c(t) * h / H  (H = water depth).
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        xy = make_layout(cfg.layout, cfg.spacing_m, cfg.half_size_m)
        self.n_chains = len(xy)
        tiers = np.array(cfg.tier_depths_m, float)
        self.n_tiers = len(tiers)
        self.chain = np.repeat(np.arange(self.n_chains), self.n_tiers)
        self.tier = np.tile(np.arange(self.n_tiers), self.n_chains)
        self.n = self.n_chains * self.n_tiers
        self.design = np.column_stack([xy[self.chain, 0], xy[self.chain, 1], tiers[self.tier]])
        H = cfg.seabed_depth_m
        self.height_frac = (H - self.design[:, 2]) / H          # h_i / H
        self.chain_xy = xy

        # GNSS-referenced chains: nearest chains to a coarse k x k grid
        k = max(1, int(round(np.sqrt(cfg.n_gnss_chains))))
        g = np.linspace(-cfg.half_size_m * 0.8, cfg.half_size_m * 0.8, k)
        targets = np.array([[gx, gy] for gx in g for gy in g])
        gnss_chains = set()
        for tgt in targets:
            gnss_chains.add(int(np.argmin(((xy - tgt) ** 2).sum(1))))
        self.gnss_chain = np.zeros(self.n_chains, bool)
        self.gnss_chain[list(gnss_chains)] = True
        self.gnss_node = self.gnss_chain[self.chain]

        # drift model
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        K = np.exp(-0.5 * d2 / cfg.drift_corr_len_m**2) + 1e-6 * np.eye(self.n_chains)
        self._L = np.linalg.cholesky(K)
        self._walk = np.zeros((self.n_chains, 2))
        self._tide_phase = rng.uniform(0, 2 * np.pi)
        upper_frac = float(self.height_frac.max())
        self._bound_top = cfg.drift_bound_frac_ir * cfg.spacing_m / upper_frac
        self.top_disp = np.zeros((self.n_chains, 2))
        self.true_pos = self.design.copy()
        self.update_drift(0.0, 0.0)

        # depth sensors (pressure) and position estimates used by the algorithms
        self.depth_meas = self.true_pos[:, 2] + rng.normal(0, cfg.depth_sensor_sigma_m, self.n)
        self.est_pos = self.design.copy()
        self.est_pos[:, 2] = self.depth_meas
        self.refresh_gnss()

        # per-node operating state
        self.duty = np.full(self.n, cfg.duty_search)
        self.pfa = np.full(self.n, cfg.pfa_search)

        # energy [J] and channel bookkeeping
        self.e_listen = np.zeros(self.n)
        self.e_sleep = np.zeros(self.n)
        self.e_tx = np.zeros(self.n)
        self.tx_time = np.zeros(self.n)
        self.n_tx = np.zeros(self.n, int)

        # static neighbourhood (design geometry) for one-hop communication
        dd = np.sqrt(((self.design[:, None, :] - self.design[None, :, :]) ** 2).sum(-1))
        self.in_range = (dd <= cfg.comm_range_m) & ~np.eye(self.n, dtype=bool)
        self.neighbors = [np.where(self.in_range[i])[0] for i in range(self.n)]

    # -- drift --------------------------------------------------------------------------
    def update_drift(self, t_s: float, dt_s: float):
        cfg = self.cfg
        if dt_s > 0:
            sig = cfg.drift_sigma_m_per_sqrt_h * np.sqrt(dt_s / 3600.0)
            inc = self._L @ self.rng.standard_normal((self.n_chains, 2)) * sig
            self._walk += inc
            # bounded random walk: soft reflection at the bound
            norm = np.linalg.norm(self._walk, axis=1, keepdims=True)
            over = norm > self._bound_top
            if over.any():
                self._walk[over[:, 0]] *= (self._bound_top / norm[over[:, 0]])
        w = 2 * np.pi * t_s / (cfg.tidal_period_h * 3600.0) + self._tide_phase
        tide = cfg.tidal_amp_m * np.array([np.cos(w), np.sin(w)])
        self.top_disp = self._walk + tide[None, :]
        self.true_pos[:, 0] = self.design[:, 0] + self.top_disp[self.chain, 0] * self.height_frac
        self.true_pos[:, 1] = self.design[:, 1] + self.top_disp[self.chain, 1] * self.height_frac

    def refresh_gnss(self):
        """GNSS chains know their node positions (buoy GNSS + chain geometry) to gnss_sigma."""
        m = self.gnss_node
        self.est_pos[m, 0:2] = self.true_pos[m, 0:2] + self.rng.normal(0, self.cfg.gnss_sigma_m, (m.sum(), 2))

    def gnss_reference(self) -> np.ndarray:
        """Reference horizontal positions for GNSS nodes (noisy truth)."""
        m = self.gnss_node
        return self.true_pos[m, 0:2] + self.rng.normal(0, self.cfg.gnss_sigma_m, (m.sum(), 2))

    # -- energy -------------------------------------------------------------------------
    def accrue_idle_energy(self, dt_s: float):
        self.e_listen += self.cfg.p_rx_w * self.duty * dt_s
        self.e_sleep += self.cfg.p_sleep_w * (1.0 - self.duty) * dt_s

    def account_tx(self, i: int, duration_s: float):
        self.e_tx[i] += self.cfg.p_tx_w * duration_s + self.cfg.e_session_j
        self.tx_time[i] += duration_s
        self.n_tx[i] += 1

    # -- diagnostics --------------------------------------------------------------------
    def position_error(self) -> np.ndarray:
        return np.linalg.norm(self.est_pos[:, 0:2] - self.true_pos[:, 0:2], axis=1)

    def local_airtime(self, duration_s: float) -> np.ndarray:
        """Fraction of time the channel is busy as heard by each node (its one-hop neighbourhood)."""
        busy = self.tx_time.copy()
        for i in range(self.n):
            busy[i] += self.tx_time[self.neighbors[i]].sum()
        return busy / duration_s
