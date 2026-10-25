"""Acoustic packet traffic: airtime and energy accounting, link loss, shadow-aware relaying,
opportunistic ranging from received packets, periodic status / CTD / ping traffic."""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .environment import RayLUT, SoundSpeedPerturbation
from .localization import NodeCalibrator
from .network import Network


class Comm:
    def __init__(self, cfg: SimConfig, net: Network, lut: RayLUT, pert: SoundSpeedPerturbation,
                 calib: NodeCalibrator, rng: np.random.Generator):
        self.cfg, self.net, self.lut, self.pert, self.calib, self.rng = cfg, net, lut, pert, calib, rng
        n = net.n
        self.ctd_last = np.full(n, -1e9)
        self.status_next = rng.uniform(0, cfg.status_period_min * 60.0, n)
        self.ping_next = cfg.ping_period_min * 60.0
        # counters for diagnostics
        self.bytes_by_kind = {}
        self.n_relayed = 0
        self.n_failed = 0
        self.n_unicast = 0

    # -- primitives ---------------------------------------------------------------------
    def duration(self, nbytes: int) -> float:
        return self.cfg.t_preamble_s + nbytes * 8.0 / self.cfg.rate_bps

    def _link_props_true(self, i: int, recv: np.ndarray):
        P = self.net.true_pos
        dx = P[recv, 0] - P[i, 0]; dy = P[recv, 1] - P[i, 1]
        r_h = np.sqrt(dx**2 + dy**2)
        d = np.sqrt(r_h**2 + (P[recv, 2] - P[i, 2]) ** 2)
        q = self.lut.query(np.full(recv.size, P[i, 2]), P[recv, 2], r_h)
        cbar = q.cbar + self.pert(0.5 * (P[recv, 0] + P[i, 0]), 0.5 * (P[recv, 1] + P[i, 1]))
        return d, q.direct, q.eta, cbar

    def link_direct_est(self, i: int, j: int) -> bool:
        E = self.net.est_pos
        r_h = float(np.hypot(E[j, 0] - E[i, 0], E[j, 1] - E[i, 1]))
        return bool(self.lut.query([E[i, 2]], [E[j, 2]], [r_h]).direct[0])

    def _maybe_piggyback(self, i: int, nbytes: int, t: float) -> int:
        cfg = self.cfg
        if cfg.c3_env and cfg.ctd_piggyback and t - self.ctd_last[i] >= cfg.ctd_period_min * 60.0:
            self.ctd_last[i] = t
            self._count("ctd_piggyback", cfg.bytes_ctd)
            return nbytes + cfg.bytes_ctd
        return nbytes

    def _count(self, kind: str, nbytes: int):
        self.bytes_by_kind[kind] = self.bytes_by_kind.get(kind, 0) + nbytes

    def _deliver(self, i: int, nbytes: int, t: float, kind: str, ranging: bool) -> np.ndarray:
        """One physical transmission by node i. Returns indices of neighbours that decoded it."""
        cfg, net = self.cfg, self.net
        nbytes = self._maybe_piggyback(i, nbytes, t)
        self._count(kind, nbytes)
        net.account_tx(i, self.duration(nbytes))
        recv = net.neighbors[i]
        if recv.size == 0:
            return recv
        d, direct, eta, cbar = self._link_props_true(i, recv)
        loss = np.where(direct, cfg.link_loss_normal, cfg.link_loss_shadow)
        ok = self.rng.random(recv.size) >= loss
        got = recv[ok]
        if ranging and got.size:
            toa = eta[ok] * d[ok] / cbar[ok] + self.rng.normal(0, cfg.toa_sigma_s, got.size)
            if cfg.c3_env:
                E = net.est_pos
                r_h_est = np.hypot(E[got, 0] - E[i, 0], E[got, 1] - E[i, 1])
                q = self.lut.query(np.full(got.size, E[i, 2]), E[got, 2], r_h_est)
                c_est = q.cbar + self.rng.normal(0, cfg.c_est_sigma_mps, got.size)
                eta_est = 1.0 + (q.eta - 1.0) * (1.0 + self.rng.normal(0, cfg.eta_est_rel_sigma, got.size))
                d_hat = c_est * toa / np.maximum(eta_est, 1.0)
            else:
                d_hat = cfg.nominal_c_mps * toa
            for k, dh in zip(got, d_hat):
                self.calib.add(i, int(k), float(dh), t)
        return got

    # -- services -----------------------------------------------------------------------
    def broadcast(self, i: int, nbytes: int, t: float, kind: str = "bcast") -> np.ndarray:
        return self._deliver(i, nbytes, t, kind, ranging=self.cfg.c1_comm_loc)

    def unicast(self, i: int, j: int, nbytes: int, t: float, kind: str = "ucast") -> bool:
        """Deliver a packet from i to j with retransmissions; shadow-aware relay when C1 is on."""
        cfg, net = self.cfg, self.net
        if i == j:
            return True
        self.n_unicast += 1
        path = [i, j]
        if cfg.c1_comm_loc and net.in_range[i, j] and not self.link_direct_est(i, j):
            common = np.where(net.in_range[i] & net.in_range[j])[0]
            best, best_d = None, float("inf")
            E = net.est_pos
            for k in common:
                if self.link_direct_est(i, int(k)) and self.link_direct_est(int(k), j):
                    dk = np.linalg.norm(E[k] - E[i]) + np.linalg.norm(E[j] - E[k])
                    if dk < best_d:
                        best, best_d = int(k), dk
            if best is not None:
                path = [i, best, j]
                self.n_relayed += 1
        elif not net.in_range[i, j]:
            # out of one-hop range: pick the common neighbour closest to the straight line
            common = np.where(net.in_range[i] & net.in_range[j])[0]
            if common.size == 0:
                self.n_failed += 1
                return False
            E = net.est_pos
            dk = np.linalg.norm(E[common] - E[i], axis=1) + np.linalg.norm(E[j] - E[common], axis=1)
            path = [i, int(common[np.argmin(dk)]), j]
        for a, b in zip(path[:-1], path[1:]):
            delivered = False
            for _ in range(cfg.max_retx):
                got = self._deliver(a, nbytes, t, kind, ranging=cfg.c1_comm_loc)
                if b in got:
                    delivered = True
                    break
            if not delivered:
                self.n_failed += 1
                return False
        return True

    def periodic(self, t: float):
        """Background status packets, standalone CTD reports and dedicated ranging pings."""
        cfg, net = self.cfg, self.net
        due = np.where(self.status_next <= t)[0]
        for i in due:
            self.broadcast(int(i), cfg.bytes_status, t, kind="status")
            self.status_next[i] += cfg.status_period_min * 60.0
        if cfg.c3_env:
            # standalone CTD report only when no packet could carry it for 1.5 periods
            slack = 1.5 if cfg.ctd_piggyback else 1.0
            ctd_due = np.where(t - self.ctd_last >= slack * cfg.ctd_period_min * 60.0)[0]
            for i in ctd_due:
                self.ctd_last[i] = t
                self._deliver(int(i), cfg.bytes_ctd, t, "ctd", ranging=cfg.c1_comm_loc)
        if (not cfg.c1_comm_loc) and cfg.calibrate_nodes and t >= self.ping_next:
            for i in range(net.n):
                self._deliver(i, cfg.bytes_ping, t, "ping", ranging=True)
            self.ping_next += cfg.ping_period_min * 60.0
