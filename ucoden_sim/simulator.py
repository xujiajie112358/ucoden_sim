"""Main simulation loop: one Monte-Carlo run of the two-tier UCODEN tracking a tagged target,
with the three cooperation loops switchable (see README.md)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from .comm import Comm
from .config import SimConfig
from .detection import decide_trigger, detect_epoch
from .environment import Acoustics, RayLUT, SoundSpeedPerturbation, SoundSpeedProfile
from .localization import NodeCalibrator, measured_toa, ranges_from_toa, select_nodes, wls_fix
from .network import Network
from .target import Target
from .tracking import Tracker, apply_policy

_ENV_CACHE: Dict[tuple, tuple] = {}


def _env(cfg: SimConfig):
    key = (cfg.seabed_depth_m, tuple(cfg.tier_depths_m), tuple(cfg.target_depth_grid_m), cfg.ray_r_max_m,
           cfg.ray_dr_m, cfg.ray_ds_m, cfg.ray_angle_max_deg, cfg.ray_angle_step_deg,
           cfg.ray_fine_angle_deg, cfg.ray_fine_step_deg, cfg.ray_ultra_angle_deg, cfg.ray_ultra_step_deg)
    if key not in _ENV_CACHE:
        ssp = SoundSpeedProfile(cfg.seabed_depth_m)
        _ENV_CACHE[key] = (ssp, RayLUT(cfg, ssp))
    return _ENV_CACHE[key]


def run_once(cfg: SimConfig, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    ssp, lut = _env(cfg)
    pert = SoundSpeedPerturbation(cfg, rng)
    ac = Acoustics(cfg)
    net = Network(cfg, rng)
    dt = cfg.call_period_s
    n_epochs = int(round(cfg.duration_h * 3600.0 / dt))
    tgt = Target(cfg, rng, n_epochs, dt)
    calib = NodeCalibrator(cfg, net)
    comm = Comm(cfg, net, lut, pert, calib, rng)
    trk = Tracker(cfg, dt)

    # per-epoch records
    triggered = np.zeros(n_epochs, bool)
    false_trig = np.zeros(n_epochs, bool)
    true_det = np.zeros(n_epochs, bool)        # at least one true detection reported
    geom_ok = np.zeros(n_epochs, bool)         # >= 4 ranges delivered with GDOP <= gdop_max
    fixed = np.zeros(n_epochs, bool)
    err = np.full(n_epochs, np.nan)
    gd = np.full(n_epochs, np.nan)
    n_used = np.zeros(n_epochs, int)
    npe = np.zeros(n_epochs)
    busy_near = 0.0                            # channel time spent within comm range of the target
    tx_prev = net.tx_time.copy()
    attempts = 0
    rejected = 0
    implausible = 0

    next_calib = cfg.calib_update_min * 60.0
    apply_policy(cfg, net, trk)

    for k in range(n_epochs):
        t = k * dt
        net.update_drift(t, dt if k > 0 else 0.0)
        net.accrue_idle_energy(dt)
        x_true = tgt.position(k)

        # ---- background traffic, CTD, pings
        comm.periodic(t)

        # ---- detection
        trk.predict()
        det = detect_epoch(cfg, net, ac, lut, pert, x_true, rng)
        trig = decide_trigger(cfg, net, det)
        fix = None
        if trig.positive:
            triggered[k] = True
            false_trig[k] = trig.all_false
            true_det[k] = not trig.all_false
            members = trig.members
            E = net.est_pos
            track_mode = cfg.c2_det_loc_track and trk.alive
            x_ref = trk.pos if track_mode else np.array([E[members, 0].mean(), E[members, 1].mean(), 500.0])
            head = int(members[np.argmin(np.linalg.norm(E[members] - x_ref[None, :], axis=1))])

            # ---- reporting / polling
            if track_mode:
                polled = select_nodes(E, members, x_ref, cfg.k_poll)
                comm.broadcast(head, cfg.bytes_track_update, t, kind="poll")
                reporters = polled
            else:
                reporters = members
            got = [head] if head in reporters else []
            for i in reporters:
                i = int(i)
                if i == head:
                    continue
                if comm.unicast(i, head, cfg.bytes_det_report, t, kind="report"):
                    got.append(i)
            U = np.array(sorted(set(got)), int)
            if cfg.c2_det_loc_track and not track_mode and U.size > cfg.k_poll:
                U = select_nodes(E, U, x_ref, cfg.k_poll)

            # ---- ranges and fix (two passes when the ray table is used)
            if U.size >= 3:
                attempts += 1
                toa = measured_toa(cfg, det.toa_true[U], det.is_false[U], rng)
                zs_guess = float(x_ref[2])
                r_h_est = np.hypot(E[U, 0] - x_ref[0], E[U, 1] - x_ref[1])
                d_hat, sig = ranges_from_toa(cfg, lut, toa, E[U, 2], r_h_est, zs_guess, rng)
                fix = wls_fix(E[U], d_hat, sig, zs_guess, x0=trk.pos if track_mode else None)
                if fix is not None and cfg.c3_env and U.size >= 4:
                    r_h_est = np.hypot(E[U, 0] - fix.x[0], E[U, 1] - fix.x[1])
                    d_hat, sig = ranges_from_toa(cfg, lut, toa, E[U, 2], r_h_est, float(fix.x[2]), rng)
                    fix = wls_fix(E[U], d_hat, sig, float(fix.x[2]), x0=fix.x)
                if fix is not None:
                    geom_ok[k] = fix.n_used >= 4 and fix.gdop <= cfg.gdop_max
                    plausible = (np.all(np.isfinite(fix.x)) and abs(fix.x[0]) <= cfg.half_size_m + 1000.0
                                 and abs(fix.x[1]) <= cfg.half_size_m + 1000.0
                                 and 0.0 <= fix.x[2] <= cfg.seabed_depth_m)
                    if not plausible:
                        implausible += 1
                        fix = None
                    elif cfg.c2_det_loc_track and not geom_ok[k]:
                        rejected += 1
                        fix = None
            if fix is not None:
                fixed[k] = True
                err[k] = float(np.linalg.norm(fix.x - x_true))
                gd[k] = fix.gdop
                n_used[k] = fix.n_used
                if cfg.c2_det_loc_track:
                    R = fix.cov[0:3, 0:3] + np.eye(3) * 5.0**2
                    trk.update(fix.x, R)
                    n_b = 1 + int(trk.gate_radius() > cfg.comm_range_m)
                    for _ in range(n_b):
                        comm.broadcast(head, cfg.bytes_track_update, t, kind="track_update")
        if fix is None:
            trk.miss()
        apply_policy(cfg, net, trk)

        # ---- node-position calibration
        if cfg.calibrate_nodes and t >= next_calib:
            net.refresh_gnss()
            calib.solve(t)
            next_calib += cfg.calib_update_min * 60.0
        elif not cfg.calibrate_nodes and t >= next_calib:
            net.refresh_gnss()
            next_calib += cfg.calib_update_min * 60.0
        npe[k] = float(np.sqrt(np.mean(net.position_error() ** 2)))
        d_tx = net.tx_time - tx_prev
        tx_prev = net.tx_time.copy()
        near = np.linalg.norm(net.true_pos - x_true[None, :], axis=1) <= cfg.comm_range_m
        busy_near += float(d_tx[near].sum())

    # ---- metrics
    dur_s = n_epochs * dt
    valid = fixed & (err <= cfg.fix_valid_err_m)
    e_total = net.e_listen + net.e_sleep + net.e_tx
    out = dict(
        rmse_m=float(np.sqrt(np.nanmean(err[fixed] ** 2))) if fixed.any() else np.nan,
        median_err_m=float(np.nanmedian(err[fixed])) if fixed.any() else np.nan,
        p_loc=float(valid.mean()),
        p_fix=float(fixed.mean()),
        p_cd=float(true_det.mean()),
        p_ava=float(geom_ok.mean()),
        p_trigger=float(triggered.mean()),
        false_trigger_rate=float(false_trig.mean()),
        airtime_pct=float(net.local_airtime(dur_s).mean() * 100.0),
        airtime_near_target_pct=float(busy_near / dur_s * 100.0),
        energy_j_per_node_h=float(e_total.mean() / cfg.duration_h),
        energy_listen_j_h=float(net.e_listen.mean() / cfg.duration_h),
        energy_tx_j_h=float(net.e_tx.mean() / cfg.duration_h),
        tx_per_node_h=float(net.n_tx.mean() / cfg.duration_h),
        npe_rmse_m=float(npe[n_epochs // 4:].mean()),
        attempts_per_h=attempts / cfg.duration_h,
        rejected_per_h=rejected / cfg.duration_h,
        implausible_per_h=implausible / cfg.duration_h,
        err_p95_m=float(np.nanpercentile(err[fixed], 95)) if fixed.any() else np.nan,
        relayed=comm.n_relayed,
        failed_deliveries=comm.n_failed,
        unicasts=comm.n_unicast,
        delivery_failure_pct=100.0 * comm.n_failed / max(comm.n_unicast, 1),
        failed_per_h=comm.n_failed / cfg.duration_h,
        mean_gdop=float(np.nanmean(np.where(np.isfinite(gd[fixed]), gd[fixed], np.nan))) if fixed.any() else np.nan,
        mean_n_used=float(n_used[fixed].mean()) if fixed.any() else np.nan,
    )
    for kind, nb in comm.bytes_by_kind.items():
        out[f"bytes_{kind}"] = nb
    return out


def _worker(args):
    cfg_dict, seed = args
    return run_once(SimConfig(**cfg_dict), seed)


def run_mc(cfg: SimConfig, n_mc: int | None = None, workers: int = 1, seed0: int | None = None) -> Dict[str, Dict[str, float]]:
    """Monte-Carlo wrapper. Returns {'mean': {...}, 'std': {...}, 'runs': [...]} ."""
    n_mc = n_mc or cfg.n_mc
    seed0 = cfg.seed if seed0 is None else seed0
    seeds = [seed0 + i for i in range(n_mc)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            runs: List[Dict[str, float]] = list(ex.map(_worker, [(asdict(cfg), s) for s in seeds]))
    else:
        runs = [run_once(cfg, s) for s in seeds]
    keys = sorted(set().union(*[r.keys() for r in runs]))
    mean = {k: float(np.nanmean([r.get(k, np.nan) for r in runs])) for k in keys}
    std = {k: float(np.nanstd([r.get(k, np.nan) for r in runs])) for k in keys}
    return {"mean": mean, "std": std, "runs": runs}
