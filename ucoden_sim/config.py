"""Simulation configuration for the task-cooperation study (Fig. 3/4/5).

Every model parameter used in the article appears here with its default value.
The three cooperation switches are ``c1_comm_loc``, ``c2_det_loc_track`` and ``c3_env``.
Scenario presets (S0, S0', S1-S4) live in :mod:`ucoden_sim.scenarios`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


@dataclass
class SimConfig:
    # ------------------------------------------------------------------ time
    duration_h: float = 24.0          # simulated time
    dt_s: float = 10.0                # integration step for drift / energy
    call_period_s: float = 60.0       # tagged target emits one pulse per period
    n_mc: int = 100                   # Monte-Carlo runs
    seed: int = 0

    # ------------------------------------------------------ area and network
    half_size_m: float = 5000.0       # area is [-half, half]^2  (10 km x 10 km)
    tier_depths_m: Tuple[float, ...] = (500.0, 1500.0)
    seabed_depth_m: float = 2000.0
    layout: str = "square"            # "square" | "triangle" | "hexagon"
    spacing_m: float = 1000.0         # IR, inter-node spacing within a tier
    perception_radius_m: float = 2000.0   # PR, nominal 50 % single-node detection range
    comm_range_m: float = 2000.0      # one-hop acoustic communication range
    n_gnss_chains: int = 9            # chains whose surface buoy carries GNSS
    gnss_sigma_m: float = 5.0         # horizontal position accuracy on GNSS chains
    depth_sensor_sigma_m: float = 5.0 # pressure-sensor depth accuracy

    # ------------------------------------------------------------ node drift
    drift_sigma_m_per_sqrt_h: float = 20.0   # random-walk std of chain-top excursion
    drift_bound_frac_ir: float = 0.1         # |excursion of upper tier| <= frac * IR
    tidal_amp_m: float = 60.0                # rotary tidal excursion amplitude (chain top)
    tidal_period_h: float = 12.42
    drift_corr_len_m: float = 3000.0         # spatial correlation of the random component

    # -------------------------------------------------------------- acoustics
    f_khz: float = 0.5                # centre frequency of the target pulse
    bandwidth_hz: float = 500.0
    pulse_s: float = 1.0              # pulse duration (energy-detector integration time)
    wind_speed_mps: float = 7.0       # sea state 3
    shipping_level: float = 0.5       # Wenz/Coates shipping factor (0..1)
    source_level_db: Optional[float] = None  # None -> calibrated so Pd=0.5 at PR
    pfa_search: float = 1e-3
    pfa_track: float = 1e-2
    toa_sigma_s: float = 1e-3         # time-of-arrival noise (clock + sampling)
    range_model_frac: float = 0.005   # extra ranging noise, fraction of range
    ssp_perturb_sigma_mps: float = 3.0      # spatial sound-speed perturbation
    ssp_perturb_corr_m: float = 2000.0
    reflected_tl_penalty_db: float = 6.0    # extra loss when only a reflected path exists
    nominal_c_mps: float = 1500.0           # what the un-cooperative network assumes
    c_est_sigma_mps: float = 2.0            # residual after local CTD calibration (C3 on)
    eta_est_rel_sigma: float = 0.05         # residual on (eta-1) after ray correction (C3 on)

    # -------------------------------------------------------------- ray table
    ray_r_max_m: float = 7000.0
    ray_dr_m: float = 50.0
    ray_ds_m: float = 10.0
    ray_angle_max_deg: float = 85.0
    ray_angle_step_deg: float = 0.25
    ray_fine_angle_deg: float = 8.0       # denser fan near horizontal for same-depth eigenrays
    ray_fine_step_deg: float = 0.02
    ray_ultra_angle_deg: float = 2.0      # densest fan for weak-gradient (deep tier) eigenrays
    ray_ultra_step_deg: float = 0.004
    target_depth_grid_m: Tuple[float, ...] = tuple(float(z) for z in range(100, 1600, 100))

    # ------------------------------------------------------------------ modem
    p_tx_w: float = 10.0
    rate_bps: float = 5000.0
    p_rx_w: float = 0.8
    p_sleep_w: float = 0.01
    t_preamble_s: float = 0.2
    e_session_j: float = 20.0         # wake-up / sync / handshake per transmission
    link_loss_normal: float = 0.05
    link_loss_shadow: float = 0.7     # link without a direct eigenray
    max_retx: int = 3
    bytes_det_report: int = 32
    bytes_range_report: int = 16
    bytes_ping: int = 8
    bytes_track_update: int = 24
    bytes_status: int = 24            # periodic health/status packet (background traffic)
    bytes_ctd: int = 16

    # ------------------------------------------------- cooperation switches
    c1_comm_loc: bool = False         # (1) communication <-> localization
    c2_det_loc_track: bool = False    # (2) detection -> localization -> tracking -> detection
    c3_env: bool = False              # (3) environment sensing -> localization & communication
    calibrate_nodes: bool = True      # False gives S0' (design positions, never calibrated)

    # -------------------------------------------------- (1) parameters
    ping_period_min: float = 30.0     # dedicated ranging pings when C1 is off
    calib_update_min: float = 10.0    # node-position re-estimation cadence
    calib_window_h: float = 0.5       # sliding window of inter-node ranges (short: moorings move)
    calib_process_sigma_m: float = 15.0   # random-walk prior on the previous estimate per update
    calib_outlier_sigma: float = 4.0      # drop links whose normalised residual exceeds this

    # -------------------------------------------------- (2) parameters
    duty_search: float = 0.5
    duty_track_in: float = 1.0
    duty_track_out: float = 0.2
    k_of_n: int = 2
    k_pair_dist_m: float = 3000.0     # detections closer than this count as one cluster
    gdop_max: float = 6.0
    k_poll: int = 6                   # nodes polled for ranges in track mode
    fix_valid_err_m: float = 100.0
    track_lost_epochs: int = 5
    gate_sigma_mult: float = 3.0
    group_radius_m: float = 4000.0    # uncoordinated grouping radius when C2 is off (2*PR)

    # -------------------------------------------------- (3) parameters
    ctd_period_min: float = 30.0      # CTD report period (rides on status packets when piggybacked)
    ctd_piggyback: bool = True        # append CTD to any packet sent in the period
    status_period_min: float = 30.0   # periodic status packet (background data traffic)

    # ---------------------------------------------------------- bookkeeping
    tag: str = "custom"

    # derived helpers -----------------------------------------------------
    @property
    def n_steps(self) -> int:
        return int(round(self.duration_h * 3600.0 / self.dt_s))

    @property
    def steps_per_call(self) -> int:
        return max(1, int(round(self.call_period_s / self.dt_s)))

    @property
    def time_bandwidth(self) -> float:
        return self.pulse_s * self.bandwidth_hz

    def with_(self, **kw) -> "SimConfig":
        """Return a modified copy (dataclasses.replace wrapper)."""
        return replace(self, **kw)
