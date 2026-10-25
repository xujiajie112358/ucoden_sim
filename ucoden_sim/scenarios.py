"""Scenario presets: S0 independent (pinged calibration), S0' independent without calibration,
S1-S3 one cooperation pair each, S4 all three."""
from __future__ import annotations

from typing import Dict

from .config import SimConfig

SCENARIO_ORDER = ["S0", "S0p", "S1", "S2", "S3", "S4"]
SCENARIO_LABELS = {
    "S0": "S0 independent",
    "S0p": "S0' independent, no calibration",
    "S1": "S1 comm<->loc",
    "S2": "S2 det-loc-track loop",
    "S3": "S3 env->loc&comm",
    "S4": "S4 all three",
}


def scenario(name: str, base: SimConfig | None = None) -> SimConfig:
    base = base or SimConfig()
    flags: Dict[str, dict] = {
        "S0": dict(c1_comm_loc=False, c2_det_loc_track=False, c3_env=False, calibrate_nodes=True),
        "S0p": dict(c1_comm_loc=False, c2_det_loc_track=False, c3_env=False, calibrate_nodes=False),
        "S1": dict(c1_comm_loc=True, c2_det_loc_track=False, c3_env=False, calibrate_nodes=True),
        "S2": dict(c1_comm_loc=False, c2_det_loc_track=True, c3_env=False, calibrate_nodes=True),
        "S3": dict(c1_comm_loc=False, c2_det_loc_track=False, c3_env=True, calibrate_nodes=True),
        "S4": dict(c1_comm_loc=True, c2_det_loc_track=True, c3_env=True, calibrate_nodes=True),
    }
    if name not in flags:
        raise KeyError(f"unknown scenario {name!r}; choose from {list(flags)}")
    return base.with_(tag=name, **flags[name])
