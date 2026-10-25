"""Tagged target: random-waypoint motion with a scheduled acoustic pulse every call period."""
from __future__ import annotations

import numpy as np

from .config import SimConfig


class Target:
    """Acoustically tagged target (e.g. a tagged whale) moving through the detection area.

    The tag emits a 1-s pulse at known, synchronised epochs, so the network measures absolute
    ranges from times of arrival (the cooperative-emitter assumption used in the OJCOMS case
    study and the JSAC-line work). Depth performs a bounded random walk.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator, n_epochs: int, dt_s: float,
                 z_min: float = 100.0, z_max: float = 1000.0, margin_m: float = 500.0):
        self.cfg = cfg
        half = cfg.half_size_m - margin_m
        pos = np.zeros((n_epochs, 3))
        p = np.array([rng.uniform(-half, half), rng.uniform(-half, half)])
        z = rng.uniform(z_min, z_max)
        wp = np.array([rng.uniform(-half, half), rng.uniform(-half, half)])
        speed = rng.uniform(1.0, 3.0)
        for k in range(n_epochs):
            d = wp - p
            dist = np.linalg.norm(d)
            step = speed * dt_s
            if dist <= step:
                p = wp.copy()
                wp = np.array([rng.uniform(-half, half), rng.uniform(-half, half)])
                speed = rng.uniform(1.0, 3.0)
            else:
                p = p + d / dist * step
            z = float(np.clip(z + rng.normal(0, 3.0), z_min, z_max))
            pos[k] = (p[0], p[1], z)
        self.pos = pos

    def position(self, epoch: int) -> np.ndarray:
        return self.pos[epoch]
