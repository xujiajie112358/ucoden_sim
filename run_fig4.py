#!/usr/bin/env python3
"""Fig. 4 - detection-localisation coupling: P_CD, P_ava, P_loc versus number of nodes for
three fixed layouts (triangle, quadrilateral/square, hexagon), all cooperation switches on (S4).

P_CD  : fraction of epochs with a collaborative detection that contains a true detection
P_ava : fraction of epochs where >= 4 ranges arrived with GDOP <= gdop_max (geometry available)
P_loc : fraction of epochs with a fix error <= fix_valid_err_m
By construction P_loc <= P_ava <= P_CD; the figure shows the gaps between them.

Usage:  python run_fig4.py --mc 30 --hours 12 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import os

# one BLAS thread per worker process: the Monte-Carlo runs are parallelised across processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from ucoden_sim.config import SimConfig
from ucoden_sim.network import make_layout
from ucoden_sim.plotting import SERIES, plt, save, style
from ucoden_sim.scenarios import scenario
from ucoden_sim.simulator import run_mc

LAYOUTS = [("triangle", "Triangle"), ("square", "Quadrilateral"), ("hexagon", "Hexagon")]
LEVELS = [("p_cd", r"$P_{\mathrm{CD}}$", "-"), ("p_ava", r"$P_{\mathrm{ava}}$", "--"), ("p_loc", r"$P_{\mathrm{loc}}$", ":")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=30)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results")
    ap.add_argument("--spacings", nargs="*", type=float, default=[1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000])
    ap.add_argument("--from-csv", default=None, help="replot from an existing fig4_table.csv")
    ap.add_argument("--comm-range-factor", type=float, default=1.6,
                    help="one-hop communication range = max(comm_range_m, factor * spacing), so that connectivity is not the limiting factor of the sweep")
    ap.add_argument("--scenario", default="S4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    if args.from_csv:
        with open(args.from_csv) as f:
            for r in csv.DictReader(f):
                rows.append({k: (v if k == "layout" else float(v)) for k, v in r.items()})
        plot(rows, args.out)
        return
    for layout, _ in LAYOUTS:
        for sp in args.spacings:
            base = SimConfig(duration_h=args.hours, layout=layout, spacing_m=sp)
            base = base.with_(comm_range_m=max(base.comm_range_m, args.comm_range_factor * sp))
            n_nodes = len(make_layout(layout, sp, base.half_size_m)) * len(base.tier_depths_m)
            res = run_mc(scenario(args.scenario, base), n_mc=args.mc, workers=args.workers)
            m = res["mean"]
            rows.append(dict(layout=layout, spacing_m=sp, n_nodes=n_nodes, p_cd=m["p_cd"], p_ava=m["p_ava"],
                             p_loc=m["p_loc"], median_err_m=m["median_err_m"], energy=m["energy_j_per_node_h"]))
            print(f"{layout:9s} IR={sp:5.0f} m  N={n_nodes:4d}  P_CD={m['p_cd']:.3f}  P_ava={m['p_ava']:.3f}  "
                  f"P_loc={m['p_loc']:.3f}  med_err={m['median_err_m']:.1f} m")

    with open(os.path.join(args.out, "fig4_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, args.out)


def plot(rows, out):
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for li, (layout, lab) in enumerate(LAYOUTS):
        sub = sorted([r for r in rows if r["layout"] == layout], key=lambda r: r["spacing_m"])
        x = [r["spacing_m"] / 1000.0 for r in sub]
        for key, name, ls in LEVELS:
            ax.plot(x, [r[key] for r in sub], ls=ls, color=SERIES[li], lw=1.8, marker="o", ms=3.0,
                    label=f"{lab} {name}")
    ax.set_xlabel("Node spacing (km)")
    ax.set_ylabel("Probability per epoch")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), columnspacing=1.6)
    save(fig, os.path.join(out, "fig4_hierarchy"))
    print("saved", os.path.join(out, "fig4_hierarchy.pdf"))


if __name__ == "__main__":
    main()
