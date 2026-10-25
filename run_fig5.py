#!/usr/bin/env python3
"""Fig. 5 - node-position calibration cost: dedicated ranging pings versus opportunistic
ranging from packets the network sends anyway (cooperation pair 1), with pairs 2 and 3 on.

Two sweeps, same figure:
  * pings (C1 off):  ping period in {5, 10, 30, 60, 120} min
  * opportunistic (C1 on): background status-packet period in {120, 60, 30, 15, 10} min
x-axis: transmit energy per node-hour (the price), y-axis: node position RMSE (NPE).
A second panel-free companion CSV holds the target localisation error for the text.

Usage:  python run_fig5.py --mc 20 --hours 12 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import os

# one BLAS thread per worker process: the Monte-Carlo runs are parallelised across processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from ucoden_sim.config import SimConfig
from ucoden_sim.plotting import SERIES, plt, save, style
from ucoden_sim.simulator import run_mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=20)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results")
    ap.add_argument("--ping-periods", nargs="*", type=float, default=[5, 10, 30, 60, 120])
    ap.add_argument("--status-periods", nargs="*", type=float, default=[120, 60, 30, 15, 10])
    ap.add_argument("--from-csv", default=None, help="replot from an existing fig5_table.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    if args.from_csv:
        with open(args.from_csv) as f:
            for r in csv.DictReader(f):
                rows.append({k: (v if k == "mode" else float(v)) for k, v in r.items()})
        plot(rows, args.out)
        return
    for tp in args.ping_periods:
        cfg = SimConfig(duration_h=args.hours, c1_comm_loc=False, c2_det_loc_track=True, c3_env=True,
                        calibrate_nodes=True, ping_period_min=tp, tag=f"ping{tp:g}")
        m = run_mc(cfg, n_mc=args.mc, workers=args.workers)["mean"]
        rows.append(dict(mode="pings", period_min=tp, npe_rmse_m=m["npe_rmse_m"], median_err_m=m["median_err_m"],
                         p_loc=m["p_loc"], energy_tx_j_h=m["energy_tx_j_h"], airtime_pct=m["airtime_pct"],
                         energy_j_per_node_h=m["energy_j_per_node_h"]))
        print(f"pings  T={tp:5.0f} min  NPE={m['npe_rmse_m']:.1f} m  tx-energy={m['energy_tx_j_h']:.0f} J/node/h  "
              f"airtime={m['airtime_pct']:.2f}%  med_err={m['median_err_m']:.1f}")
    for ts in args.status_periods:
        cfg = SimConfig(duration_h=args.hours, c1_comm_loc=True, c2_det_loc_track=True, c3_env=True,
                        calibrate_nodes=True, status_period_min=ts, tag=f"opp{ts:g}")
        m = run_mc(cfg, n_mc=args.mc, workers=args.workers)["mean"]
        rows.append(dict(mode="opportunistic", period_min=ts, npe_rmse_m=m["npe_rmse_m"], median_err_m=m["median_err_m"],
                         p_loc=m["p_loc"], energy_tx_j_h=m["energy_tx_j_h"], airtime_pct=m["airtime_pct"],
                         energy_j_per_node_h=m["energy_j_per_node_h"]))
        print(f"opp    T={ts:5.0f} min  NPE={m['npe_rmse_m']:.1f} m  tx-energy={m['energy_tx_j_h']:.0f} J/node/h  "
              f"airtime={m['airtime_pct']:.2f}%  med_err={m['median_err_m']:.1f}")

    with open(os.path.join(args.out, "fig5_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, args.out)


def plot(rows, out):
    style()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    series = [("pings", "Dedicated ranging pings", "o", SERIES[0]),
              ("opportunistic", "Opportunistic ranging (Loop 1)", "s", SERIES[1])]
    for mode, lab, mk, col in series:
        sub = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["energy_tx_j_h"])
        xs = [r["energy_tx_j_h"] for r in sub]
        ys = [r["npe_rmse_m"] for r in sub]
        ax.plot(xs, ys, marker=mk, ms=5, lw=1.8, color=col, label=lab, zorder=3)
        if mode == "pings":
            for r in sub:
                ax.annotate(f"{r['period_min']:g}", (r["energy_tx_j_h"], r["npe_rmse_m"]), fontsize=6.5, color=col,
                            textcoords="offset points", xytext=(5, 7), ha="left", va="bottom")
        else:
            # the opportunistic points sit within a few joules of each other: put every label on its own
            # horizontal slot below the curve and draw a leader line from the label to its marker
            y_lab = min(ys) - 0.28 * max(ys)
            slot_dx = 14.0   # joules between label slots
            x_lab = xs[0] - 6.0
            for k, r in enumerate(sub):
                x_lab = max(x_lab + slot_dx, r["energy_tx_j_h"] - 4.0) if k else x_lab
                ax.annotate(f"{r['period_min']:g}", (r["energy_tx_j_h"], r["npe_rmse_m"]), xytext=(x_lab, y_lab),
                            textcoords="data", fontsize=6.5, color=col, ha="center", va="top",
                            arrowprops=dict(arrowstyle="-", lw=0.6, color=col, shrinkA=0, shrinkB=3, alpha=0.9))
    ax.set_xlabel("Transmit energy per node-hour (J)")
    ax.set_ylabel("Node position RMSE (m)")
    ax.set_ylim(0, max(r["npe_rmse_m"] for r in rows) * 1.15)
    ax.set_xlim(left=70)
    ax.legend(fontsize=7, loc="upper right", title="labels: ping / status-packet period (min)", title_fontsize=6.5)
    save(fig, os.path.join(out, "fig5_calibration_cost"))
    print("saved", os.path.join(out, "fig5_calibration_cost.pdf"))


if __name__ == "__main__":
    main()
