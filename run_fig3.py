#!/usr/bin/env python3
"""Fig. 3 - independent vs cooperative operation (S0, S0', S1-S4).

Produces results/fig3_table.csv (all metrics, mean and std over Monte-Carlo runs) and
results/fig3_bars.{pdf,png}: a single-panel grouped bar chart of four lower-is-better
metrics normalised to the S0 baseline (thin error bars = +-1 std over runs):
    localization error (median), miss probability (1 - P_loc),
    third metric (--metric3: airtime | failures), energy per node-hour.

Usage:  python run_fig3.py --mc 100 --hours 24 --workers 8
Quick:  python run_fig3.py --mc 3 --hours 3 --workers 3
Replot: python run_fig3.py --from-csv results/fig3_table.csv --metric3 failures
"""
from __future__ import annotations

import argparse
import csv
import os

# one BLAS thread per worker process: the Monte-Carlo runs are parallelised across processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402

import numpy as np  # noqa: E402

from ucoden_sim.config import SimConfig  # noqa: E402
from ucoden_sim.plotting import NEUTRAL, SERIES, plt, save, style  # noqa: E402

# bar colours: loops keep the colours of Fig. 1/2 (S1 blue, S2 orange, S3 aqua); S0' grey; S4 violet
COLORS = {"S0p": "#9a9890", "S1": SERIES[0], "S2": SERIES[1], "S3": SERIES[2], "S4": SERIES[6]}
from ucoden_sim.scenarios import SCENARIO_LABELS, SCENARIO_ORDER, scenario  # noqa: E402
from ucoden_sim.simulator import run_mc  # noqa: E402

THIRD = {
    "airtime": ("airtime_near_target_pct", "Channel occupancy\nnear target", " %", lambda m: m["airtime_near_target_pct"]),
    "failures": ("delivery_failure_pct", "Report delivery\nfailures", " %", lambda m: m["delivery_failure_pct"]),
}


def metrics_for(metric3: str):
    k3, lab3, unit3, f3 = THIRD[metric3]
    return [  # (key, label, unit, lower-is-better value, 1-std over runs)
        ("median_err_m", "Localization error\n(median)", " m", lambda m: m["median_err_m"], lambda s: s["median_err_m"]),
        ("miss_prob", "Miss probability\n($1-P_{\\mathrm{loc}}$)", "", lambda m: 1.0 - m["p_loc"], lambda s: s["p_loc"]),
        (k3, lab3, unit3, f3, lambda s, k3=k3: s[k3]),
        ("energy_j_per_node_h", "Energy per\nnode-hour", " J", lambda m: m["energy_j_per_node_h"], lambda s: s["energy_j_per_node_h"]),
    ]


def read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0][1:]
    names = [h[:-5] for h in header if h.endswith("_mean")]
    res = {n: {"mean": {}, "std": {}} for n in names}
    for row in rows[1:]:
        key = row[0]
        for h, v in zip(header, row[1:]):
            n, kind = h.rsplit("_", 1)
            if v != "":
                res[n][kind][key] = float(v)
    return res


def plot(results, order, metric3, out):
    metrics = metrics_for(metric3)
    style()
    base = results["S0"]["mean"]
    base_vals = np.array([f(base) for _, _, _, f, _ in metrics])
    others = [n for n in order if n != "S0" and n in results]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    n_g, n_s = len(metrics), len(others)
    width = 0.8 / n_s
    for s, name in enumerate(others):
        m, sd = results[name]["mean"], results[name]["std"]
        vals = np.array([f(m) for _, _, _, f, _ in metrics]) / np.where(base_vals > 0, base_vals, np.nan)
        errs = np.array([g(sd) for _, _, _, _, g in metrics]) / np.where(base_vals > 0, base_vals, np.nan)
        xs = np.arange(n_g) - 0.4 + width * (s + 0.5)
        ax.bar(xs, vals, width=width * 0.92, color=COLORS.get(name, SERIES[s]), label=SCENARIO_LABELS[name], zorder=3)
        ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=NEUTRAL, elinewidth=0.7, capsize=1.5, zorder=4)
        for x, v, e, (key, _, _, f, _) in zip(xs, vals, errs, metrics):
            absval = f(m)
            txt = f"{absval:.0f}" if key in ("median_err_m", "energy_j_per_node_h") else f"{absval:.2f}"
            if np.isfinite(v):
                ax.text(x, v + (e if np.isfinite(e) else 0) + 0.02, txt, ha="center", va="bottom",
                        fontsize=6.5, color=NEUTRAL, rotation=90)
    ax.axhline(1.0, color=NEUTRAL, lw=1.0, ls="--", zorder=4)
    ax.text(-0.48, 1.03, "S0 = 1", ha="left", va="bottom", fontsize=7.5, color=NEUTRAL)
    ax.set_xticks(np.arange(n_g))
    ax.set_xticklabels([lab for _, lab, _, _, _ in metrics])
    ax.set_ylabel("Relative to S0 (lower is better)")
    ax.set_ylim(0, max(1.6, ax.get_ylim()[1]))
    ax.legend(ncol=3, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    base_txt = ",  ".join(
        f"{lab.replace(chr(10), ' ')}: {f(base):.0f}{unit}" if key in ("median_err_m", "energy_j_per_node_h")
        else f"{lab.replace(chr(10), ' ')}: {f(base):.2f}{unit}" for key, lab, unit, f, _ in metrics)
    fig.text(0.5, -0.04, f"S0 absolute values:  {base_txt}", ha="center", va="top", fontsize=8, color=NEUTRAL)
    save(fig, os.path.join(out, f"fig3_bars_{metric3}"))
    print("saved", os.path.join(out, f"fig3_bars_{metric3}.pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=100)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIO_ORDER)
    ap.add_argument("--metric3", choices=list(THIRD), default="airtime")
    ap.add_argument("--from-csv", default=None, help="replot from an existing fig3_table.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.from_csv:
        results = read_csv(args.from_csv)
        plot(results, args.scenarios, args.metric3, args.out)
        return

    base = SimConfig(duration_h=args.hours, n_mc=args.mc)
    results = {}
    for name in args.scenarios:
        t0 = time.time()
        res = run_mc(scenario(name, base), n_mc=args.mc, workers=args.workers)
        results[name] = res
        m = res["mean"]
        print(f"{name:4s}  med_err={m['median_err_m']:7.1f} m  P_loc={m['p_loc']:.3f}  "
              f"airtime_near={m['airtime_near_target_pct']:.2f}%  fail={m['delivery_failure_pct']:.1f}%  "
              f"energy={m['energy_j_per_node_h']:.0f} J/node/h  NPE={m['npe_rmse_m']:.1f} m   ({time.time() - t0:.0f} s)",
              flush=True)

    keys = sorted(set().union(*[set(r["mean"].keys()) for r in results.values()]))
    with open(os.path.join(args.out, "fig3_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + [f"{n}_mean" for n in results] + [f"{n}_std" for n in results])
        for k in keys:
            w.writerow([k] + [results[n]["mean"].get(k, "") for n in results] + [results[n]["std"].get(k, "") for n in results])
    if "S0" in results:
        for m3 in THIRD:
            plot(results, args.scenarios, m3, args.out)


if __name__ == "__main__":
    main()
