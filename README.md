# ucoden_sim

Simulator for task-level cooperation in an underwater collaborative detection network (UCODEN), written for the article

> J. Xu and M.-S. Alouini, "How Underwater Collaborative Detection Networks Work Together: Task-Level Cooperation over a Shared Acoustic Channel."

It reproduces Figs. 3-5 of the article: a two-tier moored network of 242 nodes (500 m and 1500 m deep, one node per kilometer) in a 10 km x 10 km area of 2000-m-deep water tracks an acoustically tagged target, and three cooperation loops between its tasks can be switched on and off.

## Requirements

Python 3.10 or later with numpy, scipy and matplotlib:

```bash
pip install -r requirements.txt
```

## Reproducing the figures

```bash
python run_fig3.py --mc 60 --hours 24 --metric3 failures   # Fig. 3: S0, S0', S1-S4 relative to S0
python run_fig4.py --mc 20 --hours 12                      # Fig. 4: P_CD, P_ava, P_loc versus node spacing, three tilings
python run_fig5.py --mc 20 --hours 12                      # Fig. 5: node position error versus transmit energy
```

or `bash run_all.sh` for all three. Each script writes a CSV with the mean and standard deviation of every metric, and the figure as PDF and PNG, to `results/`.
The CSVs behind the published figures are included, so the figures can be redrawn without simulating:

```bash
python run_fig3.py --from-csv results/fig3_table.csv --metric3 failures
python run_fig4.py --from-csv results/fig4_table.csv
python run_fig5.py --from-csv results/fig5_table.csv
```

The first run traces rays through the sound-speed profile and caches the lookup table in `ucoden_sim/_cache/` (a few seconds).
A 24-hour run takes 10-20 s on one core; Monte-Carlo runs are spread over `--workers` processes.

## Scenarios

| Scenario | Loop 1: communication and localization (`c1_comm_loc`) | Loop 2: detection, localization and tracking (`c2_det_loc_track`) | Loop 3: environment sensing (`c3_env`) | Node calibration (`calibrate_nodes`) |
|---|---|---|---|---|
| S0 | off | off | off | dedicated pings every 30 min |
| S0' | off | off | off | none (design positions) |
| S1 | on | off | off | on |
| S2 | off | on | off | on |
| S3 | off | off | on | on |
| S4 | on | on | on | on |

The presets are in `ucoden_sim/scenarios.py`; every parameter and its default value is in `ucoden_sim/config.py`.

| Switch | Off (independent design) | On (cooperation) |
|---|---|---|
| `c1_comm_loc` | Node positions are calibrated from dedicated ranging pings every `ping_period_min`; fixed routing. | Every decoded packet gives a range to its sender (opportunistic ranging); estimated node positions and the ray table identify shadowed links, which are relayed. |
| `c2_det_loc_track` | All nodes listen with duty cycle `duty_search` and threshold `pfa_search`; any single report triggers a fix, whatever the geometry. | `k_of_n` collaborative decision; a fix only if GDOP is at most `gdop_max`; nodes inside the predicted gate listen continuously with a relaxed threshold, nodes outside listen less; the `k_poll` best-geometry nodes are polled. |
| `c3_env` | Ranges are converted with 1500 m/s and a straight path. | CTD samples ride on status packets; ranges use the local sound speed and the ray-traced curve-to-line factor, and direct and surface-reflected arrivals are told apart. |

## Code layout

| File | Content |
|---|---|
| `config.py` | All parameters and defaults (`SimConfig`). |
| `scenarios.py` | Scenario presets S0, S0' (`S0p`), S1-S4. |
| `environment.py` | Medwin sound speed; synthetic winter north-west Pacific profile modeled on Argo observations; ray-tracing lookup table (curve-to-line factor, path-mean sound speed, direct or surface-reflected path); Thorp absorption, Wenz-type noise, energy detector. |
| `network.py` | Square, triangular and hexagonal layouts, two tiers, GNSS chains, mooring drift (rotary tide plus bounded spatially correlated random walk), energy and channel bookkeeping. |
| `target.py` | Tagged target: random waypoints at 1-3 m/s, one pulse per minute at known emission times. |
| `detection.py` | Passive sonar equation and energy detector; k-out-of-n or single-report triggering. |
| `localization.py` | Time of arrival to range, weighted least-squares fix, GDOP, node selection, recursive node-position calibration. |
| `tracking.py` | Constant-velocity Kalman filter and the duty-cycle and threshold policy. |
| `comm.py` | Packet airtime and energy, link losses (5 % direct, 70 % shadowed), retransmissions, shadow-aware relaying, opportunistic ranging, status, CTD and ping traffic. |
| `simulator.py` | Main loop over one-minute detection epochs, metrics, parallel Monte-Carlo runs. |

## Metrics

`run_once` returns, among others: `median_err_m` and `rmse_m` (target localization error), `p_loc` (fraction of epochs localized within 100 m), `p_cd` (collaborative detection), `p_ava` (at least four ranges with acceptable geometry), `delivery_failure_pct` (reports that fail to reach the node computing the fix), `airtime_near_target_pct` (channel occupancy within 2 km of the target), `energy_j_per_node_h` and `energy_tx_j_h` (energy per node-hour, total and transmit), and `npe_rmse_m` (node position error).

## Related work

The scenario follows J. Xu, M. A. Kishk, X. Irigoien and M.-S. Alouini, "UCODENs: Underwater collaborative detection networks," *IEEE Open Journal of the Communications Society*, vol. 6, pp. 77-89, 2025.

## License

No license has been chosen yet; please contact the authors before reusing the code.
