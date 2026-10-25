#!/bin/bash
# Reproduce Figs. 3-5 with the settings used in the article. Logs go to results/*.log.
set -e
cd "$(dirname "$0")"
mkdir -p results
python3 run_fig3.py --mc 60 --hours 24 --metric3 failures --out results > results/fig3.log 2>&1
python3 run_fig4.py --mc 20 --hours 12 --out results > results/fig4.log 2>&1
python3 run_fig5.py --mc 20 --hours 12 --out results > results/fig5.log 2>&1
echo "done: figures and CSV tables are in results/"
