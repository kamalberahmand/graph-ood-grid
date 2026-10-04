#!/usr/bin/env bash
# Full reproduction of every table and figure in the paper.
# Order matters: later scripts read the data and results written by earlier ones.
# Approximate run time on a two-core CPU: IEEE systems ~1-2 h, PEGASE-1354 ~1 day
# (dominated by data generation and the full N-1 screening).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SYSTEMS="${*:-case39 case118 case1354pegase}"
mkdir -p data results logs figs tables

for s in $SYSTEMS; do
  python gridgen.py "$s"                           > "logs/gen_$s.log" 2>&1   # datasets -> data/
  python run2.py "$s"                              > "logs/run2_$s.log" 2>&1  # detection, severity, attribution
  python dual_role.py "$s"                         > "logs/dual_$s.log" 2>&1  # 5-seed AUROC of S_OOD, S_sev, D_p
  python sensitivity.py "$s"                       > "logs/sens_$s.log" 2>&1  # weights, CVI composition, ratings
  python misspec.py "$s"                           > "logs/misspec_$s.log" 2>&1  # attribution under model error
  python gnnsafe.py "$s"                           > "logs/gnnsafe_$s.log" 2>&1
  python gnnsafe_rho.py "$s"                       > "logs/gnnsafe_rho_$s.log" 2>&1
  if [ "$s" = "case1354pegase" ]; then N=5; else N=25; fi
  python n1_full.py "$s" "$N"                      > "logs/n1_$s.log" 2>&1    # full N-1 screening
  python timing.py "$s"                            > "logs/timing_$s.log" 2>&1
done

python triage.py      > logs/triage.log 2>&1   # screening prioritization (Table VI)
python make_tables.py                          # tables/*.tex
python fig1_build.py                           # Fig. 1 (framework example on IEEE-118)
python fig_tradeoff.py; python fig_sens.py; python fig_n1.py
echo "done: tables/ and figs/ regenerated"
