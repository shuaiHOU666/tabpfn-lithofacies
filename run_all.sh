#!/usr/bin/env bash
# Reproduce every number in the paper (CPU: roughly 6-10 h; GPU: under 1 h).
# Usage:  bash run_all.sh [cpu|cuda]
set -e
cd "$(dirname "$0")"
DEV=${1:-cpu}
export PYTHONWARNINGS=ignore

echo "== Part I: Hugoton-Panoma =="
python run_lithology.py --demo --device $DEV --out results                       # Table 2, Figs 2-5, 7-8
python run_lithology.py --demo --oof random --device $DEV --out results/randomfold_oof
python ablation.py --device $DEV                                                 # Table 3
python lowo.py --device $DEV                                                     # Table 4
H="--data data/facies_vectors.csv --label Facies --well Well\ Name --depth Depth --features GR ILD_log10 DeltaPHI PHIND PE NM_M RELPOS --no-norm NM_M RELPOS --device $DEV"
eval python experiments.py lowo   $H --norm global --out results_exp/hug_lowo_global
eval python experiments.py lowo   $H --norm well   --out results_exp/hug_lowo_well
eval python experiments.py nwells $H --norm global --k 2 4 6 8 --repeats 3 --out results_exp/hug_nwells_global   # Fig. 6a
python finetune_transfer.py --part 1 --device $DEV --out results_ft/early_stop                         # Table 5
python finetune_transfer.py --part 1 --device $DEV --epochs 10 --no-early-stop --out results_ft/epochs10

echo "== Controlled synthetic tests =="
python make_rp_synthetic.py --out data/rp_synthetic_facies.csv                  # 8-well base case (seed 42)
python run_lithology.py --data data/rp_synthetic_facies.csv --label FACIES --well WELL --depth DEPTH \
       --features GR AI VPVS RHOB --blind W07 W08 --device $DEV --out results_rp  # Table 6, Fig. 9
for b in 0 0.03 0.06 0.10; do
  python make_rp_synthetic.py --n-wells 10 --bias $b --seed 7 --out data/synthetic/syn_bias_$b.csv
  python experiments.py nwells --data data/synthetic/syn_bias_$b.csv --features GR AI VPVS RHOB \
         --k 2 4 6 --repeats 2 --device $DEV --out results_exp/syn_bias_$b                     # Fig. 6b
done

echo "== Part II: SCAN Dinantian carbonates =="
# data/scan_facies.csv is included. To rebuild it from NLOG downloads, see data/README.md.
for n in global well; do
  python experiments.py lowo --data data/scan_facies.csv --wells BHG-01 KTG-01 O18-01 \
         --features LOG_RDEP RHOB NPHI DT --norm $n --device $DEV --out results_exp/scan_lowo_$n   # Table 7
done
python scan_lowo.py --data data/scan_facies.csv --wells BHG-01 KTG-01 O18-01 --features GR LOG_RDEP RHOB NPHI DT --out results_scan/A5
python scan_lowo.py --data data/scan_facies.csv --wells BHG-01 KTG-01 O18-01 --features LOG_RDEP RHOB NPHI DT --out results_scan/A4
python scan_lowo.py --data data/scan_facies.csv --wells BHG-01 KTG-01 O18-01 KSL-02 --features GR DT --out results_scan/B2
python finetune_transfer.py --part 2 --device $DEV --out results_ft/early_stop       # transfer of Hugoton weights
python finetune_transfer.py --part 2 --device $DEV --epochs 10 --no-early-stop --out results_ft/epochs10
echo "All done."
