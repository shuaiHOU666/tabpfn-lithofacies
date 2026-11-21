# tabpfn-lithofacies

Code, digitised core-facies labels, and reproducible results for:

> Hou, S., Zhao, J., & Cao, D. *Can a tabular foundation model improve well-log lithofacies identification? Fine-tuning and validation on public data and application to Dinantian geothermal carbonates.* (Manuscript in preparation)

This repository systematically evaluates the tabular foundation model **TabPFN (v2)** for well-log lithofacies classification in three modes:

1. **Feature extractor:** Leakage-safe, well-grouped out-of-fold embeddings concatenated with original logs and fed to RF, SVM, XGBoost, and MLP, paired with SHAP interpretation and SHAP-RFE feature selection.
2. **Direct classifier:** Pretrained TabPFN used out of the box via in-context learning (tuning-free).
3. **Fine-tuned & transferred model:** TabPFN weights fine-tuned on public well logs and transferred to a new geothermal carbonate field.

The benchmark is organized in two parts:

- **Part I — Validation on public and synthetic data:** Hugoton–Panoma benchmark (SEG 2016 ML contest, 10 wells, 9 facies) covering blind-well tests, extraction-scheme ablation, leave-one-well-out (LOWO) validation, training-well-count experiments, fine-tuning, SHAP interpretability, and controlled rock-physics synthetic wells with tunable inter-well calibration shifts.
- **Part II — Field application (SCAN Dinantian geothermal carbonates, the Netherlands):** Automated digitisation of 14 scanned/vector core-log charts (~249 m of core across 9 wells) from the Dutch SCAN programme, core-to-log matching with NLOG wireline logs (3 wells with complete log suites: BHG-01, KTG-01, O18-01; 814 labelled samples), LOWO evaluation, and cross-field transfer of Hugoton-fine-tuned weights.

中文使用说明请见 [README_zh.md](README_zh.md)。

---

## Key findings

| Question | Main finding |
|---|---|
| Does the embedding extraction scheme matter? | **Yes.** Vanilla extraction leaks sample labels (100% training vs. 23.7% blind accuracy for SVM); random out-of-fold still leaks via depth autocorrelation (43.6% vs. 44.7% raw). Only **well-grouped out-of-fold** extraction avoids leakage (50.5%). |
| Do leakage-free embeddings improve real blind wells? | **Rarely.** On Hugoton–Panoma, LOWO macro-F1 dropped for 3 of 4 classifiers, and fell by 0.08–0.15 across all classifiers with 2 training wells. On SCAN (2 training wells/fold), embeddings lowered pooled accuracy by 18–33 percentage points. |
| Does fine-tuning on small public logs help or transfer? | **No.** Fine-tuning on 7 Hugoton wells slightly lowered blind-well accuracy (45.9% → 45.3% after 10 epochs) and did not improve transfer to SCAN (56.4% fine-tuned vs. 56.6% pretrained). |
| When can embeddings help, and what works with few cored wells? | Controlled synthetic tests show modest gains (up to +0.04 macro-F1) only with ≥4 training wells and small inter-well shifts. In few-well fields, use **original logs** or **pretrained TabPFN directly** as a tuning-free baseline (61.4% pooled accuracy on SCAN vs. 68.2% for the best model on raw logs). |

---

## Installation

```bash
conda create -n tabpfn-lith python=3.11
conda activate tabpfn-lith
pip install -r requirements.txt
```

- **TabPFN v2 weights:** The checkpoint (`tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`) is downloaded automatically from Hugging Face on first run. For offline machines, download it manually and point `TABPFN_MODEL_CACHE_DIR` to its directory. Model weights are not redistributed in this repository (subject to the Prior Labs licence).
- **OCR dependency (SCAN digitisation only):** `core_labels.py` requires [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system (pass `--tesseract <path>` if not in `PATH`).

---

## Quick start

```bash
# 1. Fast pipeline verification (no TabPFN weights required; uses surrogate embeddings)
python run_lithology.py --demo --quick

# 2. Part I blind-well benchmark (Hugoton–Panoma: SHANKLE & NEWBY)
python run_lithology.py --demo

# 3. Reproduce all experiments in the manuscript
bash run_all.sh cpu          # Linux / macOS (replace 'cpu' with 'cuda' if GPU is available)
run_all.bat                  # Windows
```

### Run on your own wells (CSV or LAS)

```bash
# Single blind-well evaluation (S1–S6 + figures)
python run_lithology.py \
    --data wellA.las wellB.las wellC.las \
    --label FACIES --well WELL --depth DEPT \
    --features GR RHOB NPHI DT PE \
    --blind wellC --out results_mine

# Leave-one-well-out evaluation
python experiments.py lowo \
    --data my_table.csv --label FACIES --well WELL \
    --features GR RHOB NPHI DT --out results_mine_lowo
```

---

## Repository structure

| File / Folder | Purpose |
|---|---|
| `lith_tabpfn.py` | Core library: log cleaning, scaling, TabPFN embedding extraction (vanilla / random OOF / well-grouped OOF), classifiers, SHAP, and SHAP-RFE |
| `run_lithology.py` | End-to-end S1–S6 pipeline and figure generation |
| `ablation.py` | Embedding extraction ablation study (Table 3) |
| `lowo.py`, `experiments.py` | Leave-one-well-out and training-well-count experiments with global or per-well normalisation (Table 4, Fig. 7) |
| `finetune_transfer.py` | Fine-tuning TabPFN on Hugoton–Panoma and cross-field transfer to SCAN (Table 5, Section 4.3) |
| `make_rp_synthetic.py`, `rockphysics.py` | DEM + Gassmann carbonate–clastic synthetic well generator with controllable well-to-well calibration shifts |
| `core_labels.py` | Automated digitisation of lithology and Dunham texture columns from scanned/vector SCAN core-log PDFs |
| `scan_dataset.py` | Core-to-log depth matching and dataset assembly from NLOG LAS files → `data/scan_facies.csv` |
| `scan_lowo.py` | Leave-one-well-out experiments on the SCAN Dinantian dataset (Table 8) |
| `run_all.sh`, `run_all.bat` | Master scripts to reproduce all tables and figures |
| `data/` | Processed tables, synthetic datasets, digitised SCAN core facies (`core_facies.csv`, `scan_facies.csv`), and visual QC plots (`core_qc/`) |

---

## Mapping results to the manuscript

| Manuscript item | Output file / folder |
|---|---|
| Table 2, Figs. 3–6, 8–9 | `results/` (`metrics.csv`, `shap_*.csv`, `rfe_history.csv`, `figures/`) |
| Table 3 (extraction ablation) | `results/ablation.csv`, `results/randomfold_oof/` |
| Table 4 (Hugoton–Panoma LOWO) | `results/lowo.csv`, `results_exp/hug_lowo_global/`, `results_exp/hug_lowo_well/` |
| Table 5 (TabPFN fine-tuning) | `results_ft/early_stop/part1_hugoton_blind.csv`, `results_ft/epochs10/part1_hugoton_blind.csv` |
| Fig. 7 (Δ macro-F1 vs. training wells) | `results_exp/hug_nwells_global/`, `results_exp/syn_bias_*/`, `results_exp/fig_delta_vs_wells.png` |
| Table 6, Fig. 10 (synthetic base case) | `results_rp/` |
| Table 7, Fig. 12 & visual QC (SCAN digitisation) | `data/core_facies.csv`, `data/scan_dataset_report.csv`, `data/core_qc/` |
| Table 8 (SCAN LOWO) | `results_exp/scan_lowo_global/`, `results_exp/scan_lowo_well/`, `results_scan/A4/`, `A5/`, `B2/` |
| Section 4.3 (cross-field transfer to SCAN) | `results_ft/early_stop/part2_scan_*.csv`, `results_ft/epochs10/part2_scan_*.csv` |

---

## Data & Acknowledgements

See [data/README.md](data/README.md) for full data provenance, column definitions, and terms of use.

We thank **EBN** and **TNO** for releasing the Dutch SCAN core logs and well logs via the [NLOG portal](https://www.nlog.nl/en/scan), and the **Kansas Geological Survey** and **SEG** for the Hugoton–Panoma benchmark dataset.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). If you use the code or digitised SCAN facies labels, please also cite the original data providers: Dubois et al. (2007), Hall (2016), and Mozafari et al. (2019) / EBN & TNO (SCAN programme via NLOG).

## Licence

- **Code:** MIT License (see [`LICENSE`](LICENSE)).
- **Data:** Subject to the original providers' terms (see [`data/README.md`](data/README.md)).