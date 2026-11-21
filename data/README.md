# Data

| File | Content | Source / terms |
|---|---|---|
| `facies_vectors.csv` | Hugoton–Panoma well logs and core facies, 10 wells (incl. Recruit F9 pseudo-well) | SEG 2016 machine-learning contest (Hall, 2016; https://github.com/seg/2016-ml-contest), originally Kansas Geological Survey (Dubois et al., 2007). Redistributed for reproducibility; please follow the contest repository's terms. |
| `core_facies.csv` | **New.** Facies intervals digitised from 14 SCAN core-log charts (WELL, top, base, FACIES; core depth, m) | Derived by `core_labels.py` from *SCAN Dinantian: Facies analysis and diagenetic evolution – Appendix B* (NLOG). |
| `core_facies_log.csv` | Per-chart digitisation log (depth fit, number of intervals) | `core_labels.py` |
| `core_qc/*.png` | Quality-control images: original chart (left) vs extracted lithology / texture (right) | `core_labels.py` |
| `scan_facies.csv` | **New.** Labelled log samples: WELL, DEPTH (log depth, m), CORE_DEPTH, FACIES, GR, LOG_RDEP (log10 Ω·m), RHOB (g/cm³), NPHI (v/v), DT (µs/m), PE | `scan_dataset.py` from NLOG LAS files + `core_facies.csv` |
| `scan_dataset_report.csv` | Core-to-log depth shift per cored run and its source | `scan_dataset.py` |
| `core_shift.csv` | Manual depth-shift overrides | user-defined |
| `rp_synthetic_facies.csv` | Synthetic 8-well base case (GR, AI, Vp/Vs, RHOB; truth PHIT, PHI_CRACK) | `make_rp_synthetic.py` (seed 42) |
| `synthetic/syn_bias_*.csv` | 10 synthetic wells with well-to-well calibration shift b = 0, 3, 6, 10% | `make_rp_synthetic.py --n-wells 10 --bias b --seed 7` |

Facies classes in the SCAN data: `Lst_mud` (mudstone / wackestone), `Lst_grain` (packstone / grainstone / boundstone), `Lst_breccia`, `Dolomite`, `Shale`, `Sandstone`.

## Rebuilding the SCAN table from the original downloads

The raw SCAN files are public on NLOG (https://www.nlog.nl/en/scan) but are not redistributed here. To rebuild:

1. Download *SCAN Dinantian Facies analysis and diagenetic evolution – Appendix B* (core logs, PDF) and unzip it into `data/`.
2. Download the LAS files (logs and `*Core_data*.las`) for the cored wells (BHG-01, KTG-01, O18-01, KSL-02, …) into `data/las/<WELL>/`.
3. Run
   ```bash
   python core_labels.py --pdf-dir "data/SCAN Dinantian Facies analysis and diagenetic evolution - Appendix B" --out data/core_facies.csv --qc-dir data/core_qc
   python scan_dataset.py --las-dir data --cores data/core_facies.csv --out data/scan_facies.csv
   ```

Automatic labels should be checked against `core_qc/` before use; depth matching for KTG-01 and O18-01 was taken from released core data rather than a full manual core-log correlation.
