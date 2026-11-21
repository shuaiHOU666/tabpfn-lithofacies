# tabpfn-lithofacies

论文配套代码、数字化岩心岩相标签与全部可复现实验结果：

> Hou, S., Zhao, J., & Cao, D. *Can a tabular foundation model improve well-log lithofacies identification? Fine-tuning and validation on public data and application to Dinantian geothermal carbonates.*（稿件整理中）

本仓库系统评估了表格基础模型 **TabPFN (v2)** 在测井岩相识别任务中的三种使用方式：

1. **特征提取器**：通过防标签泄漏的按井分组折外（well-grouped out-of-fold）方式提取 TabPFN 嵌入特征，与原始测井曲线拼接后输入 RF、SVM、XGBoost、MLP 分类器，并结合 SHAP 可解释性分析与 SHAP-RFE 特征选择；
2. **直接分类器**：直接使用预训练 TabPFN 进行上下文学习（in-context learning，无需调参）；
3. **微调与迁移**：在公开测井数据集上对 TabPFN 进行梯度微调，并将权重迁移至新的地热碳酸盐岩工区。

全部实验分为两部分：

- **第一部分（公开数据与可控合成实验验证）**：基于 Hugoton–Panoma 公开数据集（SEG 2016 竞赛，10 口井，9 类岩相），开展盲井验证、嵌入提取方式消融、留一井验证（LOWO）、训练井数敏感性分析、模型微调与 SHAP 解释；并结合 DEM + Gassmann 岩石物理正演构建井间标定偏差可控的碳酸盐岩–碎屑岩合成井测试。
- **第二部分（实际工区应用：荷兰 SCAN Dinantian 地热碳酸盐岩）**：构建开源岩心柱状图自动数字化流程，从 14 张扫描/矢量岩心柱状图中提取了 9 口井约 249 m 的岩相标签，与 NLOG 测井曲线完成深度归位与匹配（BHG-01、KTG-01、O18-01 三口全曲线井共 814 个有效标签样本），执行留一井测试及 Hugoton 微调权重的跨工区迁移测试。

English documentation: [README.md](README.md).

---

## 核心结论

| 研究问题 | 实验结论 |
|---|---|
| 训练集嵌入提取方式是否影响盲井表现？ | **决定性影响。** 常规提取（样本自身标签在上下文中）会导致严重标签泄漏（SVM 训练集 100%，盲井仅 23.7%）；随机折外提取仍会因相邻深度自相关产生同井泄漏（43.6% < 原始曲线 44.7%）。只有**按井分组的折外提取**能避免泄漏（50.5%）。 |
| 无泄漏嵌入在真实测井数据上是否稳定有效？ | **极少有效，少井时反而有害。** Hugoton 留一井验证中 4 个分类器有 3 个的 macro-F1 下降；仅用 2 口训练井时，所有分类器 macro-F1 下降 0.08–0.15。在 SCAN 工区（每折 2 口训练井），引入嵌入使合并准确率下降 18–33 个百分点。 |
| 在小型公开测井集上微调 TabPFN 是否有增益？ | **没有。** 在 7 口 Hugoton 井上微调 10 个 epoch 后盲井准确率略降（45.9% → 45.3%），迁移至 SCAN 后亦无收益（56.4% vs. 预训练 56.6%）。 |
| 少取心井工区的稳妥选择是什么？ | 合成实验表明，仅在训练井 ≥4 口且井间偏移很小时嵌入才有小幅增益（macro-F1 最高 +0.04）。少井条件下建议直接使用**原始测井特征**，或将**预训练 TabPFN 直接作为免调参基线**（SCAN 合并准确率 61.4%，接近最优基线 68.2%）。 |

---

## 环境安装

```bash
conda create -n tabpfn-lith python=3.11
conda activate tabpfn-lith
pip install -r requirements.txt
```

- **TabPFN v2 权重下载**：首次运行时会自动从 Hugging Face 下载 `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`。如在国内访问受限，可先配置镜像再运行：
  ```bash
  # Linux / macOS
  export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)

  # Windows PowerShell
  $env:HF_ENDPOINT="[https://hf-mirror.com](https://hf-mirror.com)"
  ```
  离线环境可手动下载权重文件，并通过环境变量 `TABPFN_MODEL_CACHE_DIR` 指定存放目录（本仓库不二次分发模型权重，需遵守 Prior Labs 许可协议）。
- **Tesseract OCR（仅 SCAN 岩心图数字化需要）**：运行 `core_labels.py` 需安装 [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)。Windows 若未写入 `PATH`，可通过 `--tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"` 指定路径。
- **硬件与精度说明**：脚本默认自动检测 GPU（支持 `--device cuda` 或 `cpu`）；CPU 模式下固定使用 `float32` 以避免 `bfloat16` 导出嵌入报错。TabPFN v2 适用于约 1 万样本以内的表格数据。

---

## 快速开始

```bash
# 0) 流程冒烟测试（不下载 TabPFN 权重，用替代嵌入检查代码能否跑通，结果不可用于论文）
python run_lithology.py --demo --quick

# 1) 第一部分盲井实验：Hugoton–Panoma（盲井 SHANKLE、NEWBY）
python run_lithology.py --demo

# 2) 一键复现论文全部实验
bash run_all.sh cpu          # Linux / macOS（有显卡可改为 cuda）
run_all.bat                  # Windows
```

### 在自有井数据上运行（支持 CSV 或多个 LAS）

```bash
# 单次盲井测试（完整执行 S1–S6 并输出图件）
python run_lithology.py \
    --data wellA.las wellB.las wellC.las \
    --label FACIES --well WELL --depth DEPT \
    --features GR RHOB NPHI DT PE \
    --blind wellC --out results_mine

# 留一井交叉验证（LOWO）
python experiments.py lowo \
    --data my_table.csv --label FACIES --well WELL \
    --features GR RHOB NPHI DT --out results_mine_lowo
```

常用参数：`--oof well|random`（默认按井分折 `well`）、`--n-fold 5`、`--no-rfe`（跳过递归特征消除以加速）。

### SCAN（荷兰 Dinantian 碳酸盐岩）端到端流程

```bash
# 1. 从 SCAN 岩心柱状图 PDF 自动数字化提取岩相标签及质检图（输出至 data/core_facies.csv 与 data/core_qc/）
python core_labels.py --pdf-dir "<SCAN_Appendix_B_PDF目录>" --out data/core_facies.csv

# 2. 将数字化岩相与 NLOG 测井 LAS 按井名匹配并执行岩心-测井深度校正（输出 data/scan_facies.csv）
python scan_dataset.py --las-dir data --cores data/core_facies.csv --out data/scan_facies.csv

# 3. 执行 SCAN 留一井测试（去掉受放射性灰岩干扰的 GR，使用 4 条核心曲线）
python scan_lowo.py \
    --data data/scan_facies.csv \
    --wells BHG-01 KTG-01 O18-01 \
    --features LOG_RDEP RHOB NPHI DT \
    --out results_scan/A4
```

---

## 代码文件说明

| 文件 / 目录 | 功能 |
|---|---|
| `lith_tabpfn.py` | 核心函数库：数据清洗与标准化、TabPFN 嵌入提取（vanilla / 随机折外 / 按井折外）、分类器训练、SHAP 与 SHAP-RFE |
| `run_lithology.py` | 端到端 S1–S6 流程与出图脚本 |
| `ablation.py` | 嵌入提取方式消融实验（对应 Table 3） |
| `lowo.py`, `experiments.py` | 留一井验证与训练井数敏感性实验，支持全局/单井标准化（对应 Table 4、Fig. 7） |
| `finetune_transfer.py` | Hugoton–Panoma 微调与 SCAN 跨工区迁移实验（对应 Table 5、Section 4.3） |
| `make_rp_synthetic.py`, `rockphysics.py` | 基于 DEM + Gassmann 模型的碳酸盐岩–碎屑岩合成井生成（支持可控井间标定偏差） |
| `core_labels.py` | 扫描/矢量岩心柱状图岩性柱与 Dunham 结构轮廓线自动数字化 |
| `scan_dataset.py` | 岩心岩相与 NLOG LAS 曲线深度归位与样本表构建 |
| `scan_lowo.py` | SCAN 数据集留一井验证脚本（对应 Table 8） |
| `run_all.sh`, `run_all.bat` | 一键复现全部实验结果的批处理脚本 |
| `data/` | 整理后的数据表、合成数据、数字化 SCAN 岩相区间（`core_facies.csv`、`scan_facies.csv`）及逐井质检图（`core_qc/`） |

---

## 实验结果与论文图表对应关系

| 论文图表 | 仓库对应目录 / 文件 |
|---|---|
| Table 2、Figs. 3–6、Figs. 8–9 | `results/`（`metrics.csv`、`shap_*.csv`、`rfe_history.csv`、`figures/`） |
| Table 3（提取方式消融） | `results/ablation.csv`、`results/randomfold_oof/` |
| Table 4（Hugoton 留一井验证） | `results/lowo.csv`、`results_exp/hug_lowo_global/`、`results_exp/hug_lowo_well/` |
| Table 5（TabPFN 微调对比） | `results_ft/early_stop/part1_hugoton_blind.csv`、`results_ft/epochs10/part1_hugoton_blind.csv` |
| Fig. 7（训练井数与井间偏差影响） | `results_exp/hug_nwells_global/`、`results_exp/syn_bias_*/`、`results_exp/fig_delta_vs_wells.png` |
| Table 6、Fig. 10（合成井基准算例） | `results_rp/` |
| Table 7、Fig. 12 与质检图（SCAN 数字化） | `data/core_facies.csv`、`data/scan_dataset_report.csv`、`data/core_qc/` |
| Table 8（SCAN 留一井验证） | `results_exp/scan_lowo_global/`、`results_exp/scan_lowo_well/`、`results_scan/A4/`、`A5/`、`B2/` |
| Section 4.3（跨工区迁移测试） | `results_ft/early_stop/part2_scan_*.csv`、`results_ft/epochs10/part2_scan_*.csv` |

---

## 数据来源、引用与开源协议

- **数据说明**：字段定义与原始数据使用条款见 [data/README.md](data/README.md)。感谢 **EBN** 与 **TNO** 通过 [NLOG 平台](https://www.nlog.nl/en/scan)公开荷兰 SCAN 岩心与测井数据，以及 **Kansas Geological Survey** 与 **SEG** 提供 Hugoton–Panoma 公开数据集。
- **引用方式**：详见 [`CITATION.cff`](CITATION.cff)。若使用本仓库代码或数字化 SCAN 岩相标签，请同时引用原始数据来源（Dubois et al., 2007; Hall, 2016; Mozafari et al., 2019 / EBN & TNO SCAN programme）。
- **开源协议**：代码采用 MIT License（见 [`LICENSE`](LICENSE)）；数据遵循原发布方条款。