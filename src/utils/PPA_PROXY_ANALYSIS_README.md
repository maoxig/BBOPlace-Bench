# PPA + Proxy Joint Analysis

本文档说明如何基于已有的 PPA 评估结果（如 `results/analysis_reports/ppa_eval/seed_1/ppa_eval_seed_1_gp_best.csv`）与 HV 汇总（如 `results/analysis_reports/hv/json/hv_summary_seed_1.json`）生成论文可用的图表与统计分析。

## 1. 分析目标

- 对每个 `(benchmark, case, formulation)` 的 PPA 表现做统计（均值、标准差、best-of-5、bootstrap CI）。
- 分析 proxy 指标（HV）与真实 PPA 指标的对齐程度（Spearman/Pearson 相关）。
- 基于 `final_solutions.pkl` 中保存的 `Y`，建立 `def_rank -> Y` 映射，做代理目标与真实PPA的直接相关分析。
- 对比 HPO 与 MGO 在各 PPA 指标上的提升率与胜率。
- 汇总 GP/MP 两种 mode 下 best-run 的代理目标集合与维度分布。
- 自动使用可读代理名（`mode:objective`，例如 `GP:gp_hpwl`、`MP:hpwl`），不再使用不可读的 `proxy_obj_1` 风格作为分析展示名。
- 输出学术友好的 PDF 图和 LaTeX 表。

## 2. 脚本位置

- `src/utils/analyze_ppa_proxy.py`

## 3. 运行示例

### 3.1 OpenROAD GP 结果（你当前的主数据）

```bash
python src/utils/analyze_ppa_proxy.py \
  --ppa_csv results/analysis_reports/ppa_eval/seed_1/ppa_eval_seed_1_gp_best.csv \
  --hv_json results/analysis_reports/hv/json/hv_summary_seed_1.json \
  --output_dir results/analysis_reports/ppa_proxy/study_seed_1 \
  --benchmarks OpenROAD \
  --formulations MGO,HPO \
  --only_eval_ok
```

### 3.2 指定部分 case 做快速检查

```bash
python src/utils/analyze_ppa_proxy.py \
  --ppa_csv results/analysis_reports/ppa_eval/seed_1/ppa_eval_seed_1_gp_best.csv \
  --hv_json results/analysis_reports/hv/json/hv_summary_seed_1.json \
  --output_dir results/analysis_reports/ppa_proxy/study_quick \
  --benchmarks OpenROAD \
  --cases ariane133,bp,bp_be \
  --formulations MGO,HPO \
  --only_eval_ok
```

### 3.3 按 benchmark 类型分析

OpenROAD：

```bash
python src/utils/analyze_ppa_proxy.py \
  --ppa_csv results/analysis_reports/ppa_eval/seed_1/ppa_eval_seed_1_gp_best.csv \
  --hv_json results/analysis_reports/hv/json/hv_summary_seed_1.json \
  --output_dir results/analysis_reports/ppa_proxy/openroad \
  --benchmarks OpenROAD \
  --only_eval_ok
```

ICCAD2015：

```bash
python src/utils/analyze_ppa_proxy.py \
  --ppa_csv <your_iccad_ppa_csv> \
  --hv_json results/analysis_reports/hv/json/hv_summary_seed_1.json \
  --output_dir results/analysis_reports/ppa_proxy/iccad \
  --benchmarks ICCAD2015 \
  --only_eval_ok
```

整体（OpenROAD + ICCAD2015）：

```bash
python src/utils/analyze_ppa_proxy.py \
  --ppa_csv <merged_ppa_csv> \
  --hv_json results/analysis_reports/hv/json/hv_summary_seed_1.json \
  --output_dir results/analysis_reports/ppa_proxy/all \
  --benchmarks all \
  --only_eval_ok
```

## 4. 输出结构

输出目录默认结构：

```text
results/analysis_reports/ppa_proxy/study_seed_1/
  reports/
    analysis_report.md
  figures/
    box_<metric>_by_formulation.pdf/.png
    scatter_hv_vs_<metric>.pdf/.png
    profile_<metric>.pdf/.png
    heatmap_best_<metric>.pdf/.png
    heatmap_proxy_ppa_spearman.pdf/.png
    heatmap_rank_consistency_median.pdf/.png
    bar_top1_hit_rate.pdf/.png
    bar_proxy_dim_by_mode.pdf/.png
  tables/
    summary_by_case_formulation.csv/.tex
    hv_metric_correlation.csv/.tex
    hpo_vs_mgo_improvement.csv/.tex
    proxy_metric_correlation.csv/.tex
    proxy_groupwise_rank_consistency.csv/.tex
    proxy_top1_hit_rate.csv/.tex
    proxy_top1_hit_detail.csv
    proxy_object_catalog_all_modes.csv/.tex
    proxy_mode_object_sets.csv/.tex
    ppa_with_proxy_alignment.csv
    proxy_long_alignment.csv

推荐统一目录（轻量分类）：

```text
results/analysis_reports/
  hv/
    markdown/
    latex/
    json/
  ppa_eval/
    seed_<seed>/
  ppa_proxy/
    <study_name>/
      reports/
      tables/
      figures/
```
```

## 5. 建议论文呈现方式

- 主文图：
  - `scatter_hv_vs_DRT_WL.pdf`、`scatter_hv_vs_WNS.pdf`、`profile_DRT_WL.pdf`。
- 主文表：
  - `hpo_vs_mgo_improvement.tex`（胜率与提升率）
  - `hv_metric_correlation.tex`（proxy 与真实指标关联）
- 附录：
  - `summary_by_case_formulation.tex` + 全部箱线图/热力图。

## 6. 方法学说明（可直接改写到论文）

- 每个组合使用 `best-hv algorithm` 的 top-5 DEF（优先 `gp_*.def`）做 PPA 评估。
- 对 PPA 指标执行 `best-of-5` 与 `mean-of-5` 两种统计视角。
- 对 HPO vs MGO 使用配对 bootstrap（case 维度）估计均值差置信区间。
- 以 Spearman 相关检验 `HV` 与真实 PPA 指标的一致性，减少仅看均值的偏差风险。
- 从 `checkpoint/final_solutions.pkl` 读取 Y 向量，用 `def_rank` 映射到同一行 PPA 结果，构建 `proxy_obj_i` 与真实PPA的一一对应样本。
- 从 `checkpoint/final_solutions.pkl` 读取 Y 向量，用 `def_rank` 映射到同一行 PPA 结果，并将代理目标按 `mode:objective` 命名。
- 计算组内（同一 run 的 top-5 DEF）排序一致性、Top-1 命中率（`argmin(proxy_obj_i)` 是否命中 `best PPA`）。
- 同时输出 GP/MP 两种 mode 的 best-run 代理目标目录，便于讨论优化目标设计差异。
- 脚本默认不分析 runtime/duration，它们已从指标集合中移除。

## 7. 注意事项

- 目前仓库中 `ppa_eval_seed_1_gp_best.csv` 仅包含 OpenROAD 数据；若补齐 ICCAD 的 PPA CSV，可直接复用脚本。
- 若数据列名变化，请优先保持 `benchmark,case,formulation,best_hv` 与指标列一致。
- 若 `final_solutions.pkl` 缺失，脚本会回退尝试 `elite_pool.pkl`；若都不存在，对应 run 的 proxy 对齐会标记为失败。
- 指标策略：
  - `--benchmarks OpenROAD` 时优先分析 `GRT_WL/DRT_WL/WNS/TNS/Power`。
  - `--benchmarks ICCAD2015` 时优先分析 `WNS/TNS`。
  - `--benchmarks all` 时取两类 benchmark 相关指标的并集并按可用性自动处理缺失。