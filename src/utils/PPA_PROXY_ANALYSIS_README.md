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
- 为避免方向混淆，分析中会将 `WNS/TNS` 转换为 `-WNS/-TNS` 参与相关性、热力图和改进率统计（统一为“越小越好”）。

## 7. 注意事项

- 目前仓库中 `ppa_eval_seed_1_gp_best.csv` 仅包含 OpenROAD 数据；若补齐 ICCAD 的 PPA CSV，可直接复用脚本。
- 若数据列名变化，请优先保持 `benchmark,case,formulation,best_hv` 与指标列一致。
- 若 `final_solutions.pkl` 缺失，脚本会回退尝试 `elite_pool.pkl`；若都不存在，对应 run 的 proxy 对齐会标记为失败。
- 指标策略：
  - `--benchmarks OpenROAD` 时优先分析 `GRT_WL/DRT_WL/WNS/TNS/Power/Area/DRC`。
  - `--benchmarks ICCAD2015` 时优先分析 `WNS/TNS`。
  - `--benchmarks all` 时取两类 benchmark 相关指标的并集并按可用性自动处理缺失。
  - 方向统一后，图表中会显示 `-WNS/-TNS` 标签，且顺序固定为 `GRT_WL -> DRT_WL -> -WNS -> -TNS -> Power -> Area -> DRC`。
  - `Area` 对应 OpenROAD 的 `StdCellArea`（脚本会自动将 `StdCellArea` 归一为 `Area`）。

## 9. 总体 + 分 design 输出（1 + 6 批）

脚本会在原 `--output_dir` 下先输出 1 批总体分析结果，然后针对 OpenROAD 每个 design（case）自动额外输出一批：

```text
<output_dir>/
  reports/ figures/ tables/                 # 总体
  by_design/
    ariane133/reports|figures|tables/
    ariane136/reports|figures|tables/
    bp/reports|figures|tables/
    bp_be/reports|figures|tables/
    bp_fe/reports|figures|tables/
    swerv_wrapper/reports|figures|tables/
```

这对应 `1 + 6` 批分析产物，可直接用于“总体结论 + 分 design 结论”。

## 8. 多种子 PPA 评估预算控制（速度 vs 严谨）

当你有 3 个 seed 时，若每个 seed 保留 top-5 DEF，则每个 `(benchmark, case, formulation)` 最多会评估 `3 x 5 = 15` 个 DEF。

`src/utils/evaluate_best_gp_ppa.py` 已支持两层控制：

- `--def_per_seed`：每个 seed 的候选 DEF 数（默认 5）。
- `--max_total_defs_per_setting`：跨 seed 总预算（默认 0，表示不截断）。
- `--def_select_strategy`：预算截断策略。
  - `seed_round_robin`（默认）：按 seed 轮转取 rank，兼顾 seed 覆盖与前沿质量。
  - `top_rank`：只按 rank 优先，偏向最优点，速度快但多样性较弱。
  - `rank_spread`：在候选序列中均匀抽样，保留一定形状覆盖。

示例 1：3-seed 全量（每设置最多 15 个 DEF）

```bash
python src/utils/evaluate_best_gp_ppa.py \
  --workspace . \
  --seeds 1,2,3 \
  --hv_json results/analysis_reports/hv/json/hv_summary_seeds_1_2_3.json \
  --output results/analysis_reports/ppa_eval \
  --def_per_seed 5 \
  --max_total_defs_per_setting 0 \
  --def_select_strategy seed_round_robin
```

示例 2：3-seed 但每设置总共只测 5 个 DEF（推荐快速迭代）

```bash
python src/utils/evaluate_best_gp_ppa.py \
  --workspace . \
  --seeds 1,2,3 \
  --hv_json results/analysis_reports/hv/json/hv_summary_seeds_1_2_3.json \
  --output results/analysis_reports/ppa_eval \
  --def_per_seed 5 \
  --max_total_defs_per_setting 5 \
  --def_select_strategy seed_round_robin
```

示例 3：更强调 rank-1/2 最优值（偏“极值验证”）

```bash
python src/utils/evaluate_best_gp_ppa.py \
  --workspace . \
  --seeds 1,2,3 \
  --hv_json results/analysis_reports/hv/json/hv_summary_seeds_1_2_3.json \
  --output results/analysis_reports/ppa_eval \
  --def_per_seed 5 \
  --max_total_defs_per_setting 5 \
  --def_select_strategy top_rank
```

论文口径建议：

- 主实验：`max_total_defs_per_setting=15`（或 12）保证统计充分。
- 消融/快速循环：`max_total_defs_per_setting=5` 并报告策略。
- 在文中明确写出“DEF 预算策略”和“每设置有效样本数 n”。