# BBOPlace-Bench 评估脚本说明文档

本文档介绍了用于评估 BBOPlace-Bench 结果的两个主要脚本：
1.  **ICCAD2015 评估脚本** (`src/utils/timing_evaluator.py`)
2.  **OpenROAD 评估脚本** (`src/utils/openroad_evaluator.py`)

## 1. ICCAD2015 评估脚本
**用途**: 快速评估 ICCAD2015 Benchmark 格式的 DEF 文件的时序指标（TNS, WNS）。此脚本基于 DREAMPlace 的 Timer（OpenTimer）进行静态时序分析，只需几秒钟即可完成，适用于优化过程中的快速迭代。

### 工作流程
1.  加载 Benchmark 的配置（LEF, Verilog, Lib, SDC等）。
2.  读取用户提供的 DEF 文件。
3.  初始化 DREAMPlace 的 Timer。
4.  运行静态时序分析（Static Timing Analysis, STA）。
5.  输出 TNS (Total Negative Slack) 和 WNS (Worst Negative Slack)。

### 使用方法

**命令行调用**:
```bash
python3 src/utils/timing_evaluator.py \
    --def_path /path/to/your/placed.def \
    --benchmark superblue1 \
    --verbose
```

**Python 代码调用**:
```python
from src.utils.timing_evaluator import evaluate_iccad2015_timing

result = evaluate_iccad2015_timing(
    def_path="/path/to/your/placed.def",
    benchmark_name="superblue1"
)

if result:
    print(f"TNS: {result['tns']}, WNS: {result['wns']}")
```

---

## 2. OpenROAD 评估脚本
**用途**: 针对 OpenROAD (ORFS) 流程的设计（如 Ariane, BlackParrot 等）进行全流程评估。由于 ICCAD2015 Benchmark 主要是为了学术竞赛，而实际芯片设计（如 Ariane）需要经过完整的物理设计流程（Detailed Placement, CTS, Routing, RC Extraction）才能获得准确的线长（Wirelength）、时序和功耗数据。

### 工作流程
此脚本调用 `OpenROAD-flow-scripts` 的 `make` 流程，具体执行以下两个主要命令：

1.  **`make run_mp` (Macro Placement Prep)**:
    *   读取用户提供的 DEF 文件 (`MACRO_DEF`)。
    *   提取其中 Macro 的位置信息。
    *   生成一个 Tcl 文件 (通常位于 `results/.../macro_out`)，供后续流程使用。这一步确保了由 BBOPlace 生成的 Macro 布局被准确地注入到 OpenROAD 流程中。

2.  **`make run_wo_synth` (Physical Design Flow)**:
    *   **Floorplan**: 初始化芯片布局，加载 IO 位置。
    *   **Macro Place**: *关键步骤*，读取上一步生成的 `macro_out`，将 Macro 固定在 BBOPlace 指定的位置。
    *   **Place**: 执行标准单元放置（Global Placement & Detail Placement）。在 Macro 固定的情况下，OpenROAD 会自动放置成千上万个标准单元。
    *   **CTS (Clock Tree Synthesis)**: 时钟树综合。
    *   **Route**: 全局布线（Global Route）和详细布线（Detail Route）。此步骤产生真实的线长数据。
    *   **Finish**: 生成最终版网表和寄生参数提取（RC Extraction）。
    *   **STA**: 最终的静态时序分析和功耗分析。

3.  **Parse**: 解析 OpenROAD 的日志，提取关键指标：`GRT_WL`（全局布线线长）, `DRT_WL`（详细布线线长）, `WNS`, `TNS`, `Power`。

### 使用方法

**命令行调用**:
```bash
python3 src/utils/openroad_evaluator.py \
    --def_path /path/to/macro_placed.def \
    --design ariane133 \
    --platform nangate45 \
    --variant eval_xp
```
*   `--variant`: 为了避免覆盖原有的运行结果，建议指定一个独特的 variant 名称（默认 `eval_xp`），脚本会在 `src/utils/eval_xp/` 或指定目录下生成日志。

**Python 代码调用**:
```python
from src.utils.openroad_evaluator import run_evaluation, get_project_root

root_dir = get_project_root()
metrics = run_evaluation(
    def_path="/path/to/macro_placed.def",
    design="ariane133",
    platform="nangate45",
    variant="eval_experiment_1",
    work_dir="./eval_logs", 
    root_dir=root_dir
)

if metrics:
    print(metrics)
# Output: {'GRT_WL': 12345.6, 'DRT_WL': 12300.0, 'WNS': -1.5, 'TNS': -100.0, 'Power': 0.45}
```

### 注意事项
*   OpenROAD 流程耗时较长（视设计规模而定，Ariane 可能需要 20-40 分钟），不像 ICCAD2015 评估那样是即时的。
*   确保 `thirdparty/OpenROAD-flow-scripts` 下的环境配置正确。
