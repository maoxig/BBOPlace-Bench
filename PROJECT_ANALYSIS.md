# BBOPlace-Bench 项目分析报告

## 1. 项目概述

**BBOPlace-Bench** 是一个用于芯片布局（Chip Placement）任务的黑盒优化（Black-Box Optimization, BBO）算法基准测试框架。该项目旨在评估不同的 BBO 算法在芯片布局问题上的性能。

*   **核心目标**: 提供一个统一的平台，对比不同优化算法（如 EA, BO, SA, PSO）在不同布局策略（MGO, SP, HPO）下的表现。
*   **基准测试集**: 支持 ISPD2005 和 ICCAD2015 等标准基准。
*   **集成工具**: 集成了 DREAMPlace 作为第三方标准布局器，用于 HPO（超参数优化）和评估。

## 2. 目录结构分析

项目根目录结构如下：

*   `benchmarks/`: 存放基准测试数据（需用户下载）。
*   `config/`: 存放配置文件。
*   `demo/`: 演示脚本。
*   `script/`: 运行实验的 Shell 脚本。
*   `src/`: 核心源代码。
*   `thirdparty/`: 第三方库（主要是 DREAMPlace）。
*   `results/`: 实验结果输出目录（运行时生成）。

---

## 3. 详细文件与模块分析

### 3.1 配置模块 (`config/`)

该目录管理实验的所有配置参数。

*   **`config/default.yaml`**: 全局默认配置。
    *   包含 `placer` (布局器类型), `algorithm` (算法类型), `benchmark` (基准名), `seed` (随机种子), `max_evals` (最大评估次数) 等。
    *   配置了 WandB (Weights & Biases) 用于日志记录。
    *   配置了 Ray 用于并行计算。
*   **`config/benchmark.py`**: 基准测试定义。
    *   定义了 `benchmark_dict` (包含 ISPD2005 和 ICCAD2015 的具体电路名)。
    *   定义了路径 `benchmark_path_dict` 和类型 `benchmark_type_dict` (aux/def)。
*   **`config/algorithm/*.yaml`**: 算法特定配置。
    *   例如 `ea.yaml` 定义了种群大小 (`n_population`)、变异/交叉概率等。
*   **`config/placer/*.yaml`**: 布局器特定配置。
    *   例如 `mgo.yaml` 定义了网格大小 (`n_grid_x`, `n_grid_y`)。

### 3.2 脚本模块 (`script/`)

包含用于复现论文实验的 Shell 脚本。

*   **示例**: `ICCAD2015_HPO_BO_GP.sh`
    *   设置环境变量 `problem_formulation=hpo`, `algo=bo`。
    *   循环遍历基准电路（如 `superblue1`）。
    *   调用 `src/main.py` 并传递命令行参数。

### 3.3 核心源码模块 (`src/`)

这是项目的核心部分。

#### 3.3.1 入口与主流程
*   **`src/main.py`**: 程序入口。
    *   **功能**: 解析参数，加载配置，初始化日志，启动 Ray，实例化 `PlaceDB`, `Placer`, `Algorithm` 并运行。
    *   **关键函数**: `single_run(args)` 执行单次实验流程。
*   **`src/evaluator.py`**: 评估器封装。
    *   **类 `Evaluator`**: 封装了 `PlaceDB` 和 `Placer`，提供统一的 `evaluate(x)` 接口。
    *   **属性**: `n_dim` (维度), `xl` (下界), `xu` (上界)，根据不同的 `placer` 类型动态计算。
*   **`src/placedb.py`**: 数据管理。
    *   **类 `PlaceDB`**: 负责读取和解析基准文件（支持 AUX 和 DEF 格式）。
    *   **功能**: 存储节点信息 (`node_info`)、网表信息 (`net_info`)、画布尺寸等。提供 `to_pl` 和 `to_def` 方法将布局结果导出为文件。

#### 3.3.2 布局策略 (`src/placer/`)
定义了如何将优化变量映射为具体的芯片布局。

*   **`src/placer/basic_placer.py`**: 布局器基类。
    *   **类 `BasicPlacer`**: 定义了通用接口 `evaluate` (支持并行), `save_placement`, `plot`。
    *   **抽象方法**: `_genotype2phenotype(x)` 将基因型（优化变量）转换为表现型（宏模块坐标）。
*   **`src/placer/mgo_placer.py`**: 掩码引导优化 (Mask-Guided Optimization)。
    *   **原理**: 将画布划分为网格，优化变量为每个宏模块的网格坐标。
    *   **实现**: `_genotype2phenotype` 使用贪心策略结合掩码（Mask）来寻找合法的放置位置，避免重叠并优化线长。
*   **`src/placer/sp_placer.py`**: 序列对 (Sequence Pair)。
    *   **原理**: 使用两个序列（Sequence Pair）来表示模块的拓扑关系。
    *   **实现**: `_genotype2phenotype` 将两个序列转换为有向图（水平约束图和垂直约束图），通过计算最长路径确定模块坐标。
*   **`src/placer/hpo_placer.py`**: 超参数优化 (Hyperparameter Optimization)。
    *   **原理**: 优化 DREAMPlace 的超参数（如学习率、密度权重等）。
    *   **实现**: 通过 Unix Socket 与 `dmp_worker.py` 通信，将参数发送给 DREAMPlace 进程执行布局，并获取 HPWL 结果。
*   **`src/placer/dmp_worker.py`**: DREAMPlace 工作进程。
    *   **功能**: 独立进程，加载 DREAMPlace，接收参数更新，执行布局，返回结果。避免了频繁重启 DREAMPlace 的开销。

#### 3.3.3 算法实现 (`src/algorithm/`)
实现了各种 BBO 算法。

*   **`src/algorithm/basic_algo.py`**: 算法基类。
    *   **功能**: 记录结果，保存检查点 (`checkpoint`)。
*   **`src/algorithm/ea/vanilla_ea.py`**: 遗传算法 (GA)。
    *   **实现**: 基于 `pymoo` 库。定义了 `VanillaEA` 类，使用 `GA` 算法，配置了自定义的交叉和变异算子。
*   **`src/algorithm/__init__.py`**: 注册表。
    *   将 `ea`, `bo`, `sa`, `es`, `pso` 映射到对应的类。

#### 3.3.4 问题定义 (`src/problem/`)
适配 `pymoo` 的问题接口。

*   **`src/problem/pymoo_problem.py`**:
    *   **类 `PlacementProblem`**: 继承自 `pymoo.core.problem.Problem`。
    *   **子类**:
        *   `MaskGuidedOptimizationPlacementProblem`: 定义 MGO 的变量范围（网格坐标）。
        *   `SequencePairPlacementProblem`: 定义 SP 的变量范围（排列索引）。
        *   `HyperparameterPlacementProblem`: 定义 HPO 的变量范围（参数空间）。

#### 3.3.5 算子 (`src/operators/`)
定义了遗传算法的交叉和变异操作。

*   **`src/operators/crossover.py`**:
    *   `SPOrderCrossover`: 针对序列对的顺序交叉。
    *   `GuidGuideSBXCrossover`: 针对网格的 SBX 交叉。
*   **`src/operators/mutation.py`**:
    *   `MaskGuidedOptimizationSwapMutation`: 交换变异。
    *   `MaskGuidedOptimizationShiftMutation`: 移位变异。
    *   `SPInversionMutation`: 序列反转变异。

#### 3.3.6 工具 (`src/utils/`)
*   **`src/utils/plot.py`**: 可视化工具。
    *   使用 DREAMPlace 的绘图功能或 Matplotlib 绘制布局图。
*   **`src/utils/read_benchmark/`**: 读取 AUX/DEF 格式基准文件的工具。

## 4. 总结

BBOPlace-Bench 是一个结构清晰、模块化程度高的基准测试框架。它通过抽象 `Placer`（布局策略）和 `Algorithm`（优化算法），使得研究人员可以方便地：
1.  测试新的 BBO 算法在芯片布局问题上的效果。
2.  开发新的布局编码策略（如 MGO, SP）。
3.  利用现有的 DREAMPlace 工业级布局器进行超参数调优。

代码使用了 `Ray` 进行并行加速，使用 `pymoo` 作为进化算法的后端，具有较好的扩展性和性能。
