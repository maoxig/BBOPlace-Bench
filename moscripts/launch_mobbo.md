# MOBBO 批量启动说明

## 目标
- 用一个 tmux 会话批量拉起 `moscripts/MO_*.sh`
- 每个任务独立窗口，便于单独排查
- 支持按 benchmark/mode/placer/algo 拆分
- 支持可选 docker 包装运行（每个窗口独立容器）

## 相关脚本
- 启动器：`moscripts/launch_mobbo_tmux.sh`
- Docker 包装：`moscripts/docker_run_wrapper.sh`

## 快速开始

### 1) 先预览（不会真正启动）
```bash
DRY_RUN=1 bash moscripts/launch_mobbo_tmux.sh
```

### 2) 启动第二轮（默认 round=seed2）
```bash
bash moscripts/launch_mobbo_tmux.sh
```

### 3) 启动第三轮
```bash
ROUND_LABEL=seed3 bash moscripts/launch_mobbo_tmux.sh
```

## 拆分/筛选

### 按 benchmark
```bash
FILTER_BENCHMARKS=ICCAD2015 bash moscripts/launch_mobbo_tmux.sh
```

### 按 GP/MP
```bash
FILTER_MODES=GP bash moscripts/launch_mobbo_tmux.sh
FILTER_MODES=MP bash moscripts/launch_mobbo_tmux.sh
```

### 组合筛选
```bash
FILTER_BENCHMARKS=OPENROAD FILTER_MODES=MP FILTER_PLACERS=MGO bash moscripts/launch_mobbo_tmux.sh
```

### 断点/分批：START_INDEX + LIMIT
```bash
# 从第 10 个开始，最多启动 5 个
START_INDEX=10 LIMIT=5 bash moscripts/launch_mobbo_tmux.sh
```

## 逐窗口手动确认执行

如果你希望每个窗口都先停住，手动按 Enter 后再开始跑：
```bash
CONFIRM_BEFORE_RUN=1 bash moscripts/launch_mobbo_tmux.sh
```

行为说明：
- 每个窗口会先显示 `READY` 提示并等待输入。
- 在该窗口按 Enter 才会真正执行对应脚本。
- 若不想执行该窗口，可按 `Ctrl+C` 跳过。

## Docker 模式（可选）

### 使用 docker 包装器运行每个窗口
```bash
ENABLE_DOCKER=1 DOCKER_WRAPPER_SCRIPT=moscripts/docker_run_wrapper.sh bash moscripts/launch_mobbo_tmux.sh
```

### 先 dry-run 检查
```bash
DRY_RUN=1 ENABLE_DOCKER=1 bash moscripts/launch_mobbo_tmux.sh
```

## docker_run_wrapper.sh 可配项
- `DOCKER_IMAGE`（默认 `crt/bboplace-bench:cuda`）
- `DOCKER_CONTAINER_PREFIX`（默认 `mo-bbo`）
- `DOCKER_NETWORK_MODE`（默认 `host`）
- `DOCKER_USE_PRIVILEGED`（默认 `1`）
- `DOCKER_GPUS`（默认 `all`）

示例：
```bash
DOCKER_IMAGE=crt/bboplace-bench:cuda \
DOCKER_CONTAINER_PREFIX=mo-bbo \
ENABLE_DOCKER=1 \
bash moscripts/launch_mobbo_tmux.sh
```

## 常用查看命令
```bash
tmux list-windows -t MOBBO
tmux attach -t MOBBO
```

## 日志与临时目录治理（含 wandb）

默认日志会写到：
`run_logs/<ROUND_LABEL>/`

默认临时产物（含 `TMPDIR` 与 `WANDB_*`）会写到：
`run_artifacts/<ROUND_LABEL>/<script_name>/`

这样可以避免在 `moscripts/` 下继续堆积 `tmp*` 与 `.log`。

你也可以自定义：
```bash
LOG_DIR=/data/xp/mobbo_logs \
ARTIFACT_DIR=/data/xp/mobbo_artifacts \
bash moscripts/launch_mobbo_tmux.sh
```

清理已产生的 `moscripts/tmp*` 目录（启动前执行）：
```bash
CLEAN_TMP_DIRS_IN_MOSCRIPTS=1 bash moscripts/launch_mobbo_tmux.sh
```

## 调试与环境建议

如果窗口一闪而过，请开启（默认已开启）：
```bash
TMUX_REMAIN_ON_EXIT=1 bash moscripts/launch_mobbo_tmux.sh
```

如果需要在 tmux 窗口内显式激活 conda 环境：
```bash
CONDA_ENV_NAME=base bash moscripts/launch_mobbo_tmux.sh
```

说明：
- 脚本会用 `bash -lc` 启动每个窗口命令，尽量贴近交互式 shell 环境。
- 当任务报错退出时，窗口会保留（dead 状态），可直接查看报错信息。
