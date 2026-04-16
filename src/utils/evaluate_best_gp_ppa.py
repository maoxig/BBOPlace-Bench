import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import pandas as pd

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./src"))

from config.benchmark import ROOT_DIR, BENCHMARK_DIR

THIRDPARTY_DIR = os.path.join(ROOT_DIR, "thirdparty")
DREAMPLACE_DIR = os.path.join(THIRDPARTY_DIR, "dreamplace")
SOURCE_DIR = os.path.join(ROOT_DIR, "src")

sys.path.extend([ROOT_DIR, BENCHMARK_DIR, THIRDPARTY_DIR, DREAMPLACE_DIR, SOURCE_DIR])
os.environ["PYTHONPATH"] = ":".join(sys.path)

BENCHMARKS = {
    "OpenROAD": {
        "cases": ["ariane133", "ariane136", "bp", "bp_be", "bp_fe", "swerv_wrapper"],
    },
    "ICCAD2015": {
        "cases": [
            "superblue1",
            "superblue3",
            "superblue4",
            "superblue5",
            "superblue7",
            "superblue10",
            "superblue16",
            "superblue18",
        ],
    },
}

FORMULATIONS = ["MGO", "HPO"]
MODES = ["GP", "MP"]
DEFAULT_DEF_PER_SEED = 5


ACTIVE_PROCS = set()
ACTIVE_PROCS_LOCK = threading.Lock()
RUNNING_TASKS = {}
RUNNING_TASKS_LOCK = threading.Lock()


def register_proc(proc):
    with ACTIVE_PROCS_LOCK:
        ACTIVE_PROCS.add(proc)


def unregister_proc(proc):
    with ACTIVE_PROCS_LOCK:
        ACTIVE_PROCS.discard(proc)


def register_running_task(task_key, text):
    with RUNNING_TASKS_LOCK:
        RUNNING_TASKS[task_key] = {"text": text, "start": time.time()}


def unregister_running_task(task_key):
    with RUNNING_TASKS_LOCK:
        RUNNING_TASKS.pop(task_key, None)


def snapshot_running_tasks():
    with RUNNING_TASKS_LOCK:
        return dict(RUNNING_TASKS)


def _read_linux_children_pids(pid):
    path = f"/proc/{int(pid)}/task/{int(pid)}/children"
    try:
        with open(path, "r") as f:
            text = f.read().strip()
        if not text:
            return []
        return [int(x) for x in text.split() if x.strip().isdigit()]
    except Exception:
        return []


def _kill_process_tree(root_pid, sig):
    # Some tools may spawn detached descendants; walk Linux /proc tree and signal them too.
    seen = set()
    queue = [int(root_pid)]
    while queue:
        pid = queue.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        queue.extend(_read_linux_children_pids(pid))

    for pid in sorted(seen, reverse=True):
        try:
            os.kill(pid, sig)
        except Exception:
            pass


def terminate_active_processes(sig=signal.SIGTERM):
    with ACTIVE_PROCS_LOCK:
        procs = list(ACTIVE_PROCS)
    for p in procs:
        try:
            if p.poll() is None:
                os.killpg(p.pid, sig)
                _kill_process_tree(p.pid, sig)
        except Exception:
            pass


def parse_seeds(seed_text):
    items = [x.strip() for x in str(seed_text).split(",") if x.strip()]
    if not items:
        raise ValueError("--seeds is empty")
    return [int(x) for x in items]


def seed_tag(seeds):
    return "_".join([str(int(s)) for s in seeds])


def modes_tag(modes):
    return "_".join([str(m).strip().lower() for m in modes])


def normalize_run_path(workspace_root, run_path):
    if not run_path:
        return run_path
    rp = str(run_path)
    candidates = [rp]

    if rp.startswith("/workspace/"):
        candidates.append(os.path.join(workspace_root, rp[len("/workspace/") :]))

    marker = "/results/"
    if marker in rp:
        suffix = rp.split(marker, 1)[1]
        candidates.append(os.path.join(workspace_root, "results", suffix))

    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]


def find_default_hv_json(workspace_root, output_dir, seed):
    candidates = [
        os.path.join(output_dir, f"hv_summary_seed_{seed}.json"),
        os.path.join(workspace_root, "results", "analysis_reports", "hv", "json", f"hv_summary_seed_{seed}.json"),
        os.path.join(workspace_root, "results", "analysis_reports", "hv", f"hv_summary_seed_{seed}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def find_default_hv_json_multi(workspace_root, output_dir, seeds):
    tag = seed_tag(seeds)
    candidates = [
        os.path.join(output_dir, f"hv_summary_seeds_{tag}.json"),
        os.path.join(workspace_root, "results", "analysis_reports", "hv", "json", f"hv_summary_seeds_{tag}.json"),
        os.path.join(workspace_root, "results", "analysis_reports", "hv", f"hv_summary_seeds_{tag}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def parse_csv_or_all(value, valid_values=None):
    if value is None:
        return None

    raw = value.strip()
    if raw.lower() == "all" or raw == "":
        return None

    items = [x.strip() for x in raw.split(",") if x.strip()]
    if valid_values is not None:
        valid_lower = {x.lower(): x for x in valid_values}
        normalized = []
        for item in items:
            key = item.lower()
            if key not in valid_lower:
                raise ValueError(f"Unsupported value '{item}', valid options: {sorted(valid_values)}")
            normalized.append(valid_lower[key])
        return normalized

    return items


def load_hv_summary(summary_json, seed=None, seeds=None):
    with open(summary_json, "r") as f:
        data = json.load(f)

    if seeds is not None:
        json_seeds = [int(x) for x in data.get("seeds", [])]
        req_seeds = [int(x) for x in seeds]
        if json_seeds and sorted(json_seeds) != sorted(req_seeds):
            print(f"[WARN] hv json seeds={json_seeds}, requested seeds={req_seeds}. Continue anyway.")
    elif seed is not None:
        seed_in_json = data.get("seed")
        if seed_in_json is not None and int(seed_in_json) != int(seed):
            print(f"[WARN] hv json seed={seed_in_json}, requested seed={seed}. Continue anyway.")

    return data


def find_def_candidates(run_path, def_per_seed, mode="GP"):
    placements_dir = os.path.join(run_path, "placements")
    if not os.path.isdir(placements_dir):
        return []

    mode_upper = str(mode).upper()
    preferred_prefix = "gp" if mode_upper == "GP" else "mp"

    selected = []
    for i in range(1, int(def_per_seed) + 1):
        preferred_def = os.path.join(placements_dir, f"{preferred_prefix}_{i}.def")
        base_def = os.path.join(placements_dir, f"{i}.def")

        if os.path.exists(preferred_def):
            selected.append((i, preferred_def, True))
        elif os.path.exists(base_def):
            selected.append((i, base_def, False))

    return selected


def _pick_evenly_spaced(items, budget):
    if budget <= 0:
        return []
    if len(items) <= budget:
        return list(items)

    picks = []
    used = set()
    if budget == 1:
        return [items[0]]

    for i in range(budget):
        idx = int(round(i * (len(items) - 1) / (budget - 1)))
        if idx not in used:
            picks.append(items[idx])
            used.add(idx)

    if len(picks) < budget:
        for i, item in enumerate(items):
            if i in used:
                continue
            picks.append(item)
            used.add(i)
            if len(picks) >= budget:
                break
    return picks


def select_task_candidates(candidates, max_total_defs_per_setting, def_select_strategy):
    if max_total_defs_per_setting is None or int(max_total_defs_per_setting) <= 0:
        return candidates

    budget = int(max_total_defs_per_setting)
    if len(candidates) <= budget:
        return candidates

    strategy = str(def_select_strategy)

    if strategy == "top_rank":
        ordered = sorted(candidates, key=lambda x: (x["def_rank"], x["seed"]))
        return ordered[:budget]

    if strategy == "rank_spread":
        ordered = sorted(candidates, key=lambda x: (x["def_rank"], x["seed"]))
        return _pick_evenly_spaced(ordered, budget)

    if strategy == "seed_round_robin":
        by_seed = {}
        for c in candidates:
            by_seed.setdefault(int(c["seed"]), []).append(c)
        for s in by_seed:
            by_seed[s] = sorted(by_seed[s], key=lambda x: (x["def_rank"], x["run_path"]))

        picked = []
        seeds_sorted = sorted(by_seed.keys())
        rank = 0
        while len(picked) < budget:
            any_added = False
            for s in seeds_sorted:
                lst = by_seed[s]
                if rank < len(lst):
                    picked.append(lst[rank])
                    any_added = True
                    if len(picked) >= budget:
                        break
            if not any_added:
                break
            rank += 1
        return picked[:budget]

    raise ValueError(f"Unsupported --def_select_strategy={strategy}")


def parse_iccad_output(stdout, report_path):
    metrics = {}

    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    key = k.strip()
                    value = v.strip()
                    if key in {"n_tns", "n_wns", "Runtime", "runtime"}:
                        try:
                            if key.lower() == "runtime":
                                metrics["runtime_sec"] = float(value.split()[0])
                            else:
                                metrics[key] = float(value.split()[0])
                        except ValueError:
                            pass
        except Exception:
            pass

    patterns = {
        "n_tns": r"n_tns \(score\):\s*([-+eE0-9\.]+)",
        "n_wns": r"n_wns \(score\):\s*([-+eE0-9\.]+)",
        "runtime_sec": r"Runtime:\s*([-+eE0-9\.]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, stdout)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass

    return metrics if metrics else None


def parse_openroad_output(stdout, work_dir, min_mtime=None):
    metrics = {}
    metrics_file = os.path.join(work_dir, "metrics.txt")

    if os.path.exists(metrics_file):
        if min_mtime is not None:
            try:
                if os.path.getmtime(metrics_file) < float(min_mtime):
                    return None
            except Exception:
                pass
        try:
            with open(metrics_file, "r") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    key = k.strip()
                    value = v.strip()
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = value
        except Exception:
            pass

    patterns = {
        "GRT_WL": r"GRT_WL:\s*([-+eE0-9\.]+)",
        "DRT_WL": r"DRT_WL:\s*([-+eE0-9\.]+)",
        "DRC": r"DRC:\s*([-+eE0-9\.]+)",
        "StdCellArea": r"StdCellArea:\s*([-+eE0-9\.]+)",
        "WNS": r"WNS:\s*([-+eE0-9\.]+)",
        "TNS": r"TNS:\s*([-+eE0-9\.]+)",
        "Power": r"Power:\s*([-+eE0-9\.]+)",
        "runtime": r"runtime:\s*([-+eE0-9\.]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, stdout)
        if m and key not in metrics:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass

    return metrics if metrics else None


def find_reusable_openroad_metrics(run_path, def_path):
    def_stem = os.path.basename(str(def_path)).replace(".def", "")
    base = os.path.join(str(run_path), "ppa_eval")
    if not os.path.isdir(base):
        return None

    candidates = []
    for session in os.listdir(base):
        metrics_file = os.path.join(base, session, def_stem, "metrics.txt")
        if os.path.isfile(metrics_file):
            try:
                mtime = os.path.getmtime(metrics_file)
            except Exception:
                mtime = 0.0
            candidates.append((mtime, os.path.dirname(metrics_file)))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    for _, work_dir in candidates:
        cached = parse_openroad_output("", work_dir, min_mtime=None)
        if cached is not None:
            return cached
    return None


def write_partial_results(output_dir, seeds, multi_seed, result_rows, eval_tag):
    if multi_seed:
        json_path = os.path.join(output_dir, f"ppa_eval_seeds_{seed_tag(seeds)}_{eval_tag}.partial.json")
        csv_path = os.path.join(output_dir, f"ppa_eval_seeds_{seed_tag(seeds)}_{eval_tag}.partial.csv")
    else:
        json_path = os.path.join(output_dir, f"ppa_eval_seed_{seeds[0]}_{eval_tag}.partial.json")
        csv_path = os.path.join(output_dir, f"ppa_eval_seed_{seeds[0]}_{eval_tag}.partial.csv")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "n_rows": len(result_rows),
        "rows": result_rows,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame(result_rows).to_csv(csv_path, index=False)


def build_task_variant(task, base_variant):
    key = f"{task.get('seed')}|{task.get('benchmark')}|{task.get('case')}|{task.get('formulation')}|{task.get('def_path')}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    return f"{base_variant}_s{int(task.get('seed', 0))}_r{int(task.get('def_rank', 0))}_{h}"


def evaluate_one_def_subprocess(
    benchmark,
    case_name,
    def_path,
    workspace_root,
    platform,
    variant,
    eval_base_dir,
    timeout_sec,
    cleanup_flow_work,
    reuse_existing_results,
    run_path,
):
    start = time.time()

    if benchmark == "ICCAD2015":
        cmd = [
            sys.executable,
            os.path.join(workspace_root, "src", "utils", "iccad2015_evaluator.py"),
            "--def_path",
            def_path,
            "--benchmark",
            case_name,
            "--root_dir",
            workspace_root,
        ]
        report_path = f"{def_path}.timing.txt"
        if reuse_existing_results and os.path.exists(report_path):
            cached = parse_iccad_output("", report_path)
            if cached is not None:
                return cached, "reused existing report", 0, 0.0
    elif benchmark == "OpenROAD":
        work_dir = os.path.join(eval_base_dir, os.path.basename(def_path).replace(".def", ""))
        os.makedirs(work_dir, exist_ok=True)
        if reuse_existing_results and os.path.exists(os.path.join(work_dir, "metrics.txt")):
            cached = parse_openroad_output("", work_dir, min_mtime=None)
            if cached is not None:
                return cached, "reused existing metrics", 0, 0.0
        if reuse_existing_results:
            cached = find_reusable_openroad_metrics(run_path, def_path)
            if cached is not None:
                # Persist into current session work dir for consistency.
                metrics_file = os.path.join(work_dir, "metrics.txt")
                with open(metrics_file, "w") as f:
                    for k, v in cached.items():
                        f.write(f"{k}: {v}\n")
                return cached, "reused existing metrics from prior session", 0, 0.0
        cmd = [
            sys.executable,
            os.path.join(workspace_root, "src", "utils", "openroad_evaluator.py"),
            "--def_path",
            def_path,
            "--design",
            case_name,
            "--platform",
            platform,
            "--variant",
            variant,
            "--work_dir",
            work_dir,
            "--flow_work_home",
            os.path.join(work_dir, "orfs_work"),
            "--cleanup_flow_work",
            cleanup_flow_work,
        ]
        if reuse_existing_results:
            cmd.append("--reuse_existing_logs")
    else:
        return None, "unsupported benchmark", -1, 0.0

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=workspace_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        register_proc(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            return None, "timeout", -9, time.time() - start
        except KeyboardInterrupt:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            raise
    finally:
        if proc is not None:
            unregister_proc(proc)

    duration = time.time() - start
    stdout = stdout or ""
    stderr = stderr or ""
    return_code = proc.returncode if proc is not None else -1

    if benchmark == "ICCAD2015":
        metrics = parse_iccad_output(stdout, report_path)
    else:
        metrics = parse_openroad_output(stdout, work_dir, min_mtime=start)

    err_text = ""
    if return_code != 0:
        err_text = (stderr.strip() or stdout.strip())[-800:]
    elif metrics is None:
        err_text = "no metrics parsed from this run"

    return metrics, err_text, return_code, duration


def parse_case_filters(args):
    all_cases = BENCHMARKS["ICCAD2015"]["cases"] + BENCHMARKS["OpenROAD"]["cases"]
    common_cases = parse_csv_or_all(args.cases, valid_values=all_cases)
    iccad_cases = parse_csv_or_all(args.iccad_cases, valid_values=BENCHMARKS["ICCAD2015"]["cases"])
    openroad_cases = parse_csv_or_all(args.openroad_cases, valid_values=BENCHMARKS["OpenROAD"]["cases"])

    case_filters = {
        "ICCAD2015": iccad_cases,
        "OpenROAD": openroad_cases,
    }

    # --cases acts as a global additional filter.
    if common_cases is not None:
        common_set = set(common_cases)
        for bench in BENCHMARKS:
            bench_cases = set(BENCHMARKS[bench]["cases"])
            bench_common = bench_cases.intersection(common_set)

            if case_filters[bench] is None:
                case_filters[bench] = sorted(bench_common)
            else:
                current_cases = case_filters[bench] or []
                case_filters[bench] = sorted(set(current_cases).intersection(bench_common))

    return common_cases, case_filters


def run_single_task(
    task,
    workspace_root,
    platform,
    variant,
    timeout_sec,
    session_tag,
    cleanup_flow_work,
    max_retries,
    retry_backoff_sec,
    reuse_existing_results,
):
    eval_base_dir = os.path.join(task["run_path"], "ppa_eval", session_tag)
    os.makedirs(eval_base_dir, exist_ok=True)
    task_variant = build_task_variant(task, variant)
    task_key = f"{task_variant}:{task.get('def_path')}"
    task_text = (
        f"seed={task.get('seed')} {task.get('benchmark')}/{task.get('case')}/{task.get('formulation')} "
        f"def#{task.get('def_rank')}"
    )
    register_running_task(task_key, task_text)

    try:
        attempts_used = 0
        metrics = None
        err_text = ""
        return_code = -1
        duration = 0.0
        max_attempts = max(1, int(max_retries) + 1)
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            metrics, err_text, return_code, duration = evaluate_one_def_subprocess(
                benchmark=task["benchmark"],
                case_name=task["case"],
                def_path=task["def_path"],
                workspace_root=workspace_root,
                platform=platform,
                variant=task_variant,
                eval_base_dir=eval_base_dir,
                timeout_sec=timeout_sec,
                cleanup_flow_work=cleanup_flow_work,
                reuse_existing_results=reuse_existing_results,
                run_path=task["run_path"],
            )
            if metrics is not None and return_code == 0:
                break
            if attempt < max_attempts:
                print(
                    f"[RETRY] {task['benchmark']}/{task['case']}/{task['formulation']} "
                    f"def#{task['def_rank']} attempt {attempt}/{max_attempts} failed, retrying..."
                )
                if float(retry_backoff_sec) > 0:
                    time.sleep(float(retry_backoff_sec))

        row = dict(task)
        row.update(
            {
                "seed": int(task.get("seed", 0)),
                "eval_ok": metrics is not None and return_code == 0,
                "attempts_used": int(attempts_used),
                "return_code": return_code,
                "duration_sec": duration,
                "error": err_text,
                "flow_variant": task_variant,
                "eval_session": session_tag,
            }
        )
        if metrics:
            row.update(metrics)

        return row
    finally:
        unregister_running_task(task_key)


def print_result_row(row, benchmark):
    if row.get("eval_ok"):
        if benchmark == "ICCAD2015":
            print(
                "[RESULT] "
                f"n_tns={row.get('n_tns')} "
                f"n_wns={row.get('n_wns')} "
                f"runtime_sec={row.get('runtime_sec', row.get('duration_sec')):.2f}"
            )
        else:
            print(
                "[RESULT] "
                f"WNS={row.get('WNS')} TNS={row.get('TNS')} "
                f"GRT_WL={row.get('GRT_WL')} DRT_WL={row.get('DRT_WL')} "
                f"DRC={row.get('DRC')} StdCellArea={row.get('StdCellArea')} "
                f"runtime_sec={row.get('runtime', row.get('duration_sec')):.2f}"
            )
    else:
        print(f"[RESULT] FAILED rc={row.get('return_code')} err={row.get('error')}")


def _clear_live_line():
    if sys.stdout.isatty():
        sys.stdout.write("\r\x1b[2K")


def _truncate_text(text, width):
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def print_live_status(line):
    if not sys.stdout.isatty():
        return
    cols = shutil.get_terminal_size((120, 20)).columns
    _clear_live_line()
    sys.stdout.write(_truncate_text(line, max(20, cols - 1)))
    sys.stdout.flush()


def print_log_line(line, compact=False):
    if compact and sys.stdout.isatty():
        _clear_live_line()
        print(line)
        return
    print(line)


def build_task_plan(
    hv_summary,
    selected_modes,
    selected_benchmarks,
    selected_formulations,
    benchmark_case_filters,
    seeds,
    workspace_root,
    def_per_seed,
    max_total_defs_per_setting,
    def_select_strategy,
    algo_eval_mode,
):
    tasks = []
    multi_seed = bool(hv_summary.get("multi_seed"))

    for benchmark, bench_cfg in BENCHMARKS.items():
        if selected_benchmarks is not None and benchmark not in selected_benchmarks:
            continue

        bench_data = hv_summary.get("benchmarks", {}).get(benchmark, {})
        case_filter = benchmark_case_filters.get(benchmark)

        for mode_name in selected_modes:
            mode_data = bench_data.get("modes", {}).get(mode_name, {})
            mode_cases = mode_data.get("cases", {})

            for case_name in bench_cfg["cases"]:
                if case_filter is not None and case_name not in case_filter:
                    continue

                case_data = mode_cases.get(case_name, {})
                for form in FORMULATIONS:
                    if selected_formulations is not None and form not in selected_formulations:
                        continue

                    form_data = case_data.get(form, {})
                    setting_candidates = []
                    if multi_seed:
                        algorithms_data = form_data.get("algorithms", {}) or {}
                        if str(algo_eval_mode) == "all_algorithms":
                            algo_names = sorted(list(algorithms_data.keys()))
                        else:
                            best_algo = form_data.get("best_algo_by_mean")
                            algo_names = [best_algo] if best_algo else []

                        for algo_name in algo_names:
                            if not algo_name:
                                continue
                            algo_data = algorithms_data.get(algo_name, {}) or {}
                            hv_stats = algo_data.get("hv_stats") or {}
                            hv_mean = hv_stats.get("mean")
                            hv_by_seed = algo_data.get("hv_by_seed") or {}
                            run_path_by_seed = algo_data.get("run_path_by_seed") or {}

                            for seed in seeds:
                                run_path = normalize_run_path(workspace_root, run_path_by_seed.get(str(seed)))
                                if not run_path:
                                    continue
                                def_list = find_def_candidates(run_path, def_per_seed, mode=mode_name)
                                if not def_list:
                                    continue

                                for rank, def_path, used_pref in def_list:
                                    setting_candidates.append(
                                        {
                                            "seed": int(seed),
                                            "benchmark": benchmark,
                                            "case": case_name,
                                            "formulation": form,
                                            "mode": mode_name,
                                            "best_algo": algo_name,
                                            "best_hv": hv_by_seed.get(str(seed)),
                                            "best_hv_mean": hv_mean,
                                            "run_path": run_path,
                                            "def_rank": rank,
                                            "def_path": def_path,
                                            "used_mode_def": used_pref,
                                        }
                                    )
                    else:
                        algorithms_data = form_data.get("algorithms", {}) or {}
                        if str(algo_eval_mode) == "all_algorithms":
                            algo_names = sorted(list(algorithms_data.keys()))
                        else:
                            best_algo = form_data.get("best_algo")
                            algo_names = [best_algo] if best_algo else []

                        for algo_name in algo_names:
                            if not algo_name:
                                continue

                            algo_data = algorithms_data.get(algo_name, {}) or {}
                            # Prefer per-algorithm run_path/hv from hv summary; fallback to legacy best_* fields.
                            run_path = normalize_run_path(workspace_root, algo_data.get("run_path"))
                            hv_value = algo_data.get("hv")
                            if not run_path and str(algo_eval_mode) != "all_algorithms" and form_data.get("best_algo") == algo_name:
                                run_path = normalize_run_path(workspace_root, form_data.get("best_run_path"))
                                hv_value = form_data.get("best_hv")

                            if not run_path:
                                continue

                            def_list = find_def_candidates(run_path, def_per_seed, mode=mode_name)
                            if not def_list:
                                continue

                            for rank, def_path, used_pref in def_list:
                                setting_candidates.append(
                                    {
                                        "seed": int(seeds[0]),
                                        "benchmark": benchmark,
                                        "case": case_name,
                                        "formulation": form,
                                        "mode": mode_name,
                                        "best_algo": algo_name,
                                        "best_hv": hv_value,
                                        "run_path": run_path,
                                        "def_rank": rank,
                                        "def_path": def_path,
                                        "used_mode_def": used_pref,
                                    }
                                )

                    selected_candidates = select_task_candidates(
                        candidates=setting_candidates,
                        max_total_defs_per_setting=max_total_defs_per_setting,
                        def_select_strategy=def_select_strategy,
                    )
                    tasks.extend(selected_candidates)

    return tasks


def task_shard_index(task, num_shards):
    key = "|".join(
        [
            str(task.get("seed", "")),
            str(task.get("benchmark", "")),
            str(task.get("case", "")),
            str(task.get("formulation", "")),
            str(task.get("def_rank", "")),
            str(task.get("def_path", "")),
        ]
    )
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % int(num_shards)


def apply_task_sharding(tasks, num_shards, shard_id):
    if int(num_shards) <= 1:
        return list(tasks)
    sid0 = int(shard_id) - 1
    out = []
    for t in tasks:
        if task_shard_index(t, num_shards) == sid0:
            out.append(t)
    return out


def print_plan(tasks, preview_limit, estimate_per_def_min):
    print("\n=== Evaluation Plan ===")
    print(f"Total DEF tasks: {len(tasks)}")

    combo_set = set((t["seed"], t["benchmark"], t["case"], t["formulation"], t["best_algo"]) for t in tasks)
    print(f"Total benchmark/case/formulation combos: {len(combo_set)}")
    seed_set = sorted(set(int(t["seed"]) for t in tasks))
    print(f"Seeds covered: {seed_set}")

    if estimate_per_def_min is not None and len(tasks) > 0:
        est_total_min = estimate_per_def_min * len(tasks)
        print(f"Estimated total time: {est_total_min:.1f} min (per DEF ~ {estimate_per_def_min:.2f} min)")

    print("\nPreview:")
    for idx, t in enumerate(tasks[:preview_limit], 1):
        mode_flag = str(t.get("mode", "GP")).lower() if t.get("used_mode_def") else "base"
        print(
            f"  [{idx}] seed={t['seed']} {t['benchmark']}/{t['case']}/{t['formulation']}/{t.get('mode','GP')} "
            f"best={t['best_algo']} hv={t['best_hv']} def#{t['def_rank']}({mode_flag})"
        )
    if len(tasks) > preview_limit:
        print(f"  ... {len(tasks) - preview_limit} more tasks")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mode-specific PPA for best-HV algorithm per case/formulation (subprocess mode)"
    )
    parser.add_argument("--workspace", type=str, default=".", help="Workspace root")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("results", "analysis_reports", "ppa_eval"),
        help="Base output directory",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed to evaluate in single-seed mode")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed evaluation, e.g. 1,2,3",
    )
    parser.add_argument("--hv_json", type=str, default=None, help="Path to hv_summary_seed_<seed>.json or hv_summary_seeds_<...>.json")
    parser.add_argument("--platform", type=str, default="nangate45", help="OpenROAD platform")
    parser.add_argument("--variant", type=str, default="eval", help="OpenROAD flow variant")

    parser.add_argument("--benchmarks", type=str, default="all", help="all or comma list: ICCAD2015,OpenROAD")
    parser.add_argument("--cases", type=str, default="all", help="Global case filter (all benchmarks)")
    parser.add_argument("--iccad_cases", type=str, default="all", help="ICCAD2015-only cases, e.g. superblue1")
    parser.add_argument("--openroad_cases", type=str, default="all", help="OpenROAD-only cases, e.g. bp,bp_fe")
    parser.add_argument("--formulations", type=str, default="all", help="all or comma list: MGO,HPO")
    parser.add_argument("--modes", type=str, default="GP", help="Comma list from GP,MP (default: GP)")
    parser.add_argument(
        "--algo_eval_mode",
        type=str,
        default="best_hv",
        choices=["best_hv", "all_algorithms"],
        help="Evaluate only best-HV algorithm per setting, or evaluate all algorithms from hv summary",
    )

    parser.add_argument("--preview_limit", type=int, default=30, help="How many planned tasks to print")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker count for PPA evaluation")
    parser.add_argument(
        "--def_per_seed",
        type=int,
        default=DEFAULT_DEF_PER_SEED,
        help="Number of top DEF candidates per seed to consider before cross-seed selection",
    )
    parser.add_argument(
        "--max_total_defs_per_setting",
        type=int,
        default=0,
        help="Cross-seed DEF budget per (benchmark,case,formulation); 0 means keep all candidates",
    )
    parser.add_argument(
        "--def_select_strategy",
        type=str,
        default="seed_round_robin",
        choices=["seed_round_robin", "top_rank", "rank_spread"],
        help="Strategy when max_total_defs_per_setting is enabled",
    )
    parser.add_argument("--estimate_per_def_min", type=float, default=None, help="Optional ETA estimate per DEF")
    parser.add_argument(
        "--cleanup_flow_work",
        type=str,
        default="success",
        choices=["never", "success", "always"],
        help="Cleanup isolated ORFS work dirs to reduce disk usage",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print/save plan, do not execute evaluation")
    parser.add_argument("--confirm", action="store_true", help="Ask confirmation before execution")
    parser.add_argument("--timeout_sec", type=int, default=3600, help="Timeout per DEF subprocess in seconds (default: 3600)")
    parser.add_argument("--max_retries", type=int, default=1, help="Retry count after the first attempt (default: 1)")
    parser.add_argument("--retry_backoff_sec", type=float, default=10.0, help="Backoff seconds between retries")
    parser.add_argument(
        "--reuse_existing_results",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Reuse prior metrics/logs when available and skip rerun",
    )
    parser.add_argument(
        "--progress_style",
        type=str,
        default="auto",
        choices=["auto", "compact", "plain"],
        help="Progress rendering style: auto chooses compact for TTY, plain otherwise",
    )
    parser.add_argument("--heartbeat_sec", type=float, default=10.0, help="Heartbeat interval in seconds")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for multi-machine runs")
    parser.add_argument("--shard_id", type=int, default=1, help="Current shard id in [1, num_shards]")
    parser.add_argument("--worker_tag", type=str, default="", help="Optional worker tag to isolate output files")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 1 or args.shard_id > args.num_shards:
        raise ValueError("--shard_id must satisfy 1 <= shard_id <= num_shards")

    workspace_root = os.path.abspath(args.workspace)
    base_output_dir = os.path.abspath(args.output)
    seeds = [int(args.seed)] if args.seeds is None else parse_seeds(args.seeds)
    multi_seed = len(seeds) > 1
    if multi_seed:
        output_dir = os.path.join(base_output_dir, f"seeds_{seed_tag(seeds)}")
    else:
        output_dir = os.path.join(base_output_dir, f"seed_{seeds[0]}")

    shard_label = f"shard_{args.shard_id}of{args.num_shards}" if int(args.num_shards) > 1 else "single"
    if args.worker_tag:
        output_dir = os.path.join(output_dir, f"{shard_label}_{args.worker_tag}")
    else:
        output_dir = os.path.join(output_dir, shard_label)
    os.makedirs(output_dir, exist_ok=True)

    if args.hv_json:
        hv_json = os.path.abspath(args.hv_json)
    else:
        hv_json = find_default_hv_json_multi(workspace_root, base_output_dir, seeds) if multi_seed else find_default_hv_json(workspace_root, base_output_dir, seeds[0])
    if not os.path.exists(hv_json):
        raise FileNotFoundError(f"HV summary json not found: {hv_json}")

    selected_benchmarks = parse_csv_or_all(args.benchmarks, valid_values=list(BENCHMARKS.keys()))
    selected_formulations = parse_csv_or_all(args.formulations, valid_values=FORMULATIONS)
    selected_modes = parse_csv_or_all(args.modes, valid_values=MODES)
    if selected_modes is None:
        selected_modes = ["GP", "MP"]
    common_cases_filter, benchmark_case_filters = parse_case_filters(args)
    mode_tag_text = modes_tag(selected_modes)
    algo_scope_tag = "best" if str(args.algo_eval_mode) == "best_hv" else "all"
    eval_tag = f"{mode_tag_text}_{algo_scope_tag}"

    hv_summary = load_hv_summary(hv_json, seed=seeds[0], seeds=seeds if multi_seed else None)
    max_total_defs = None if int(args.max_total_defs_per_setting) <= 0 else int(args.max_total_defs_per_setting)
    tasks = build_task_plan(
        hv_summary,
        selected_modes,
        selected_benchmarks,
        selected_formulations,
        benchmark_case_filters,
        seeds,
        workspace_root,
        args.def_per_seed,
        max_total_defs,
        args.def_select_strategy,
        args.algo_eval_mode,
    )
    all_tasks = list(tasks)
    tasks = apply_task_sharding(tasks, args.num_shards, args.shard_id)

    if multi_seed:
        plan_out = os.path.join(output_dir, f"ppa_eval_seeds_{seed_tag(seeds)}_{eval_tag}_plan.json")
    else:
        plan_out = os.path.join(output_dir, f"ppa_eval_seed_{seeds[0]}_{eval_tag}_plan.json")
    with open(plan_out, "w") as f:
        json.dump(
            {
                "seed": int(seeds[0]),
                "seeds": [int(s) for s in seeds],
                "multi_seed": bool(multi_seed),
                "generated_at": datetime.now().isoformat(),
                "hv_json": hv_json,
                "platform": args.platform,
                "variant": args.variant,
                "def_per_seed": int(args.def_per_seed),
                "max_total_defs_per_setting": int(args.max_total_defs_per_setting),
                "def_select_strategy": args.def_select_strategy,
                "cleanup_flow_work": args.cleanup_flow_work,
                "benchmarks_filter": args.benchmarks,
                "cases_filter": args.cases,
                "iccad_cases_filter": args.iccad_cases,
                "openroad_cases_filter": args.openroad_cases,
                "effective_cases_filter": {
                    "common": common_cases_filter,
                    "ICCAD2015": benchmark_case_filters["ICCAD2015"],
                    "OpenROAD": benchmark_case_filters["OpenROAD"],
                },
                "formulations_filter": args.formulations,
                "modes_filter": selected_modes,
                "algo_eval_mode": args.algo_eval_mode,
                "n_tasks": len(tasks),
                "n_tasks_before_sharding": len(all_tasks),
                "num_shards": int(args.num_shards),
                "shard_id": int(args.shard_id),
                "worker_tag": args.worker_tag,
                "tasks": tasks,
            },
            f,
            indent=2,
        )

    if int(args.num_shards) > 1:
        print(
            f"[INFO] Task sharding enabled: shard={args.shard_id}/{args.num_shards}, "
            f"assigned={len(tasks)}, total_before_sharding={len(all_tasks)}"
        )

    print_plan(tasks, args.preview_limit, args.estimate_per_def_min)
    print(f"Output directory: {output_dir}")
    print(f"Plan saved to: {plan_out}")

    if args.dry_run:
        print("[INFO] dry-run mode, evaluation skipped")
        return

    if args.confirm:
        answer = input("\nProceed with evaluation? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("[INFO] aborted by user")
            return

    if len(tasks) == 0:
        print("[INFO] no task selected")
        return

    timeout_sec = None if args.timeout_sec <= 0 else args.timeout_sec
    reuse_existing_results = str(args.reuse_existing_results).lower() == "true"
    result_rows = []
    run_start = time.time()
    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    workers = max(1, int(args.jobs))
    print(f"[INFO] Evaluation workers: {workers}")
    print(f"[INFO] Evaluation session: {session_tag}")
    if args.progress_style == "plain":
        compact_progress = False
    elif args.progress_style == "compact":
        compact_progress = bool(sys.stdout.isatty())
    else:
        compact_progress = bool(sys.stdout.isatty())

    try:
        if workers == 1:
            for idx, task in enumerate(tasks, 1):
                print(
                    f"\n[RUN {idx}/{len(tasks)}] {task['benchmark']}/{task['case']}/{task['formulation']} "
                    f"mode={task.get('mode','GP')} algo={task['best_algo']} def#{task['def_rank']}"
                )

                row = run_single_task(
                    task,
                    workspace_root,
                    args.platform,
                    args.variant,
                    timeout_sec,
                    session_tag,
                    args.cleanup_flow_work,
                    args.max_retries,
                    args.retry_backoff_sec,
                    reuse_existing_results,
                )
                result_rows.append(row)
                write_partial_results(output_dir, seeds, multi_seed, result_rows, eval_tag)
                print_result_row(row, task["benchmark"])

                elapsed = time.time() - run_start
                avg = elapsed / idx
                remaining = avg * (len(tasks) - idx)
                print(f"[PROGRESS] done={idx}/{len(tasks)} elapsed={elapsed/60:.1f}m avg={avg:.1f}s eta={remaining/60:.1f}m")
        else:
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
            interrupted = False
            live_active = False
            try:
                future_to_task = {
                    ex.submit(
                        run_single_task,
                        t,
                        workspace_root,
                        args.platform,
                        args.variant,
                        timeout_sec,
                        session_tag,
                        args.cleanup_flow_work,
                        args.max_retries,
                        args.retry_backoff_sec,
                        reuse_existing_results,
                    ): t
                    for t in tasks
                }
                if compact_progress:
                    print_log_line(
                        f"[INFO] Submitted {len(tasks)} tasks to {workers} workers (compact live progress enabled)",
                        compact=True,
                    )
                else:
                    for idx, t in enumerate(tasks, 1):
                        print(
                            f"[SUBMIT {idx}/{len(tasks)}] seed={t['seed']} {t['benchmark']}/{t['case']}/{t['formulation']} "
                            f"mode={t.get('mode','GP')} def#{t['def_rank']}"
                        )
                done = 0
                pending = set(future_to_task.keys())
                heartbeat_sec = max(1.0, float(args.heartbeat_sec))
                while pending:
                    completed, pending = concurrent.futures.wait(
                        pending,
                        timeout=heartbeat_sec,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    if not completed:
                        running = snapshot_running_tasks()
                        elapsed = time.time() - run_start
                        if compact_progress:
                            eta_str = "n/a" if done == 0 else f"{(elapsed / done) * (len(tasks) - done) / 60:.1f}m"
                            msg = f"[LIVE] done={done}/{len(tasks)} running={len(running)} elapsed={elapsed/60:.1f}m eta={eta_str}"
                            if running:
                                oldest = min(running.values(), key=lambda x: float(x.get("start", time.time())))
                                age = time.time() - float(oldest.get("start", time.time()))
                                msg += f" | {oldest.get('text')} age={age:.0f}s"
                            print_live_status(msg)
                            live_active = True
                        else:
                            print(
                                f"[HEARTBEAT] done={done}/{len(tasks)} running={len(running)} "
                                f"elapsed={elapsed/60:.1f}m"
                            )
                            if running:
                                sample = sorted(
                                    running.values(),
                                    key=lambda x: float(x.get("start", 0.0)),
                                )[: min(3, len(running))]
                                for x in sample:
                                    age = time.time() - float(x.get("start", time.time()))
                                    print(f"  [RUNNING] {x.get('text')} age={age:.0f}s")
                        continue

                    for fut in completed:
                        task = future_to_task[fut]
                        done += 1
                        try:
                            row = fut.result()
                        except Exception as e:
                            row = dict(task)
                            row.update(
                                {
                                    "seed": int(task.get("seed", seeds[0])),
                                    "eval_ok": False,
                                    "return_code": -1,
                                    "duration_sec": 0.0,
                                    "error": f"internal exception: {e}",
                                    "eval_session": session_tag,
                                }
                            )
                        result_rows.append(row)
                        write_partial_results(output_dir, seeds, multi_seed, result_rows, eval_tag)

                        if compact_progress and live_active:
                            _clear_live_line()
                            sys.stdout.flush()
                            live_active = False

                        print(
                            f"\n[DONE {done}/{len(tasks)}] seed={task['seed']} {task['benchmark']}/{task['case']}/{task['formulation']} "
                            f"mode={task.get('mode','GP')} def#{task['def_rank']} ok={row.get('eval_ok')}"
                        )
                        print_result_row(row, task["benchmark"])

                        elapsed = time.time() - run_start
                        avg = elapsed / done
                        remaining = avg * (len(tasks) - done)
                        print(f"[PROGRESS] done={done}/{len(tasks)} elapsed={elapsed/60:.1f}m avg={avg:.1f}s eta={remaining/60:.1f}m")
                if compact_progress and live_active:
                    _clear_live_line()
                    sys.stdout.flush()
            except KeyboardInterrupt:
                interrupted = True
                if compact_progress and live_active:
                    _clear_live_line()
                    sys.stdout.flush()
                print("\n[WARN] KeyboardInterrupt received. Terminating active worker subprocesses...")
                terminate_active_processes(signal.SIGTERM)
                time.sleep(0.5)
                terminate_active_processes(signal.SIGKILL)
                ex.shutdown(wait=False, cancel_futures=True)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(130)
            finally:
                if not interrupted:
                    ex.shutdown(wait=True, cancel_futures=False)
    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt received. Terminating active worker subprocesses...")
        terminate_active_processes(signal.SIGTERM)
        time.sleep(0.5)
        terminate_active_processes(signal.SIGKILL)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)

    final = {
        "seed": int(seeds[0]),
        "seeds": [int(s) for s in seeds],
        "multi_seed": bool(multi_seed),
        "generated_at": datetime.now().isoformat(),
        "hv_json": hv_json,
        "platform": args.platform,
        "variant": args.variant,
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "worker_tag": args.worker_tag,
        "n_rows": len(result_rows),
        "rows": result_rows,
    }

    if multi_seed:
        json_out = os.path.join(output_dir, f"ppa_eval_seeds_{seed_tag(seeds)}_{eval_tag}.json")
    else:
        json_out = os.path.join(output_dir, f"ppa_eval_seed_{seeds[0]}_{eval_tag}.json")
    with open(json_out, "w") as f:
        json.dump(final, f, indent=2)

    result_df = pd.DataFrame(result_rows)
    if multi_seed:
        csv_out = os.path.join(output_dir, f"ppa_eval_seeds_{seed_tag(seeds)}_{eval_tag}.csv")
    else:
        csv_out = os.path.join(output_dir, f"ppa_eval_seed_{seeds[0]}_{eval_tag}.csv")
    result_df.to_csv(csv_out, index=False)

    if multi_seed and not result_df.empty:
        for seed in seeds:
            sub = result_df[result_df["seed"] == int(seed)].copy()
            if sub.empty:
                continue
            seed_out = os.path.join(output_dir, f"ppa_eval_seed_{int(seed)}_{eval_tag}.csv")
            sub.to_csv(seed_out, index=False)

    ok_count = sum(1 for r in result_rows if r.get("eval_ok"))
    print(f"\nSaved PPA JSON: {json_out}")
    print(f"Saved PPA CSV : {csv_out}")
    print(f"Summary       : ok={ok_count}/{len(result_rows)}")


if __name__ == "__main__":
    main()
