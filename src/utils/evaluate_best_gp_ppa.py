import argparse
import json
import os
import re
import subprocess
import sys
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
N_DEF_TOP = 5


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


def load_hv_summary(summary_json, seed):
    with open(summary_json, "r") as f:
        data = json.load(f)

    seed_in_json = data.get("seed")
    if seed_in_json is not None and int(seed_in_json) != int(seed):
        print(f"[WARN] hv json seed={seed_in_json}, requested seed={seed}. Continue anyway.")

    return data


def find_def_candidates(run_path):
    placements_dir = os.path.join(run_path, "placements")
    if not os.path.isdir(placements_dir):
        return []

    selected = []
    for i in range(1, N_DEF_TOP + 1):
        gp_def = os.path.join(placements_dir, f"gp_{i}.def")
        base_def = os.path.join(placements_dir, f"{i}.def")

        if os.path.exists(gp_def):
            selected.append((i, gp_def, True))
        elif os.path.exists(base_def):
            selected.append((i, base_def, False))

    return selected


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


def parse_openroad_output(stdout, work_dir):
    metrics = {}
    metrics_file = os.path.join(work_dir, "metrics.txt")

    if os.path.exists(metrics_file):
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


def evaluate_one_def_subprocess(
    benchmark,
    case_name,
    def_path,
    workspace_root,
    platform,
    variant,
    eval_base_dir,
    timeout_sec,
):
    start = time.time()

    if benchmark == "ICCAD2015":
        cmd = [
            "python",
            os.path.join(workspace_root, "src", "utils", "iccad2015_evaluator.py"),
            "--def_path",
            def_path,
            "--benchmark",
            case_name,
            "--root_dir",
            workspace_root,
        ]
        report_path = f"{def_path}.timing.txt"
    elif benchmark == "OpenROAD":
        work_dir = os.path.join(eval_base_dir, os.path.basename(def_path).replace(".def", ""))
        os.makedirs(work_dir, exist_ok=True)
        cmd = [
            "python",
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
        ]
    else:
        return None, "unsupported benchmark", -1, 0.0

    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout", -9, time.time() - start

    duration = time.time() - start
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if benchmark == "ICCAD2015":
        metrics = parse_iccad_output(stdout, report_path)
    else:
        metrics = parse_openroad_output(stdout, work_dir)

    err_text = ""
    if proc.returncode != 0:
        err_text = (stderr.strip() or stdout.strip())[-800:]

    return metrics, err_text, proc.returncode, duration


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


def build_task_plan(hv_summary, selected_benchmarks, selected_formulations, benchmark_case_filters):
    tasks = []

    for benchmark, bench_cfg in BENCHMARKS.items():
        if selected_benchmarks is not None and benchmark not in selected_benchmarks:
            continue

        bench_data = hv_summary.get("benchmarks", {}).get(benchmark, {})
        gp_data = bench_data.get("modes", {}).get("GP", {})
        gp_cases = gp_data.get("cases", {})
        case_filter = benchmark_case_filters.get(benchmark)

        for case_name in bench_cfg["cases"]:
            if case_filter is not None and case_name not in case_filter:
                continue

            case_data = gp_cases.get(case_name, {})
            for form in FORMULATIONS:
                if selected_formulations is not None and form not in selected_formulations:
                    continue

                form_data = case_data.get(form, {})
                best_algo = form_data.get("best_algo")
                best_hv = form_data.get("best_hv")
                best_run_path = form_data.get("best_run_path")

                if not best_algo or not best_run_path:
                    continue

                def_list = find_def_candidates(best_run_path)
                if not def_list:
                    continue

                for rank, def_path, used_gp in def_list:
                    tasks.append(
                        {
                            "benchmark": benchmark,
                            "case": case_name,
                            "formulation": form,
                            "mode": "GP",
                            "best_algo": best_algo,
                            "best_hv": best_hv,
                            "run_path": best_run_path,
                            "def_rank": rank,
                            "def_path": def_path,
                            "used_gp_def": used_gp,
                        }
                    )

    return tasks


def print_plan(tasks, preview_limit, estimate_per_def_min):
    print("\n=== Evaluation Plan ===")
    print(f"Total DEF tasks: {len(tasks)}")

    combo_set = set((t["benchmark"], t["case"], t["formulation"], t["best_algo"]) for t in tasks)
    print(f"Total benchmark/case/formulation combos: {len(combo_set)}")

    if estimate_per_def_min is not None and len(tasks) > 0:
        est_total_min = estimate_per_def_min * len(tasks)
        print(f"Estimated total time: {est_total_min:.1f} min (per DEF ~ {estimate_per_def_min:.2f} min)")

    print("\nPreview:")
    for idx, t in enumerate(tasks[:preview_limit], 1):
        gp_flag = "gp" if t["used_gp_def"] else "base"
        print(
            f"  [{idx}] {t['benchmark']}/{t['case']}/{t['formulation']} "
            f"best={t['best_algo']} hv={t['best_hv']} def#{t['def_rank']}({gp_flag})"
        )
    if len(tasks) > preview_limit:
        print(f"  ... {len(tasks) - preview_limit} more tasks")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GP PPA for best-HV algorithm per case/formulation (subprocess mode)"
    )
    parser.add_argument("--workspace", type=str, default=".", help="Workspace root")
    parser.add_argument("--output", type=str, default="analysis_reports", help="Output directory")
    parser.add_argument("--seed", type=int, required=True, help="Seed to evaluate")
    parser.add_argument("--hv_json", type=str, default=None, help="Path to hv_summary_seed_<seed>.json")
    parser.add_argument("--platform", type=str, default="nangate45", help="OpenROAD platform")
    parser.add_argument("--variant", type=str, default="xp", help="OpenROAD flow variant")

    parser.add_argument("--benchmarks", type=str, default="all", help="all or comma list: ICCAD2015,OpenROAD")
    parser.add_argument("--cases", type=str, default="all", help="Global case filter (all benchmarks)")
    parser.add_argument("--iccad_cases", type=str, default="all", help="ICCAD2015-only cases, e.g. superblue1")
    parser.add_argument("--openroad_cases", type=str, default="all", help="OpenROAD-only cases, e.g. bp,bp_fe")
    parser.add_argument("--formulations", type=str, default="all", help="all or comma list: MGO,HPO")

    parser.add_argument("--preview_limit", type=int, default=30, help="How many planned tasks to print")
    parser.add_argument("--estimate_per_def_min", type=float, default=None, help="Optional ETA estimate per DEF")
    parser.add_argument("--dry_run", action="store_true", help="Only print/save plan, do not execute evaluation")
    parser.add_argument("--confirm", action="store_true", help="Ask confirmation before execution")
    parser.add_argument("--timeout_sec", type=int, default=0, help="Timeout per DEF subprocess (0 for no timeout)")
    args = parser.parse_args()

    workspace_root = os.path.abspath(args.workspace)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    hv_json = os.path.abspath(args.hv_json) if args.hv_json else os.path.join(output_dir, f"hv_summary_seed_{args.seed}.json")
    if not os.path.exists(hv_json):
        raise FileNotFoundError(f"HV summary json not found: {hv_json}")

    selected_benchmarks = parse_csv_or_all(args.benchmarks, valid_values=list(BENCHMARKS.keys()))
    selected_formulations = parse_csv_or_all(args.formulations, valid_values=FORMULATIONS)
    common_cases_filter, benchmark_case_filters = parse_case_filters(args)

    hv_summary = load_hv_summary(hv_json, args.seed)
    tasks = build_task_plan(hv_summary, selected_benchmarks, selected_formulations, benchmark_case_filters)

    plan_out = os.path.join(output_dir, f"ppa_eval_seed_{args.seed}_gp_best_plan.json")
    with open(plan_out, "w") as f:
        json.dump(
            {
                "seed": int(args.seed),
                "generated_at": datetime.now().isoformat(),
                "hv_json": hv_json,
                "platform": args.platform,
                "variant": args.variant,
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
                "n_tasks": len(tasks),
                "tasks": tasks,
            },
            f,
            indent=2,
        )

    print_plan(tasks, args.preview_limit, args.estimate_per_def_min)
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
    result_rows = []
    run_start = time.time()

    for idx, task in enumerate(tasks, 1):
        eval_base_dir = os.path.join(task["run_path"], "ppa_eval")
        os.makedirs(eval_base_dir, exist_ok=True)

        print(
            f"\n[RUN {idx}/{len(tasks)}] {task['benchmark']}/{task['case']}/{task['formulation']} "
            f"algo={task['best_algo']} def#{task['def_rank']}"
        )

        metrics, err_text, return_code, duration = evaluate_one_def_subprocess(
            benchmark=task["benchmark"],
            case_name=task["case"],
            def_path=task["def_path"],
            workspace_root=workspace_root,
            platform=args.platform,
            variant=args.variant,
            eval_base_dir=eval_base_dir,
            timeout_sec=timeout_sec,
        )

        row = dict(task)
        row.update(
            {
                "seed": int(args.seed),
                "eval_ok": metrics is not None and return_code == 0,
                "return_code": return_code,
                "duration_sec": duration,
                "error": err_text,
            }
        )
        if metrics:
            row.update(metrics)

        result_rows.append(row)

        if row.get("eval_ok"):
            if task["benchmark"] == "ICCAD2015":
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
                    f"runtime_sec={row.get('runtime', row.get('duration_sec')):.2f}"
                )
        else:
            print(f"[RESULT] FAILED rc={row.get('return_code')} err={row.get('error')}")

        elapsed = time.time() - run_start
        avg = elapsed / idx
        remaining = avg * (len(tasks) - idx)
        print(f"[PROGRESS] done={idx}/{len(tasks)} elapsed={elapsed/60:.1f}m avg={avg:.1f}s eta={remaining/60:.1f}m")

    final = {
        "seed": int(args.seed),
        "generated_at": datetime.now().isoformat(),
        "hv_json": hv_json,
        "platform": args.platform,
        "variant": args.variant,
        "n_rows": len(result_rows),
        "rows": result_rows,
    }

    json_out = os.path.join(output_dir, f"ppa_eval_seed_{args.seed}_gp_best.json")
    with open(json_out, "w") as f:
        json.dump(final, f, indent=2)

    csv_out = os.path.join(output_dir, f"ppa_eval_seed_{args.seed}_gp_best.csv")
    pd.DataFrame(result_rows).to_csv(csv_out, index=False)

    ok_count = sum(1 for r in result_rows if r.get("eval_ok"))
    print(f"\nSaved PPA JSON: {json_out}")
    print(f"Saved PPA CSV : {csv_out}")
    print(f"Summary       : ok={ok_count}/{len(result_rows)}")


if __name__ == "__main__":
    main()
