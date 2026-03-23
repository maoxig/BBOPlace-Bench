import argparse
import json
import os
from datetime import datetime

import pandas as pd

from src.utils.iccad2015_evaluator import evaluate_iccad2015_timing
from src.utils.openroad_evaluator import run_evaluation

BENCHMARKS = {
    "OpenROAD": {
        "cases": ["ariane133", "ariane136", "bp", "bp_be", "bp_fe", "swerv_wrapper"],
    },
    "ICCAD2015": {
        "cases": ["superblue1", "superblue3", "superblue4", "superblue5", "superblue7", "superblue10", "superblue16", "superblue18"],
    },
}

FORMULATIONS = ["MGO", "HPO"]
N_DEF_TOP = 5


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


def load_hv_summary(summary_json, seed):
    with open(summary_json, "r") as f:
        data = json.load(f)

    seed_in_json = data.get("seed")
    if seed_in_json is not None and int(seed_in_json) != int(seed):
        print(f"[WARN] hv json seed={seed_in_json}, requested seed={seed}. Continue anyway.")

    return data


def evaluate_one_def(benchmark, case_name, def_path, workspace_root, platform, variant, eval_base_dir):
    if benchmark == "ICCAD2015":
        return evaluate_iccad2015_timing(
            def_path=def_path,
            benchmark_name=case_name,
            root_dir=workspace_root,
            verbose=False,
        )

    if benchmark == "OpenROAD":
        work_dir = os.path.join(eval_base_dir, os.path.basename(def_path).replace(".def", ""))
        os.makedirs(work_dir, exist_ok=True)
        return run_evaluation(
            def_path=def_path,
            design=case_name,
            platform=platform,
            variant=variant,
            work_dir=work_dir,
            root_dir=workspace_root,
        )

    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate GP PPA for best-HV algorithm per case/formulation")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace root")
    parser.add_argument("--output", type=str, default="analysis_reports", help="Output directory")
    parser.add_argument("--seed", type=int, required=True, help="Seed to evaluate")
    parser.add_argument("--hv_json", type=str, default=None, help="Path to hv_summary_seed_<seed>.json")
    parser.add_argument("--platform", type=str, default="nangate45", help="OpenROAD platform")
    parser.add_argument("--variant", type=str, default="eval_xp", help="OpenROAD flow variant")
    args = parser.parse_args()

    workspace_root = os.path.abspath(args.workspace)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    hv_json = args.hv_json
    if hv_json is None:
        hv_json = os.path.join(output_dir, f"hv_summary_seed_{args.seed}.json")
    hv_json = os.path.abspath(hv_json)

    if not os.path.exists(hv_json):
        raise FileNotFoundError(f"HV summary json not found: {hv_json}")

    hv_summary = load_hv_summary(hv_json, args.seed)

    result_rows = []

    for benchmark, bench_cfg in BENCHMARKS.items():
        bench_data = hv_summary.get("benchmarks", {}).get(benchmark, {})
        gp_data = bench_data.get("modes", {}).get("GP", {})
        gp_cases = gp_data.get("cases", {})

        for case_name in bench_cfg["cases"]:
            case_data = gp_cases.get(case_name, {})

            for form in FORMULATIONS:
                form_data = case_data.get(form, {})
                best_algo = form_data.get("best_algo")
                best_hv = form_data.get("best_hv")
                best_run_path = form_data.get("best_run_path")

                if not best_algo or not best_run_path:
                    print(f"[SKIP] {benchmark}/{case_name}/{form}: no best algo or run path in HV summary")
                    continue

                def_list = find_def_candidates(best_run_path)
                if not def_list:
                    print(f"[SKIP] {benchmark}/{case_name}/{form}: no def found in {best_run_path}")
                    continue

                eval_base_dir = os.path.join(best_run_path, "ppa_eval")
                os.makedirs(eval_base_dir, exist_ok=True)

                print(f"[INFO] Evaluating {benchmark}/{case_name}/{form} best={best_algo}, defs={len(def_list)}")

                for rank, def_path, used_gp in def_list:
                    metrics = evaluate_one_def(
                        benchmark=benchmark,
                        case_name=case_name,
                        def_path=def_path,
                        workspace_root=workspace_root,
                        platform=args.platform,
                        variant=args.variant,
                        eval_base_dir=eval_base_dir,
                    )

                    row = {
                        "seed": int(args.seed),
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
                        "eval_ok": metrics is not None,
                    }

                    if metrics:
                        for k, v in metrics.items():
                            row[k] = v

                    result_rows.append(row)

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

    print(f"Saved PPA JSON: {json_out}")
    print(f"Saved PPA CSV : {csv_out}")


if __name__ == "__main__":
    main()
