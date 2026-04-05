import argparse
import json
import os
import pickle
import re
from datetime import datetime

import numpy as np
from pymoo.indicators.hv import HV

# Configuration
BENCHMARKS = {
    "OpenROAD": {
        "cases": ["ariane133", "ariane136", "bp", "bp_be", "bp_fe", "swerv_wrapper"],
        "prefix": "MO_OPENROAD",
    },
    "ICCAD2015": {
        "cases": ["superblue1", "superblue3", "superblue4", "superblue5", "superblue7", "superblue10", "superblue16", "superblue18"],
        "prefix": "MO_ICCAD2015",
    },
}

FORMULATIONS = ["MGO", "HPO"]
ALGOS = ["NSGA2", "NSGA3", "MOEAD", "SMSEMOA", "SPEA2"]
MODES = ["GP", "MP"]


def parse_seeds(seed_text):
    items = [x.strip() for x in str(seed_text).split(",") if x.strip()]
    if not items:
        raise ValueError("--seeds is empty")
    return [int(x) for x in items]


def load_pareto_front(result_path):
    """
    Load the pareto front from the checkpoint directory.
    Mimics the logic in analyze_final_results.py.
    """
    checkpoint_path = os.path.join(result_path, "checkpoint")
    if not os.path.isdir(checkpoint_path):
        return None

    final_sol_file = os.path.join(checkpoint_path, "final_solutions.pkl")
    if os.path.exists(final_sol_file):
        try:
            with open(final_sol_file, "rb") as f:
                final_solutions = pickle.load(f)
            if final_solutions:
                y_list = [sol["Y"] for sol in final_solutions]
                return np.array(y_list)
        except Exception as e:
            print(f"Error loading {final_sol_file}: {e}")

    return None


def find_latest_seed_run(base_result_path, seed):
    """
    Find latest successful run directory matching seed_{seed}_* under base_result_path.
    Success criterion: run contains loadable and non-empty final solution front.
    Fallback: if no successful run exists, return latest run regardless of success.
    """
    if not os.path.isdir(base_result_path):
        return None

    pattern = re.compile(rf"^seed_{seed}_.+")
    candidates = []
    for name in os.listdir(base_result_path):
        full = os.path.join(base_result_path, name)
        if os.path.isdir(full) and pattern.match(name):
            candidates.append(name)

    if not candidates:
        return None

    # Timestamp suffix is lexical sortable with format %Y-%m-%d_%H-%M-%S.
    candidates.sort()

    successful = []
    for name in candidates:
        run_path = os.path.join(base_result_path, name)
        front = load_pareto_front(run_path)
        if front is not None and len(front) > 0:
            successful.append(name)

    if successful:
        return os.path.join(base_result_path, successful[-1])

    return os.path.join(base_result_path, candidates[-1])


def compute_reference_point(all_fronts):
    if not all_fronts:
        return None

    all_points = np.vstack(all_fronts)
    if all_points.size == 0:
        return None

    valid_mask = np.all(all_points < 1e14, axis=1)
    if np.any(valid_mask):
        valid_points = all_points[valid_mask]
        nadir_point = np.max(valid_points, axis=0)
    else:
        nadir_point = np.max(all_points, axis=0)

    ref_point = nadir_point * 1.1 + 1e-6
    return ref_point


def calculate_hv(front, ref_point):
    if front is None or len(front) == 0 or ref_point is None:
        return None

    try:
        ind = HV(ref_point=ref_point)
        return float(ind(front))
    except Exception as e:
        print(f"HV Calculation Error: {e}")
        return None


def get_hv(hv_data, case, form, algo):
    return hv_data.get(case, {}).get(form, {}).get("algorithms", {}).get(algo, {}).get("hv")


def get_hv_stats(hv_data, case, form, algo):
    return hv_data.get(case, {}).get(form, {}).get("algorithms", {}).get(algo, {}).get("hv_stats")


def generate_markdown_table(benchmark_name, mode, hv_data):
    md_output = []

    header_row = "| Case | " + " | ".join([f"{form}-{algo}" for form in FORMULATIONS for algo in ALGOS]) + " |"
    separator_row = "| :--- | " + " | ".join([":---:" for _ in range(len(FORMULATIONS) * len(ALGOS))]) + " |"

    md_output.append(header_row)
    md_output.append(separator_row)

    cases = BENCHMARKS[benchmark_name]["cases"]
    for case in cases:
        row_str = f"| {case} |"
        for form in FORMULATIONS:
            for algo in ALGOS:
                val = get_hv(hv_data, case, form, algo)
                if isinstance(val, float):
                    row_str += f" {val:.4e} |"
                else:
                    row_str += " - |"
        md_output.append(row_str)

    return "\n".join(md_output)


def generate_markdown_table_stats(benchmark_name, mode, hv_data):
    md_output = []

    header_row = "| Case | " + " | ".join([f"{form}-{algo}" for form in FORMULATIONS for algo in ALGOS]) + " |"
    separator_row = "| :--- | " + " | ".join([":---:" for _ in range(len(FORMULATIONS) * len(ALGOS))]) + " |"

    md_output.append(header_row)
    md_output.append(separator_row)

    cases = BENCHMARKS[benchmark_name]["cases"]
    for case in cases:
        row_str = f"| {case} |"
        for form in FORMULATIONS:
            for algo in ALGOS:
                stats = get_hv_stats(hv_data, case, form, algo)
                if isinstance(stats, dict) and stats.get("mean") is not None:
                    mean_v = stats.get("mean")
                    std_v = stats.get("std", 0.0)
                    n_v = stats.get("n", 0)
                    row_str += f" {mean_v:.4e} ± {std_v:.2e} (n={n_v}) |"
                else:
                    row_str += " - |"
        md_output.append(row_str)

    return "\n".join(md_output)


def generate_latex_table(benchmark_name, mode, hv_data):
    cases = BENCHMARKS[benchmark_name]["cases"]

    latex_output = []
    latex_output.append(r"\begin{table*}[t]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Hypervolume Comparison on " + benchmark_name + " (" + mode + r" Mode)}")
    latex_output.append(r"\label{tab:" + benchmark_name.lower() + r"_" + mode.lower() + r"}")
    latex_output.append(r"\resizebox{\textwidth}{!}{")

    col_def = "l" + "c" * (len(FORMULATIONS) * len(ALGOS))
    latex_output.append(r"\begin{tabular}{" + col_def + r"}")
    latex_output.append(r"\toprule")

    header_1 = r"\multirow{2}{*}{Benchmarks}"
    for form in FORMULATIONS:
        header_1 += r" & \multicolumn{" + str(len(ALGOS)) + r"}{c}{" + form + r"}"
    header_1 += r" \\" 
    latex_output.append(header_1)

    cmid_indices = []
    current_idx = 2
    for _ in FORMULATIONS:
        cmid_indices.append(f"{current_idx}-{current_idx + len(ALGOS) - 1}")
        current_idx += len(ALGOS)
    cmid_str = " ".join([r"\cmidrule(lr){" + s + r"}" for s in cmid_indices])
    latex_output.append(cmid_str)

    header_2 = ""
    for _ in FORMULATIONS:
        for algo in ALGOS:
            header_2 += r" & " + algo
    header_2 += r" \\" 
    latex_output.append(header_2)
    latex_output.append(r"\midrule")

    for case in cases:
        row_str = case.replace("_", r"\_")

        for form in FORMULATIONS:
            form_vals = []
            for algo in ALGOS:
                val = get_hv(hv_data, case, form, algo)
                if isinstance(val, float):
                    form_vals.append(val)
            max_val = max(form_vals) if form_vals else None

            if max_val is not None and max_val > 0:
                sci_str_max = "{:.2e}".format(max_val)
                _, max_exponent_str = sci_str_max.split("e")
                form_exponent = int(max_exponent_str)
            else:
                form_exponent = 0

            for algo in ALGOS:
                val = get_hv(hv_data, case, form, algo)
                if isinstance(val, float):
                    base = val / (10 ** form_exponent)
                    tex_val = r"${:.2f} \times 10^{{{}}}$".format(base, form_exponent)
                    # Only underline best. No second-best formatting.
                    if max_val is not None and val == max_val:
                        tex_val = r"\underline{" + tex_val + r"}"
                    row_str += " & " + tex_val
                else:
                    row_str += " & -"

        row_str += r" \\" 
        latex_output.append(row_str)

    latex_output.append(r"\bottomrule")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"}")
    latex_output.append(r"\end{table*}")

    return "\n".join(latex_output)


def generate_latex_table_stats(benchmark_name, mode, hv_data):
    cases = BENCHMARKS[benchmark_name]["cases"]

    latex_output = []
    latex_output.append(r"\begin{table*}[t]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Hypervolume Mean$\\pm$Std on " + benchmark_name + " (" + mode + r" Mode)}")
    latex_output.append(r"\label{tab:" + benchmark_name.lower() + r"_" + mode.lower() + r"_mean_std}")
    latex_output.append(r"\resizebox{\textwidth}{!}{")

    col_def = "l" + "c" * (len(FORMULATIONS) * len(ALGOS))
    latex_output.append(r"\begin{tabular}{" + col_def + r"}")
    latex_output.append(r"\toprule")

    header_1 = r"\multirow{2}{*}{Benchmarks}"
    for form in FORMULATIONS:
        header_1 += r" & \multicolumn{" + str(len(ALGOS)) + r"}{c}{" + form + r"}"
    header_1 += r" \\" 
    latex_output.append(header_1)

    cmid_indices = []
    current_idx = 2
    for _ in FORMULATIONS:
        cmid_indices.append(f"{current_idx}-{current_idx + len(ALGOS) - 1}")
        current_idx += len(ALGOS)
    cmid_str = " ".join([r"\cmidrule(lr){" + s + r"}" for s in cmid_indices])
    latex_output.append(cmid_str)

    header_2 = ""
    for _ in FORMULATIONS:
        for algo in ALGOS:
            header_2 += r" & " + algo
    header_2 += r" \\" 
    latex_output.append(header_2)
    latex_output.append(r"\midrule")

    for case in cases:
        row_str = case.replace("_", r"\_")

        # underline best mean per formulation
        for form in FORMULATIONS:
            means = []
            for algo in ALGOS:
                st = get_hv_stats(hv_data, case, form, algo)
                if isinstance(st, dict) and st.get("mean") is not None:
                    means.append(st["mean"])
            best_mean = max(means) if means else None

            for algo in ALGOS:
                st = get_hv_stats(hv_data, case, form, algo)
                if isinstance(st, dict) and st.get("mean") is not None:
                    mean_v = st.get("mean")
                    std_v = st.get("std", 0.0)
                    cell = r"${:.2e} \pm {:.1e}$".format(mean_v, std_v)
                    if best_mean is not None and mean_v == best_mean:
                        cell = r"\underline{" + cell + r"}"
                    row_str += " & " + cell
                else:
                    row_str += " & -"

        row_str += r" \\" 
        latex_output.append(row_str)

    latex_output.append(r"\bottomrule")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"}")
    latex_output.append(r"\end{table*}")

    return "\n".join(latex_output)


def analyze_benchmark_multi_seed(benchmark_name, output_dir, workspace_root, seeds):
    cases = BENCHMARKS[benchmark_name]["cases"]
    prefix = BENCHMARKS[benchmark_name]["prefix"]
    results_dir = os.path.join(workspace_root, "results")

    markdown_dir = os.path.join(output_dir, "markdown")
    latex_dir = os.path.join(output_dir, "latex")
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(markdown_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    seeds_tag = "_".join([str(s) for s in seeds])
    analysis_file = os.path.join(markdown_dir, f"{benchmark_name}_seeds_{seeds_tag}_Analysis.md")
    bench_summary = {
        "benchmark": benchmark_name,
        "seeds": [int(s) for s in seeds],
        "modes": {},
    }

    with open(analysis_file, "w") as f:
        f.write(f"# Analysis for {benchmark_name} (seeds={seeds})\n\n")

        for mode in MODES:
            print(f"Processing {benchmark_name} - {mode} - seeds {seeds}...")
            hv_results = {}

            for case in cases:
                hv_results[case] = {}

                for form in FORMULATIONS:
                    run_by_algo_seed = {}
                    front_by_algo_seed = {}
                    fronts_all_seeds = []

                    for algo in ALGOS:
                        run_by_algo_seed[algo] = {}
                        front_by_algo_seed[algo] = {}

                        for seed in seeds:
                            folder_name = f"{prefix}_{form}_{algo}_{mode}"
                            base_result_path = os.path.join(results_dir, case, folder_name, form.lower(), algo.lower())
                            run_path = find_latest_seed_run(base_result_path, seed)
                            run_by_algo_seed[algo][str(seed)] = run_path

                            front = None
                            if run_path:
                                front = load_pareto_front(run_path)
                            front_by_algo_seed[algo][str(seed)] = front

                            if front is not None and len(front) > 0:
                                fronts_all_seeds.append(front)

                    # Unified reference point across (algo x seed) for this setting.
                    ref_point = compute_reference_point(fronts_all_seeds)

                    form_result = {
                        "algorithms": {},
                        "reference_point": ref_point.tolist() if ref_point is not None else None,
                    }

                    for algo in ALGOS:
                        hv_by_seed = {}
                        vals = []

                        for seed in seeds:
                            seed_key = str(seed)
                            front = front_by_algo_seed[algo].get(seed_key)
                            hv_val = calculate_hv(front, ref_point) if front is not None else None
                            hv_by_seed[seed_key] = hv_val
                            if hv_val is not None:
                                vals.append(hv_val)

                        stats = None
                        if vals:
                            stats = {
                                "mean": float(np.mean(vals)),
                                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                "n": int(len(vals)),
                            }

                        form_result["algorithms"][algo] = {
                            "hv_by_seed": hv_by_seed,
                            "hv_stats": stats,
                            "run_path_by_seed": run_by_algo_seed[algo],
                        }

                    # best algo by mean hv
                    best_algo = None
                    best_mean = None
                    for algo in ALGOS:
                        st = form_result["algorithms"][algo].get("hv_stats")
                        if st is not None:
                            m = st["mean"]
                            if best_mean is None or m > best_mean:
                                best_mean = m
                                best_algo = algo
                    form_result["best_algo_by_mean"] = best_algo
                    form_result["best_hv_mean"] = best_mean

                    hv_results[case][form] = form_result

            f.write(f"## {mode} Mode Results (Mean ± Std)\n\n")
            table_md = generate_markdown_table_stats(benchmark_name, mode, hv_results)
            f.write(table_md)
            f.write("\n\n")

            latex_file = os.path.join(latex_dir, f"{benchmark_name}_{mode}_seeds_{seeds_tag}_Table.tex")
            table_latex = generate_latex_table_stats(benchmark_name, mode, hv_results)
            with open(latex_file, "w") as lf:
                lf.write(table_latex)
            print(f"  LaTeX table saved to {latex_file}")

            mode_json_file = os.path.join(json_dir, f"{benchmark_name}_{mode}_seeds_{seeds_tag}_hv.json")
            with open(mode_json_file, "w") as jf:
                json.dump(
                    {
                        "benchmark": benchmark_name,
                        "mode": mode,
                        "seeds": [int(s) for s in seeds],
                        "generated_at": datetime.now().isoformat(),
                        "cases": hv_results,
                    },
                    jf,
                    indent=2,
                )
            print(f"  JSON saved to {mode_json_file}")

            bench_summary["modes"][mode] = {"cases": hv_results}

    print(f"Analysis saved to {analysis_file}")
    return bench_summary


def analyze_benchmark(benchmark_name, output_dir, workspace_root, seed):
    cases = BENCHMARKS[benchmark_name]["cases"]
    prefix = BENCHMARKS[benchmark_name]["prefix"]
    results_dir = os.path.join(workspace_root, "results")

    markdown_dir = os.path.join(output_dir, "markdown")
    latex_dir = os.path.join(output_dir, "latex")
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(markdown_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    analysis_file = os.path.join(markdown_dir, f"{benchmark_name}_seed{seed}_Analysis.md")
    bench_summary = {
        "benchmark": benchmark_name,
        "seed": int(seed),
        "modes": {},
    }

    with open(analysis_file, "w") as f:
        f.write(f"# Analysis for {benchmark_name} (seed={seed})\n\n")

        for mode in MODES:
            print(f"Processing {benchmark_name} - {mode} - seed {seed}...")

            hv_results = {}

            for case in cases:
                hv_results[case] = {}

                for form in FORMULATIONS:
                    fronts = []
                    fronts_by_algo = {}
                    run_by_algo = {}

                    for algo in ALGOS:
                        folder_name = f"{prefix}_{form}_{algo}_{mode}"
                        base_result_path = os.path.join(results_dir, case, folder_name, form.lower(), algo.lower())

                        run_path = find_latest_seed_run(base_result_path, seed)
                        run_by_algo[algo] = run_path

                        if run_path:
                            front = load_pareto_front(run_path)
                            if front is not None:
                                fronts.append(front)
                                fronts_by_algo[algo] = front

                    ref_point = compute_reference_point(fronts)

                    form_result = {
                        "algorithms": {},
                        "best_algo": None,
                        "best_hv": None,
                        "best_run_path": None,
                        "reference_point": ref_point.tolist() if ref_point is not None else None,
                    }

                    best_algo = None
                    best_hv = None

                    for algo in ALGOS:
                        front = fronts_by_algo.get(algo)
                        hv_val = calculate_hv(front, ref_point) if front is not None else None

                        form_result["algorithms"][algo] = {
                            "hv": hv_val,
                            "run_path": run_by_algo.get(algo),
                        }

                        if hv_val is not None and (best_hv is None or hv_val > best_hv):
                            best_hv = hv_val
                            best_algo = algo

                    if best_algo is not None:
                        form_result["best_algo"] = best_algo
                        form_result["best_hv"] = best_hv
                        form_result["best_run_path"] = run_by_algo.get(best_algo)

                    hv_results[case][form] = form_result

            f.write(f"## {mode} Mode Results\n\n")
            table_md = generate_markdown_table(benchmark_name, mode, hv_results)
            f.write(table_md)
            f.write("\n\n")

            latex_file = os.path.join(latex_dir, f"{benchmark_name}_{mode}_seed{seed}_Table.tex")
            table_latex = generate_latex_table(benchmark_name, mode, hv_results)
            with open(latex_file, "w") as lf:
                lf.write(table_latex)
            print(f"  LaTeX table saved to {latex_file}")

            mode_json_file = os.path.join(json_dir, f"{benchmark_name}_{mode}_seed{seed}_hv.json")
            with open(mode_json_file, "w") as jf:
                json.dump(
                    {
                        "benchmark": benchmark_name,
                        "mode": mode,
                        "seed": int(seed),
                        "generated_at": datetime.now().isoformat(),
                        "cases": hv_results,
                    },
                    jf,
                    indent=2,
                )
            print(f"  JSON saved to {mode_json_file}")

            bench_summary["modes"][mode] = {"cases": hv_results}

    print(f"Analysis saved to {analysis_file}")
    return bench_summary


def main():
    parser = argparse.ArgumentParser(description="Generate seed-specific HV tables and JSON summaries")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace root directory")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("results", "analysis_reports", "hv"),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, required=True, help="Seed id for this analysis round")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for unified-ref multi-seed analysis, e.g. 1,2,3",
    )
    args = parser.parse_args()

    workspace_root = os.path.abspath(args.workspace)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    seeds = [int(args.seed)] if args.seeds is None else parse_seeds(args.seeds)
    multi_seed = len(seeds) > 1

    print(f"Starting analysis in {workspace_root}, output to {output_dir}, seeds={seeds}")

    summary = {
        "seed": int(args.seed),
        "seeds": [int(s) for s in seeds],
        "multi_seed": bool(multi_seed),
        "generated_at": datetime.now().isoformat(),
        "benchmarks": {},
    }

    for benchmark in BENCHMARKS:
        if multi_seed:
            summary["benchmarks"][benchmark] = analyze_benchmark_multi_seed(benchmark, output_dir, workspace_root, seeds)
        else:
            summary["benchmarks"][benchmark] = analyze_benchmark(benchmark, output_dir, workspace_root, args.seed)

    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    if multi_seed:
        seeds_tag = "_".join([str(s) for s in seeds])
        summary_json = os.path.join(json_dir, f"hv_summary_seeds_{seeds_tag}.json")
    else:
        summary_json = os.path.join(json_dir, f"hv_summary_seed_{args.seed}.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Combined JSON summary saved to {summary_json}")


if __name__ == "__main__":
    main()
