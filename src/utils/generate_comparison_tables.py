import os
import pickle
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
import argparse

# Configuration
BENCHMARKS = {
    "OpenROAD": {
        "cases": ["ariane133", "ariane136", "bp", "bp_be", "bp_fe", "swerv_wrapper"],
        "prefix": "MO_OPENROAD"
    },
    "ICCAD2015": {
        "cases": ["superblue1", "superblue3", "superblue4", "superblue5", "superblue7", "superblue10", "superblue16", "superblue18"],
        "prefix": "MO_ICCAD2015"
    }
}

FORMULATIONS = ["MGO", "HPO"]
ALGOS = ["NSGA2", "NSGA3", "MOEAD", "SMSEMOA", "SPEA2"]
MODES = ["GP", "MP"]

def load_pareto_front(result_path):
    """
    Load the pareto front from the checkpoint directory.
    Mimics the logic in analyze_final_results.py.
    """
    checkpoint_path = os.path.join(result_path, "checkpoint")
    
    if not os.path.isdir(checkpoint_path):
        return None

    # 1. Try loading final_solutions.pkl
    final_sol_file = os.path.join(checkpoint_path, "final_solutions.pkl")
    if os.path.exists(final_sol_file):
        try:
            with open(final_sol_file, 'rb') as f:
                final_solutions = pickle.load(f)
            if final_solutions:
                Y_list = [sol['Y'] for sol in final_solutions]
                return np.array(Y_list)
        except Exception as e:
            print(f"Error loading {final_sol_file}: {e}")
            
            
    return None

def compute_reference_point(all_fronts):
    """
    Compute a unified reference point (Nadir Point * 1.1) from a list of pareto fronts.
    """
    if not all_fronts:
        return None
    
    # Concatenate all points
    all_points = np.vstack(all_fronts)
    
    if all_points.size == 0:
        return None
        

    # Calculate Nadir Point (max in each objective)
    # Check for INF (1e16) and filter it out before calculating Nadir
    # If all points are INF in a dimension, then Reference Point will be INF (HV = 0 makes sense)
    
    # Simple logic: If any point is >= 1e15 (near INF), we ignore it for reference point calculation
    # to avoid skewing the HV for other valid solutions.
    # HOWEVER, if an algorithm produces INF solutions, they are dominated by valid ones.
    # But if we include INF in ref point calculation, the volume becomes huge.
    # Usually in optimization benchmarks, we might use a fixed reference point or the worst valid point.
    
    # Strategy: Filter out points that are "too large" (failed runs/constraints violated significantly)
    # assuming they are outliers, unless ALL points are large.
    
    valid_mask = np.all(all_points < 1e14, axis=1)
    if np.any(valid_mask):
        valid_points = all_points[valid_mask]
        nadir_point = np.max(valid_points, axis=0)
    else:
        # Fallback if everything is huge
        nadir_point = np.max(all_points, axis=0)
    
    # Apply a small margin (e.g., 1.1x) to ensure extreme points calculate correctly
    # Use a small epsilon for stability if values are 0 (though unlikely for cost metrics)
    ref_point = nadir_point * 1.1 + 1e-6
    
    return ref_point

def calculate_hv(front, ref_point):
    if front is None or len(front) == 0:
        return 0.0
    
    try:
        ind = HV(ref_point=ref_point)
        return ind(front)
    except Exception as e:
        print(f"HV Calculation Error: {e}")
        return 0.0

def generate_markdown_table(benchmark_name, mode, hv_data):
    """
    Generate a Markdown table string.
    hv_data format: { case: { (formulation, algo): hv_value } }
    """
    # Header construction
    headers = [f"| **{benchmark_name} ({mode})**"]
    separators = ["| :--- "]
    
    # 1st Header Row: Formulations
    row1 = f"| **Case** |"
    for form in FORMULATIONS:
        row1 += f" **{form}** |" + " |" * (len(ALGOS) - 1)
        
    # 2nd Header Row: Algorithms
    row2 = f"| |"
    for form in FORMULATIONS:
        for algo in ALGOS:
            row2 += f" {algo} |"
            separators.append(":---:|")
            
    # Markdown table header logic needs standard format
    # Because Markdown tables don't support row spanning natively in all renderers,
    # we simulate it or just use a clear header structure. 
    # Standard Markdown table doesn't support colspan nicely.
    # However, for academic/report view, we can write:
    # | Case | MGO-NSGA2 | MGO-NSGA3 | ... | HPO-NSGA2 | ... |
    # But user asked for spanning conceptually.
    # We will use HTML style for spanning if needed, or just composite headers.
    # Let's stick to composite headers for compatibility: "MGO<br>NSGA2".
    
    md_output = []
    
    # Using composite headers for better compatibility
    header_row = "| Case | " + " | ".join([f"{form}-{algo}" for form in FORMULATIONS for algo in ALGOS]) + " |"
    separator_row = "| :--- | " + " | ".join([":---:" for _ in range(len(FORMULATIONS) * len(ALGOS))]) + " |"
    
    md_output.append(header_row)
    md_output.append(separator_row)
    
    cases = BENCHMARKS[benchmark_name]["cases"]
    
    for case in cases:
        row_str = f"| {case} |"
        for form in FORMULATIONS:
            for algo in ALGOS:
                val = hv_data.get(case, {}).get((form, algo), "N/A")
                if isinstance(val, float):
                    row_str += f" {val:.4e} |"
                else:
                    row_str += " - |"
        md_output.append(row_str)
        
    return "\n".join(md_output)

def generate_latex_table(benchmark_name, mode, hv_data):
    """
    Generate a LaTeX table string for academic papers.
    hv_data format: { case: { (formulation, algo): hv_value } }
    """
    cases = BENCHMARKS[benchmark_name]["cases"]


    # Header Construction
    latex_output = []
    latex_output.append(r"\begin{table*}[t]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Hypervolume Comparison on " + benchmark_name + " (" + mode + r" Mode)}")
    latex_output.append(r"\label{tab:" + benchmark_name.lower() + r"_" + mode.lower() + r"}")
    latex_output.append(r"\resizebox{\textwidth}{!}{")
    
    # Column definition: Case (l) + 2 * 5 Algorithms (c)
    col_def = "l" + "c" * (len(FORMULATIONS) * len(ALGOS))
    latex_output.append(r"\begin{tabular}{" + col_def + r"}")
    latex_output.append(r"\toprule")
    
    # Header Row 1: Formulation Spanning
    # e.g., \multirow{2}{*}{Benchmarks} & \multicolumn{5}{c}{MGO} & \multicolumn{5}{c}{HPO} \\
    header_1 = r"\multirow{2}{*}{Benchmarks}"
    for form in FORMULATIONS:
         header_1 += r" & \multicolumn{" + str(len(ALGOS)) + r"}{c}{\textbf{" + form + r"}}"
    header_1 += r" \\"
    latex_output.append(header_1)
    
    # Header Row 2: CMidrules
    # \cmidrule(lr){2-6} \cmidrule(lr){7-11}
    cmid_indices = []
    current_idx = 2
    for _ in FORMULATIONS:
        cmid_indices.append(f"{current_idx}-{current_idx + len(ALGOS) - 1}")
        current_idx += len(ALGOS)
    
    cmid_str = " ".join([r"\cmidrule(lr){" + s + r"}" for s in cmid_indices])
    latex_output.append(cmid_str)

    # Header Row 3: Algorithm Names
    header_2 = ""
    for _ in FORMULATIONS:
        for algo in ALGOS:
            header_2 += r" & " + algo
    header_2 += r" \\"
    latex_output.append(header_2)
    latex_output.append(r"\midrule")
    
    # Data Rows
    for case in cases:
        row_str = case.replace("_", r"\_")
        
        for form in FORMULATIONS:
            # 1. Determine local max and exponent per Formulation (MGO / HPO independently)
            form_values = []
            for algo in ALGOS:
                val = hv_data.get(case, {}).get((form, algo), None)
                if val is not None and isinstance(val, (int, float)):
                    form_values.append(val)
            
            max_val = max(form_values) if form_values else 0.0
            
            if max_val > 0:
                sci_str_max = "{:.2e}".format(max_val)
                _, max_exponent_str = sci_str_max.split("e")
                form_exponent = int(max_exponent_str)
            else:
                form_exponent = 0

            # 2. Format values for this formulation
            for algo in ALGOS:
                val = hv_data.get(case, {}).get((form, algo), None)
                
                if val is not None and isinstance(val, (int, float)):
                    # Normalize to form_exponent
                    base = val / (10 ** form_exponent)
                    
                    tex_val = r"${:.2f} \times 10^{{{}}}$".format(base, form_exponent)
                    
                    if val == max_val and max_val > 0:
                         tex_val = r"\underline{" + tex_val + r"}"
                         
                    row_str += " & " + tex_val
                else:
                    row_str += " & -"
                    
        row_str += r" \\"
        latex_output.append(row_str)
        
    latex_output.append(r"\bottomrule")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"}") # End resizebox
    latex_output.append(r"\end{table*}")
    
    return "\n".join(latex_output)

def analyze_benchmark(benchmark_name, output_dir, workspace_root):
    cases = BENCHMARKS[benchmark_name]["cases"]
    prefix = BENCHMARKS[benchmark_name]["prefix"]
    results_dir = os.path.join(workspace_root, "results")
    
    analysis_file = os.path.join(output_dir, f"{benchmark_name}_Analysis.md")
    
    with open(analysis_file, "w") as f:
        f.write(f"# Analysis for {benchmark_name}\n\n")
        
        for mode in MODES:
            print(f"Processing {benchmark_name} - {mode}...")
            
            # Store results for this mode
            # Structure: hv_results[case][(formulation, algo)] = hv_value
            hv_results = {}
            
            for case in cases:
                hv_results[case] = {}
                
                # We need to compute Reference Point independently for each Formulation
                # because MGO and HPO have different objectives (metrics).
                # Setting = (Benchmark Case, Formulation)
                
                for form in FORMULATIONS:
                    # Collect all fronts for this specific Setting (Formulation)
                    # to compute a unified Reference Point for the 5 Algos within this Formulation.
                    form_fronts = []
                    temp_dict = {} # algo -> front
                    
                    for algo in ALGOS:
                        folder_name = f"{prefix}_{form}_{algo}_{mode}"
                        base_result_path = os.path.join(results_dir, case, folder_name, form.lower(), algo.lower())
                        
                        final_path = None
                        if os.path.exists(base_result_path):
                            subdirs = [f for f in os.listdir(base_result_path) if os.path.isdir(os.path.join(base_result_path, f))]
                            subdirs.sort()
                            if subdirs:
                                final_path = os.path.join(base_result_path, subdirs[-1])
                        
                        if final_path:
                            front = load_pareto_front(final_path)
                            if front is not None:
                                form_fronts.append(front)
                                temp_dict[algo] = front
                    
                    if not form_fronts:
                         # No results for this formulation in this case
                         continue
                         
                    # Compute Ref Point for this Formulation
                    ref_point = compute_reference_point(form_fronts)
                    
                    # Calculate HV for each Algo in this Formulation using the formulation-specific Ref Point
                    for algo, front in temp_dict.items():
                         hv_val = calculate_hv(front, ref_point)
                         hv_results[case][(form, algo)] = hv_val
            

            # 4. Generate Markdown Table
            f.write(f"## {mode} Mode Results\n\n")
            table_md = generate_markdown_table(benchmark_name, mode, hv_results)
            f.write(table_md)
            f.write("\n\n")
            
            # 5. Generate LaTeX Table
            latex_file = os.path.join(output_dir, f"{benchmark_name}_{mode}_Table.tex")
            table_latex = generate_latex_table(benchmark_name, mode, hv_results)
            with open(latex_file, "w") as lf:
                lf.write(table_latex)
            print(f"  LaTeX table saved to {latex_file}")
            
    print(f"Analysis saved to {analysis_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Academic HV Tables")
    parser.add_argument("--workspace", type=str, default=".", help="Workspace root directory")
    parser.add_argument("--output", type=str, default="analysis_reports", help="Output directory")
    args = parser.parse_args()
    
    # Ensure absolute paths
    workspace_root = os.path.abspath(args.workspace)
    output_dir = os.path.abspath(args.output)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Starting analysis in {workspace_root}, output to {output_dir}")
    
    for benchmark in BENCHMARKS:
        analyze_benchmark(benchmark, output_dir, workspace_root)

if __name__ == "__main__":
    main()
