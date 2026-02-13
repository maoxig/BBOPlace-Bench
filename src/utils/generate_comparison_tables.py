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
                # 1. Collect all solution fronts for this case + mode to determine Ref Point
                case_fronts = []
                temp_dict = {} # (form, algo) -> front
                
                valid_case = False 
                
                for form in FORMULATIONS:
                    for algo in ALGOS:
                        # Construct path
                        # Folder name: MO_{BENCHMARK}_{FORM}_{ALGO}_{MODE}
                        # e.g., MO_OPENROAD_MGO_NSGA2_GP
                        # Actual structure: results/case/Full_Name/form(lower)/algo(lower)/seed_timestamp/checkpoint
                        
                        folder_name = f"{prefix}_{form}_{algo}_{mode}"
                        base_result_path = os.path.join(results_dir, case, folder_name, form.lower(), algo.lower())
                        
                        final_path = None
                        if os.path.exists(base_result_path):
                            # Find the latest seed folder
                            subdirs = [f for f in os.listdir(base_result_path) if os.path.isdir(os.path.join(base_result_path, f))]
                            subdirs.sort() # Sorts by seed_X_YYYY... so latest date is last
                            if subdirs:
                                final_path = os.path.join(base_result_path, subdirs[-1])
                        
                        if final_path:
                            front = load_pareto_front(final_path)
                            if front is not None:
                                case_fronts.append(front)
                                temp_dict[(form, algo)] = front
                                valid_case = True
                            else:
                                # Start verbose check only if directory existed but front failed loading
                                # print(f"  Front failed to load for {final_path}")
                                pass
                        else:
                             # print(f"  Path not found: {base_result_path}")
                             pass
                            
                hv_results[case] = {}
                
                if not valid_case:
                    print(f"  No results found for case: {case}")
                    continue
                
                # 2. Compute Unified Reference Point
                if case_fronts:
                    ref_point = compute_reference_point(case_fronts)
                    # print(f"  Case {case}: Ref Point = {ref_point}")
                    
                    # 3. Calculate HV for each
                    for (key, front) in temp_dict.items():
                        hv_val = calculate_hv(front, ref_point)
                        hv_results[case][key] = hv_val
            
            # 4. Generate Table
            f.write(f"## {mode} Mode Results\n\n")
            table_md = generate_markdown_table(benchmark_name, mode, hv_results)
            f.write(table_md)
            f.write("\n\n")
            
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
