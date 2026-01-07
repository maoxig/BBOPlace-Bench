# filepath: /home/xp/project/BBOPlace-Bench/src/utils/analyze_results.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def load_pareto_front(result_path):
    """
    Load the pareto front from the checkpoint directory.
    Prioritizes elite_pool.pkl (new format), falls back to pareto_front.npy.
    """
    checkpoint_path = os.path.join(result_path, "checkpoint")
    
    # Try loading elite pool first (new format)
    elite_file = os.path.join(checkpoint_path, "elite_pool.pkl")
    if os.path.exists(elite_file):
        print(f"Loading elite pool from {elite_file}")
        with open(elite_file, 'rb') as f:
            elite_pool = pickle.load(f)
        
        if elite_pool:
            Y_list = [sol['Y'] for sol in elite_pool]
            return np.array(Y_list)
        else:
            print("Warning: Elite pool is empty.")
            return np.array([])
            
    # Fallback to old format
    pareto_file = os.path.join(checkpoint_path, "pareto_front.npy")
    
    if os.path.exists(pareto_file):
        print(f"Loading pareto front from {pareto_file}")
        return np.load(pareto_file)
    
    print(f"Error: No pareto front file found in {checkpoint_path}")
    return None

def load_metrics(result_path):
    """
    Load the metrics.csv file.
    """
    metrics_file = os.path.join(result_path, "metrics.csv")
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file not found at {metrics_file}")
        return None
    
    return pd.read_csv(metrics_file)

def calculate_hv(pareto_front, ref_point=None):
    """
    Calculate Hypervolume.
    If ref_point is None, it estimates it from the data (nadir point * 1.1).
    """
    if pareto_front is None or len(pareto_front) == 0:
        return 0.0
    
    if ref_point is None:
        # Estimate reference point: slightly worse than the worst value in each dimension
        ref_point = np.max(pareto_front, axis=0) * 1.1
        print(f"Reference point not provided. Using estimated reference point: {ref_point}")
    
    ind = HV(ref_point=ref_point)
    hv_value = ind(pareto_front)
    return hv_value

def plot_pareto_front(pareto_front, save_path, obj_labels=None, all_points=None, hv_value=None):
    """
    Plot the Pareto front. Supports 2D and 3D.
    """
    if pareto_front is None:
        return

    n_obj = pareto_front.shape[1]
    
    if obj_labels is None:
        obj_labels = [f"Obj {i+1}" for i in range(n_obj)]
    
    print(f"Plotting Pareto front with objectives: {obj_labels}")
    fig = plt.figure(figsize=(10, 8))
    
    title = f"Pareto Front ({n_obj}D)"
    if hv_value is not None:
        title += f"\nHV: {hv_value:.4e}"

    if n_obj == 2:
        ax = fig.add_subplot(111)
        
        # Plot all points as background
        if all_points is not None and len(all_points) > 0:
             if all_points.shape[1] >= 2:
                  ax.scatter(all_points[:, 0], all_points[:, 1], c='lightgray', s=15, alpha=0.3, label='Visited Solutions', zorder=1)

        # Plot Elite Solutions
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', s=60, edgecolors='black', label='Elite Solutions (Pareto)', zorder=2)
        
        # Connect the dots for 2D Pareto Front (Trade-off line)
        # Sort by first objective
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]
        ax.plot(sorted_front[:, 0], sorted_front[:, 1], c='red', alpha=0.6, linestyle='--', linewidth=1.5, zorder=2)

        ax.set_xlabel(obj_labels[0])
        ax.set_ylabel(obj_labels[1])
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        

    elif n_obj == 3:
        ax = fig.add_subplot(111, projection='3d')
        
        if all_points is not None and len(all_points) > 0:
             if all_points.shape[1] >= 3:
                 ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='lightgray', s=10, alpha=0.2, label='Visited Solutions')

        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c='red', s=60, edgecolors='black', label='Elite Solutions')
        ax.set_xlabel(obj_labels[0])
        ax.set_ylabel(obj_labels[1])
        ax.set_zlabel(obj_labels[2])
        ax.set_title(title)
        ax.legend()
    else:
        print(f"Plotting for {n_obj} objectives is not supported (only 2D and 3D).")
        return

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Pareto front plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze NSGA-II Results")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory (containing metrics.csv and checkpoint/)")
    parser.add_argument("--ref_point", type=float, nargs='+', help="Reference point for HV calculation (space-separated values)")
    parser.add_argument("--obj_labels", type=str, nargs='+', help="Labels for objectives (space-separated)")
    
    args = parser.parse_args()
    
    print(f"Analyzing results in: {args.result_path}")
    
    # 1. Load Data
    pareto_front = load_pareto_front(args.result_path)
    metrics_df = load_metrics(args.result_path)
    
    if pareto_front is None:
        return

    # Extract all visited points for plotting context (Background)
    all_points = None
    target_cols = []
    
    # Identify which columns correspond to current objectives
    if args.obj_labels:
        # User specified labels
        potential_cols = [f"{label}/current" for label in args.obj_labels]
        # Check if they exist
        if metrics_df is not None and all(c in metrics_df.columns for c in potential_cols):
             target_cols = potential_cols
    
    if not target_cols and metrics_df is not None:
         # Auto-detect from metrics_df if possible
         target_cols = [c for c in metrics_df.columns if c.endswith("/current")]
         # Match number of columns in loaded pareto front
         if len(target_cols) > pareto_front.shape[1]:
              target_cols = target_cols[:pareto_front.shape[1]]

    if metrics_df is not None and target_cols:
         all_points = metrics_df[target_cols].values
         # If labels weren't provided, set them now
         if not args.obj_labels:
              args.obj_labels = [c.replace("/current", "") for c in target_cols]

    # Filter objectives if user provided fewer labels than available columns
    if args.obj_labels:
        n_labels = len(args.obj_labels)
        n_cols = pareto_front.shape[1]
        
        if n_labels < n_cols:
            print(f"Filtering Pareto front: using first {n_labels} columns based on provided labels.")
            # Slice to keep only the first n_labels columns
            pareto_front = pareto_front[:, :n_labels]
            
            if all_points is not None and all_points.shape[1] > n_labels:
                all_points = all_points[:, :n_labels]

            # Remove duplicates that might have been created by projection
            pareto_front = np.unique(pareto_front, axis=0)
            
            # Re-calculate non-dominated front in the lower-dimensional space
            nds = NonDominatedSorting()
            fronts = nds.do(pareto_front)
            # fronts[0] contains indices of the first front (non-dominated solutions)
            pareto_front = pareto_front[fronts[0]]
            
            print(f"Re-calculated Pareto front size: {len(pareto_front)}")

    # 2. Calculate HV
    ref_point = np.array(args.ref_point) if args.ref_point else None
    hv = calculate_hv(pareto_front, ref_point)
    print(f"Hypervolume: {hv}")
    
    # 3. Plot Pareto Front
    plot_file = os.path.join(args.result_path, "pareto_front_plot.png")
    plot_pareto_front(pareto_front, plot_file, args.obj_labels, all_points=all_points, hv_value=hv)
    
    # 4. Save Analysis Summary
    summary_file = os.path.join(args.result_path, "analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Analysis for {args.result_path}\n")
        f.write(f"Hypervolume: {hv}\n")
        if ref_point is not None:
             f.write(f"Reference Point: {ref_point}\n")
        f.write(f"Number of Pareto Solutions: {len(pareto_front)}\n")
        f.write("\nPareto Front Solutions:\n")
        f.write(str(pareto_front))
    
    print(f"Analysis summary saved to {summary_file}")

    # 5. Optional: Save Pareto Front as CSV for easier viewing
    pareto_csv = os.path.join(args.result_path, "pareto_front.csv")
    pd.DataFrame(pareto_front, columns=args.obj_labels if args.obj_labels else None).to_csv(pareto_csv, index=False)
    print(f"Pareto front CSV saved to {pareto_csv}")

if __name__ == "__main__":
    main()
