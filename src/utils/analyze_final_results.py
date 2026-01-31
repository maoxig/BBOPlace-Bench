
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
    Looks for final_solutions.pkl (newest format), then elite_pool.pkl, then pareto_front.npy.
    """
    checkpoint_path = os.path.join(result_path, "checkpoint")
    
    # 1. Try loading final_solutions.pkl (The new standard)
    final_sol_file = os.path.join(checkpoint_path, "final_solutions.pkl")
    if os.path.exists(final_sol_file):
        print(f"Loading final solutions from {final_sol_file}")
        with open(final_sol_file, 'rb') as f:
            final_solutions = pickle.load(f)
        
        if final_solutions:
            Y_list = [sol['Y'] for sol in final_solutions]
            return np.array(Y_list)
        else:
            print("Warning: Final solutions list is empty.")
            # Fall through to try other files just in case
            
    # 2. Try loading elite pool (Intermediate format)
    elite_file = os.path.join(checkpoint_path, "elite_pool.pkl")
    if os.path.exists(elite_file):
        print(f"Loading elite pool from {elite_file}")
        with open(elite_file, 'rb') as f:
            elite_pool = pickle.load(f)
        
        if elite_pool:
            Y_list = [sol['Y'] for sol in elite_pool]
            return np.array(Y_list)
            
    # 3. Fallback to old format
    pareto_file = os.path.join(checkpoint_path, "pareto_front.npy")
    
    if os.path.exists(pareto_file):
        print(f"Loading pareto front from {pareto_file}")
        return np.load(pareto_file)
    
    print(f"Error: No solution file (final_solutions.pkl, elite_pool.pkl, or pareto_front.npy) found in {checkpoint_path}")
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
        # Use a small epsilon to avoid issues if max is 0
        ref_point = np.max(pareto_front, axis=0) * 1.1 + 1e-6
        print(f"Reference point not provided. Using estimated reference point: {ref_point}")
    
    try:
        ind = HV(ref_point=ref_point)
        hv_value = ind(pareto_front)
        return hv_value
    except Exception as e:
        print(f"Error calculating HV: {e}")
        return 0.0

def plot_pareto_front(pareto_front, save_path, obj_labels=None, all_points=None, hv_value=None, max_num_points=None):
    """
    Plot the Pareto front. Supports 2D and 3D.
    """
    if pareto_front is None:
        return

    n_obj = pareto_front.shape[1]
    
    if obj_labels is None:
        obj_labels = [f"Obj {i+1}" for i in range(n_obj)]
    
    print(f"Plotting Pareto front with objectives: {obj_labels}")
    
    # Apply max_points limit
    if max_num_points is not None and len(pareto_front) > max_num_points:
        print(f"Limiting Pareto front to {max_num_points} points.")
        nds = NonDominatedSorting()
        fronts = nds.do(pareto_front)
        selected_points = []
        for front in fronts:
            if len(selected_points) + len(front) <= max_num_points:
                selected_points.extend(front)
            else:
                remaining = max_num_points - len(selected_points)
                selected_points.extend(front[:remaining])
                break
        pareto_front = pareto_front[selected_points]

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

        # Plot All Candidates (Pareto Front inputs)
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], c='blue', s=40, edgecolors='black', label='Final Candidates', zorder=2, alpha=0.6)

        # Top 5 Selection (Assuming input pareto_front is sorted by NDS+CD, first 5 are the selected ones)
        # In BasicAlgo, we save sorted list. So we can just highlight the first min(5, len) points.
        K_SELECTED = 5
        n_selected = min(len(pareto_front), K_SELECTED)
        if n_selected > 0:
            selected_subset = pareto_front[:n_selected]
            ax.scatter(selected_subset[:, 0], selected_subset[:, 1], c='red', s=80, marker='*', edgecolors='black', label=f'Selected for PPA (Top {n_selected})', zorder=3)
            # Annotate
            for i in range(n_selected):
                ax.text(selected_subset[i, 0], selected_subset[i, 1], str(i+1), fontsize=9, color='black', ha='right', va='bottom')

        # Connect the dots for 2D Pareto Front (Trade-off line)
        nds = NonDominatedSorting()
        fronts = nds.do(pareto_front)
        nd_indices = fronts[0]
        nd_front = pareto_front[nd_indices]
        
        if len(nd_front) > 0:
            sorted_indices = np.argsort(nd_front[:, 0])
            sorted_front = nd_front[sorted_indices]
            ax.plot(sorted_front[:, 0], sorted_front[:, 1], c='green', alpha=0.5, linestyle='--', linewidth=1.5, zorder=2, label='Pareto Front')

        ax.set_xlabel(obj_labels[0])
        ax.set_ylabel(obj_labels[1])
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        

    elif n_obj == 3:
        ax = fig.add_subplot(111, projection='3d')
        
        if all_points is not None and len(all_points) > 0:
             if all_points.shape[1] >= 3:
                 ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='lightgray', s=10, alpha=0.2, label='Visited Solution')

        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c='blue', s=40, edgecolors='black', label='Final Candidates')
        
        K_SELECTED = 5
        n_selected = min(len(pareto_front), K_SELECTED)
        if n_selected > 0:
             selected_subset = pareto_front[:n_selected]
             ax.scatter(selected_subset[:, 0], selected_subset[:, 1], selected_subset[:, 2], c='red', s=80, marker='*', edgecolors='black', label=f'Selected for PPA')

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
    parser = argparse.ArgumentParser(description="Analyze Final Results")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory (containing metrics.csv and checkpoint/)")
    parser.add_argument("--ref_point", type=float, nargs='+', help="Reference point for HV calculation (space-separated values)")
    parser.add_argument("--obj_labels", type=str, nargs='+', help="Labels for objectives (space-separated)")
    parser.add_argument("--max_num_points", type=int, default=None, help="Maximum number of points to plot in the Pareto front")
    args = parser.parse_args()
    
    print(f"Analyzing final results in: {args.result_path}")
    
    # 1. Load Data
    final_Y = load_pareto_front(args.result_path)
    metrics_df = load_metrics(args.result_path)
    
    if final_Y is None:
        print("Could not load any solution data.")
        return

    print(f"Loaded {len(final_Y)} solutions.")

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
         # Match number of columns in loaded solutions
         if len(target_cols) > final_Y.shape[1]:
              target_cols = target_cols[:final_Y.shape[1]]

    if metrics_df is not None and target_cols:
         all_points = metrics_df[target_cols].values
         # If labels weren't provided, set them now
         if not args.obj_labels:
              args.obj_labels = [c.replace("/current", "") for c in target_cols]

    # Filter objectives if user provided fewer labels than available columns
    if args.obj_labels:
        n_labels = len(args.obj_labels)
        n_cols = final_Y.shape[1]
        
        if n_labels < n_cols:
            print(f"Filtering solutions: using first {n_labels} columns based on provided labels.")
            # Slice to keep only the first n_labels columns
            final_Y = final_Y[:, :n_labels]
            
            if all_points is not None and all_points.shape[1] > n_labels:
                all_points = all_points[:, :n_labels]

            # Remove duplicates that might have been created by projection
            final_Y = np.unique(final_Y, axis=0)
            
            # Since we projected, maybe check for non-dominated again to show a clean front in plot
            # But we keep all selected solutions for HV calculation usually, or filter. 
            # Let's clean it up for analysis purposes.
            nds = NonDominatedSorting()
            fronts = nds.do(final_Y)
            final_Y = final_Y[fronts[0]]
            
            print(f"Re-filtered non-dominated solutions size: {len(final_Y)}")

    # 2. Calculate HV
    ref_point = np.array(args.ref_point) if args.ref_point else None
    hv = calculate_hv(final_Y, ref_point)
    print(f"Hypervolume: {hv}")
    
    # 3. Plot Pareto Front
    plot_file = os.path.join(args.result_path, "final_solutions_plot.png")
    plot_pareto_front(final_Y, plot_file, args.obj_labels, all_points=all_points, hv_value=hv, max_num_points=args.max_num_points)
    
    # 4. Save Analysis Summary
    summary_file = os.path.join(args.result_path, "final_analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Final Analysis for {args.result_path}\n")
        f.write(f"Hypervolume: {hv}\n")
        if ref_point is not None:
             f.write(f"Reference Point: {ref_point}\n")
        f.write(f"Number of Solutions: {len(final_Y)}\n")
        f.write("\nSolutions (Y):\n")
        f.write(str(final_Y))
    
    print(f"Analysis summary saved to {summary_file}")

    # 5. Save as CSV for easier viewing
    csv_file = os.path.join(args.result_path, "final_solutions.csv")
    pd.DataFrame(final_Y, columns=args.obj_labels if args.obj_labels else None).to_csv(csv_file, index=False)
    print(f"Solutions CSV saved to {csv_file}")

if __name__ == "__main__":
    main()
