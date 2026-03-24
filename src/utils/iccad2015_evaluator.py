import os
import sys
import argparse
import contextlib
import logging
import time
import torch as th

def get_project_root():
    """
    Get the project root directory assuming this script is in src/utils/
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 2 levels: src/utils/ -> src/ -> root/
    return os.path.abspath(os.path.join(current_dir, "..", ".."))

def setup_dreamplace_env(root_dir):
    """
    Add DREAMPlace and thirdparty paths to sys.path
    """
    sys.path.append(root_dir)
    sys.path.append(os.path.join(root_dir, "thirdparty"))
    sys.path.append(os.path.join(root_dir, "thirdparty", "dreamplace"))
    
    # Set potential environment variable for modified DREAMPlace (if recompiled)
    flute_lut_dir = os.path.join(root_dir, "thirdparty", "DREAMPlace_source", "thirdparty", "flute")
    if os.path.exists(flute_lut_dir):
        os.environ["DREAMPLACE_FLUTE_LUT_DIR"] = flute_lut_dir
    else:
        # Fallback if source structure is different
        pass

@contextlib.contextmanager
def suppress_output(suppress=True):
    """
    Context manager to suppress both Python and C-level stdout/stderr.
    """
    if not suppress:
        yield
        return
        
    # Open devnull
    with open(os.devnull, "w") as devnull:
        # Save Python streams
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        
        # Save C file descriptors
        c_redirect_success = False
        try:
            fd_devnull = os.open(os.devnull, os.O_WRONLY)
            old_fd_stdout = os.dup(1)
            old_fd_stderr = os.dup(2)
            os.dup2(fd_devnull, 1)
            os.dup2(fd_devnull, 2)
            os.close(fd_devnull)
            c_redirect_success = True
        except Exception:
            c_redirect_success = False
            pass
            
        try:
            yield
        finally:
            # Restore Python streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Restore C file descriptors
            if c_redirect_success:
                try:
                    os.dup2(old_fd_stdout, 1)
                    os.dup2(old_fd_stderr, 2)
                    os.close(old_fd_stdout)
                    os.close(old_fd_stderr)
                except Exception:
                    pass

def evaluate_iccad2015_timing(def_path, benchmark_name, root_dir=None, verbose=False):
    """
    Evaluate TNS and WNS for a given DEF file on ICCAD2015 benchmark using DREAMPlace's timer.
    
    Args:
        def_path (str): Path to the DEF file to evaluate.
        benchmark_name (str): Name of the benchmark (e.g., 'superblue1').
        root_dir (str, optional): Root directory of the project. If None, inferred automatically.
        verbose (bool): Whether to see DREAMPlace logs. Defaults to False.
        
    Returns:
        dict: {
            "tns": float,      # Total Negative Slack (Real value, usually negative)
            "wns": float,      # Worst Negative Slack (Real value, usually negative)
            "n_tns": float,    # Normalized TNS score (Positive, matches dmp_actor logic)
            "n_wns": float     # Normalized WNS score (Positive, matches dmp_actor logic)
        } or None if failed.
    """
    if root_dir is None:
        root_dir = get_project_root()
        
    setup_dreamplace_env(root_dir)
    
    try:
        from thirdparty.dreamplace.Params import Params as DMPParams
        from thirdparty.dreamplace.PlaceDB import PlaceDB as DMPPlaceDB
        import thirdparty.dreamplace.Timer as Timer
        from thirdparty.dreamplace.NonLinearPlace import NonLinearPlace
    except ImportError as e:
        print(f"Error importing DREAMPlace: {e}")
        return None

    if not os.path.exists(def_path):
        print(f"Error: DEF path does not exist: {def_path}")
        return None

    # Determine benchmark source directory
    # Looking for benchmarks/iccad2015/{benchmark_name}
    benchmark_dir = os.path.join(root_dir, "benchmarks", "iccad2015", benchmark_name)
    if not os.path.exists(benchmark_dir):
        print(f"Error: Benchmark directory not found: {benchmark_dir}")
        return None

    # Load default params from config
    json_config_path = os.path.join(root_dir, "config", "algorithm", "dmp_config", f"{benchmark_name}.json")
    if not os.path.exists(json_config_path):
        print(f"Error: JSON config not found: {json_config_path}")
        return None

    # Define suffix to path helper
    def suffix2path(suffix: str) -> str:
        # Standard ICCAD2015 naming: superblue1/superblue1.def, etc.
        return os.path.join(benchmark_dir, f"{benchmark_name}") + suffix

    # Use context manager to suppress noisy output unless verbose
    with suppress_output(not verbose):
        try:
            start_time = time.time()
            
            params = DMPParams()
            params.load(json_config_path)

            # Ensure we are logged (even if suppressed, standard logging might pass through depending on config)
            # but mainly we rely on stdout suppression.
            
            # Configure Inputs for ICCAD2015 (DEF based)
            # We override the def_input to the user's provided DEF
            # But we keep other inputs pointing to the original benchmark files
            params.fromJson({
                "def_input": def_path,  # User provided DEF
                "lef_input": suffix2path(".lef"),
                "verilog_input": suffix2path(".v"),
                "early_lib_input": suffix2path("_Early.lib"),
                "late_lib_input": suffix2path("_Late.lib"),
                "sdc_input": suffix2path(".sdc"),
                "plot_flag": 0,
                "random_center_init_flag": 0,
                # "gpu": 0 # Let config decide, or force CPU if needed.
            })
            
            # Initialize PlaceDB (follow dmp_actor "def" path behavior)
            placedb = DMPPlaceDB()

                # Fallback for environments where read() is unavailable.
            placedb(params)
            
            # Initialize Timer
            # Timer usually runs on CPU (OpenTimer)
            timer = Timer.Timer()
            timer(params, placedb)
            timer.update_timing()

            # IMPORTANT:
            # Use timing_op(pos) path when available. This is the same pattern as dmp_actor
            # and is sensitive to placement coordinates from the DEF.
            tns_raw = None
            wns_raw = None
            time_unit = None

            placer = NonLinearPlace(params, placedb, timer=timer)
            # Keep consistent with dmp_actor._update_dmp_placer.

            timing_op = getattr(placer.op_collections, "timing_op", None)
            if timing_op is not None:
                time_unit = timing_op.timer.time_unit()
                if time_unit == 0:
                    time_unit = 1e-12

                timing_op(placer.pos[0].data.cpu())
                timing_op.timer.update_timing()

                tns_raw = timing_op.timer.report_tns_elw(split=1)
                wns_raw = timing_op.timer.report_wns(split=1)


            
            
            n_tns_val = tns_raw / (time_unit * 1e17)
            n_wns_val = wns_raw / (time_unit * 1e15)
            
            end_time = time.time()
            runtime = end_time - start_time
            
            return {
                "tns": float(tns_raw),
                "wns": float(wns_raw),
                "n_tns": -float(n_tns_val), 
                "n_wns": -float(n_wns_val),
                "time_unit": time_unit,
                "runtime": runtime
            }
            
        except Exception as e:
            # If creating params/placedb fails, we should know
            # Since output is suppressed, we might need to print to a file or force stderr
            # But the 'finally' block restores stderr, so if we raise, the outer handler catches it.
            raise e

def main():
    parser = argparse.ArgumentParser(description="Evaluate Timing (TNS/WNS) for ICCAD2015 DEF")
    parser.add_argument("--def_path", type=str, required=True, help="Path to the DEF file")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name (e.g. superblue1)")
    parser.add_argument("--root_dir", type=str, default=None, help="Project root directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Simple check
    if not os.path.exists(args.def_path):
        print(f"Error: DEF File not found {args.def_path}")
        sys.exit(1)
    
    # Infer root dir if not provided, to check benchmark existence before running
    root = args.root_dir if args.root_dir else get_project_root()
    bench_dir = os.path.join(root, "benchmarks", "iccad2015", args.benchmark)
    if not os.path.exists(bench_dir):
        print(f"Error: Benchmark {args.benchmark} not found in {bench_dir}")
        sys.exit(1)

    print(f"Evaluating timing for {args.benchmark} using {args.def_path}...")
    
    try:
        result = evaluate_iccad2015_timing(args.def_path, args.benchmark, args.root_dir, args.verbose)
        
        if result:
            print("\n=== Timing Results ===")
            print(f"TNS (raw): {result['tns']}")
            print(f"WNS (raw): {result['wns']}")
            print(f"n_tns (score): {result['n_tns']}")
            print(f"n_wns (score): {result['n_wns']}")
            print(f"Runtime: {result['runtime']:.4f} s")
            
            # Save report
            def_dir = os.path.dirname(args.def_path)
            def_name = os.path.basename(args.def_path)
            report_path = os.path.join(def_dir, f"{def_name}.timing.txt")
            with open(report_path, "w") as f:
                f.write(f"Timing Report for {def_name}\n")
                f.write(f"Benchmark: {args.benchmark}\n")
                f.write(f"Full Path: {os.path.abspath(args.def_path)}\n")
                f.write(f"TNS: {result['tns']}\n")
                f.write(f"WNS: {result['wns']}\n")
                f.write(f"n_tns: {result['n_tns']}\n")
                f.write(f"n_wns: {result['n_wns']}\n")
                f.write(f"Time Unit: {result['time_unit']}\n")
                f.write(f"Runtime: {result['runtime']:.4f} s\n")
            print(f"\nReport saved to: {report_path}")
            
        else:
            print("Evaluation failed (returned None).")
            sys.exit(1)
            
    except Exception as e:
        print(f"Evaluation failed with exception: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
