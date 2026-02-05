# python ~/project/BBOPlace-Bench/src/util/analyze_final_results.py --result_path /home/xp/project/BBOPlace-Bench/results/superblue1/MO_ICCAD2015_MGO_NSGA2_MP/mgo/nsga2/seed_1_2026-01-31_09-27-25 --obj_labels hpwl regularity rudy
#!/bin/bash

BASE_DIR="/home/xp/project/BBOPlace-Bench/results"
PYTHON_SCRIPT="/home/xp/project/BBOPlace-Bench/src/utils/analyze_final_results.py"


# Call the script for specific benchmarks and methods
for benchmark in ariane133 ariane136 bp bp_be bp_fe swerv_wrapper; do
    benchmark_dir="$BASE_DIR/$benchmark"
    if [ -d "$benchmark_dir" ]; then
        for method_dir in "$benchmark_dir"/MO_*; do
            if [ -d "$method_dir" ]; then
                find "$method_dir" -type d -name "seed_*" | while read -r seed_path; do
                    echo "Analyzing: $seed_path"
                    if [[ "$method_dir" == *"_MP" ]]; then
                        python "$PYTHON_SCRIPT" --result_path "$seed_path" --obj_labels hpwl regularity rudy
                    else
                        python "$PYTHON_SCRIPT" --result_path "$seed_path" --obj_labels gp_hpwl overflow route_utilization
                    fi
                done
            fi
        done
    fi
done
#for benchmark in superblue1 superblue3 superblue4 superblue5 superblue7 superblue10 superblue16 superblue18