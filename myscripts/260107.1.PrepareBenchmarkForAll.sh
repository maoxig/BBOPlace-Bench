#!/bin/bash

# Define the benchmark dictionaries
declare -A benchmark_dict
benchmark_dict["ispd2005"]="adaptec1 adaptec2 adaptec3 adaptec4 bigblue1 bigblue2 bigblue3 bigblue4"
benchmark_dict["openroad"]="swerv_wrapper"

declare -A benchmark_type_dict
benchmark_type_dict["ispd2005"]="aux"
benchmark_type_dict["openroad"]="openroad_def"

# Iterate over datasets and benchmarks
for dataset in "${!benchmark_dict[@]}"; do
    benchmarks=${benchmark_dict[$dataset]}
    benchmark_type=${benchmark_type_dict[$dataset]}
    
    for benchmark in $benchmarks; do
        echo "Processing dataset: $dataset, benchmark: $benchmark, type: $benchmark_type"
        python ../src/utils/prepare_benchmark_cache.py --dataset "$dataset" --benchmark "$benchmark" --benchmark_type "$benchmark_type" --n_macro 10000
    done
done