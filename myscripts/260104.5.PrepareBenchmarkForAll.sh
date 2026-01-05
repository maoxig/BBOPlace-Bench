#!/bin/bash

# Define the benchmark dictionaries
declare -A benchmark_dict
benchmark_dict["ispd2005"]="adaptec1 adaptec2 adaptec3 adaptec4 bigblue1 bigblue2 bigblue3 bigblue4"
benchmark_dict["iccad2015"]="superblue1 superblue3 superblue4 superblue5 superblue7 superblue10 superblue16 superblue18"
benchmark_dict["openroad"]="ariane133 ariane136 bp bp_be bp_fe bp_multi swerv_wrapper"

declare -A benchmark_type_dict
benchmark_type_dict["ispd2005"]="aux"
benchmark_type_dict["iccad2015"]="def"
benchmark_type_dict["openroad"]="openroad_def"

# Iterate over datasets and benchmarks
for dataset in "${!benchmark_dict[@]}"; do
    benchmarks=${benchmark_dict[$dataset]}
    benchmark_type=${benchmark_type_dict[$dataset]}
    
    for benchmark in $benchmarks; do
        echo "Processing dataset: $dataset, benchmark: $benchmark, type: $benchmark_type"
        python ../src/utils/prepare_benchmark_cache.py --dataset "$dataset" --benchmark "$benchmark" --benchmark_type "$benchmark_type"
    done
done