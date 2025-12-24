problem_formulation=mgo
algo=ea
benchmark_prefix=superblue

for i in 3
do
benchmark=${benchmark_prefix}${i}

RAY_local_fs_capacity_threshold=0.99 RAY_DISK_USAGE_THRESHOLD=0.99 RAY_OBJECT_STORE_ALLOW_SPILL_TO_DISK=true python ../src/main.py \
    --name=ICCAD2015_MGO_EA_GP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=10 \
    --eval_gp_hpwl=True \
    --n_population=50 \
    --n_sampling_repeat=5 \
    --max_evals=1000 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=shuffle \
    --crossover=uniform \
    --eval_metrics='["hpwl", "regularity", "gp_hpwl"]' \
    --max_saving_placement=5 
done