problem_formulation=mgo
algo=nsga3
benchmark_prefix=superblue

for benchmark in ariane133 ariane136 bp bp_be bp_fe swerv_wrapper
do
python ../src/main.py \
    --gpu="0,1,2,3"\
    --name=MO_OPENROAD_MGO_NSGA3_MP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=10 \
    --n_population=50 \
    --max_evals=10000 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=shuffle \
    --crossover=uniform \
    --eval_metrics='["hpwl","regularity","dataflow_cost"]' \
    --eval_gp_hpwl=False \
    --n_max_saving_placement=100 \
    --verbose=False \
    --n_partitions=6
done