problem_formulation=mgo
algo=nsga3
benchmark_prefix=superblue

for i in 1  #3 4 5 7 10 16 18
do
benchmark=${benchmark_prefix}${i}
python ../src/main.py \
    --gpu="0,1,2,3"\
    --name=ICCAD2015_MGO_NSGA3_MP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=20 \
    --n_population=50 \
    --max_evals=1000 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=shuffle \
    --crossover=uniform \
    --eval_metrics='["hpwl","regularity","dataflow_cost"]' \
    --eval_gp_hpwl=False \
    --n_max_saving_placement=10 \
    --verbose=False \
    --n_partitions=6
done