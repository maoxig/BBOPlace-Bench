problem_formulation=hpo
algo=nsga3
benchmark_prefix=superblue

for i in 1 #3 4 5 7 10 16 18
do
benchmark=${benchmark_prefix}${i}
python ../src/main.py \
    --gpu="0,1,2,3,4,5,6,7"\
    --name=ICCAD2015_HPO_NSGA3_GP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=10 \
    --n_population=10 \
    --max_evals=200 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=random_resetting \
    --crossover=uniform \
    --eval_metrics='["hpwl","gp_hpwl","n_wns"]' \
    --eval_gp_hpwl=True \
    --n_max_saving_placement=10 \
    --verbose=False \
    --n_partitions=6
done