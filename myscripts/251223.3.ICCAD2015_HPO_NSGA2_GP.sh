problem_formulation=hpo
algo=nsga2
benchmark_prefix=superblue

for i in 1
do
benchmark=${benchmark_prefix}${i}
python ../src/main.py \
    --name=ICCAD2015_HPO_NSGA2_GP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=10 \
    --eval_gp_hpwl=True \
    --n_sampling_repeat=2 \
    --n_max_saving_placement=4 \
    --max_evals=200 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=random_resetting \
    --crossover=uniform \
    --eval_metrics='["hpwl", "regularity", "gp_hpwl"]' \
    --error_redirect=False
done