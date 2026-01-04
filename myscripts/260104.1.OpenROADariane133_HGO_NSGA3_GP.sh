problem_formulation=hpo
algo=nsga3

for i in 1
do
benchmark=ariane133
python ../src/main.py \
    --gpu="0,1"\
    --name=OpenROAD_HGO_NSGA3_GP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=4 \
    --n_population=10 \
    --n_sampling_repeat=2 \
    --max_evals=200 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=random_resetting \
    --crossover=uniform \
    --eval_metrics='["hpwl","dataflow_cost","macro_grouping_cost","gp_hpwl"]' \
    --eval_gp_hpwl=True \
    --error_redirect=False \
    --n_max_saving_placement=10 \
    --verbose=True
done