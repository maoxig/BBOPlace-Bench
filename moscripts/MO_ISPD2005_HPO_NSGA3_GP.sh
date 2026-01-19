problem_formulation=hpo
algo=nsga3

for benchmark in adaptec1 adaptec2 adaptec3 adaptec4 bigblue1 bigblue3
do
python ../src/main.py \
    --gpu="0,1,2,3"\
    --name=MO_ISPD2005_HPO_NSGA3_GP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=10 \
    --n_population=20 \
    --max_evals=500 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=random_resetting \
    --crossover=uniform \
    --eval_metrics='["gp_hpwl","dataflow_cost"]' \
    --eval_gp_hpwl=True \
    --n_max_saving_placement=10 \
    --verbose=False \
    --n_partitions=12
done