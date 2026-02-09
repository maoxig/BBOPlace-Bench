problem_formulation=mgo
algo=moead

for benchmark in ariane133 ariane136 bp bp_be bp_fe swerv_wrapper
do
python ../src/main.py \
    --gpu="0,1,2,3"\
    --name=MO_OPENROAD_MGO_MOEAD_GP \
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
    --mutation=shuffle \
    --crossover=uniform \
    --eval_metrics='["gp_hpwl","overflow","route_utilization"]' \
    --eval_gp_hpwl=True \
    --n_max_saving_placement=5 \
    --adaptive_normalization=True \
    --verbose=False
done
