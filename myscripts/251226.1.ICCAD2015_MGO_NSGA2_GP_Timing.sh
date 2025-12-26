problem_formulation=mgo
algo=nsga2
benchmark_prefix=superblue

for i in 3
do
benchmark=${benchmark_prefix}${i}
python ../src/main.py \
    --gpu="0,4,5,6"\
    --name=ICCAD2015_MGO_NSGA2_GP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=20 \
    --n_population=50 \
    --n_sampling_repeat=5 \
    --max_evals=1000 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=shuffle \
    --crossover=uniform \
    --eval_metrics='["hpwl","n_wns","n_tns"]' \
    --eval_gp_hpwl=True \
    --error_redirect=False \
    --n_max_saving_placement=5
done