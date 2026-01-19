problem_formulation=hpo
algo=smsemoa

for benchmark in superblue1 superblue3 superblue4 superblue5 superblue7 superblue10 superblue16 superblue18
do
python ../src/main.py \
    --gpu="0,1,2,3"\
    --name=MO_ICCAD2015_HPO_SMSEMOA_MP \
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
    --eval_metrics='["hpwl","regularity","macro_grouping_cost"]' \
    --eval_gp_hpwl=False \
    --n_max_saving_placement=10 \
    --verbose=False
done
