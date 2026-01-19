problem_formulation=mgo
algo=nsga2

for benchmark in adaptec1 adaptec2 adaptec3 adaptec4 bigblue1 bigblue2 bigblue3 bigblue4
do
python ../src/main.py \
    --gpu="0,1,2,3"\
    --name=MO_ISPD2005_MGO_NSGA2_MP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=10 \
    --n_population=20 \
    --max_evals=5000 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=shuffle \
    --crossover=uniform \
    --eval_metrics='["hpwl","regularity","macro_grouping_cost"]' \
    --eval_gp_hpwl=False \
    --n_max_saving_placement=10 \
    --verbose=False
done
