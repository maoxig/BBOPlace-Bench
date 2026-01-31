problem_formulation=mgo
algo=nsga2

for benchmark in superblue3 # superblue4 superblue5 superblue7 superblue10 superblue16 superblue18
do
python ../src/main.py \
    --gpu="0,1,2,3"\
    --name=MO_ICCAD2015_MGO_NSGA2_MP \
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
    --eval_metrics='["hpwl","regularity","rudy"]' \
    --eval_gp_hpwl=False \
    --n_max_saving_placement=5 \
    --verbose=True
done
