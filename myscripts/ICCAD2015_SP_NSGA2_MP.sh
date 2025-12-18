problem_formulation=sp
algo=nsga2
benchmark_prefix=superblue

for i in 1
do
benchmark=${benchmark_prefix}${i}
export CUDA_VISIBLE_DEVICES=1,2,3 
python ../src/main.py \
    --name=ICCAD2015_SP_NSGA2_MP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=10 \
    --eval_gp_hpwl=False \
    --n_population=50 \
    --n_sampling_repeat=5 \
    --max_evals=1000 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=inversion \
    --crossover=order \
    --eval_metrics='["hpwl","congestion",]'
done