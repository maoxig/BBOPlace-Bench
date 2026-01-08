problem_formulation=mgo
algo=nsga3

for benchmark in superblue1 # superblue3 superblue4 superblue5 superblue7 superblue10 superblue16 superblue18
do
python ../src/main.py \
    --gpu="0,1,2,3"\
    --name=MO_ICCAD2015_MGO_NSGA3_GP \
    --benchmark=${benchmark} \
    --placer=${problem_formulation} \
    --algorithm=${algo} \
    --run_mode=single \
    --n_cpu_max=10 \
    --n_population=50 \
    --max_evals=1000 \
    --max_eval_time=72 \
    --n_macro=512 \
    --sampling=random \
    --mutation=shuffle \
    --crossover=uniform \
    --eval_metrics='["hpwl","gp_hpwl","n_wns"]' \
    --eval_gp_hpwl=True \
    --n_max_saving_placement=10 \
    --verbose=False \
    --n_partitions=6 \
    --checkpoint=/home/xp/project/BBOPlace-Bench/results/superblue1/ICCAD2015_MGO_NSGA3_GP/mgo/nsga3/seed_1_2026-01-07_10-05-56/checkpoint
done