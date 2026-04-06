docker run -it \
  --name mo-bbo-ppa \
  --gpus all \
  --rm \
  --shm-size=32g \
  --network host \
  -v .:/workspace \
  -w /workspace \
  --privileged \
  -v /nas1:/nas1 \
  orfs-dreamplace:latest \
  /bin/bash
 
 python3 src/utils/evaluate_best_gp_ppa.py --workspace . --seeds 1,2,3 --hv_json results/analysis_reports/hv/json/hv_summary_seeds_1_2_3.json --benchmarks OpenROAD --formulations MGO,HPO --def_per_seed 5 --jobs 4 --variant eval --output results/analysis_reports/ppa_eval --cleanup_flow_work always --num_shards 2 --shard_id 1 --worker_tag m1