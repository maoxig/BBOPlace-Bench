"""
重构后的 HPOPlacer - 使用 Ray Actor 并行优化超参数
"""

import logging
import os
import math
import ray
import time
import numpy as np
from typing import Dict, Union, List
from src.placer.basic_placer import BasicPlacer
from src.utils.constant import EPS
from src.placer.dmp_actor import DREAMPlaceActor
from src.utils.compute_res import comp_res

try:
    from thirdparty.dreamplace.Params import Params as DMPParams
except ImportError:
    DMPParams = None

Numeric = Union[int, float]

# 参数转换函数
orig_func = lambda x: x
round_func = lambda x: round(x)
sel_func = lambda options: lambda x: options[math.trunc(x) % len(options)]

# 超参数搜索空间
params_space = {
    # 分类参数
    "GP_num_bins_x": (0, 2, sel_func([1024, 2048])),
    "GP_num_bins_y": (0, 2, sel_func([1024, 2048])),
    "GP_optimizer": (0, 2, sel_func(["adam", "nesterov"])),
    "GP_wirelength": (0, 2, sel_func(["weighted_average", "logsumexp"])),
    "GP_iteration": (0, 1, sel_func([1000])),

    # 连续参数
    "GP_Llambda_density_weight_iteration": (1, 3, round_func),
    "GP_Lsub_iteration": (1, 3, round_func),
    "GP_learning_rate": (0.001, 0.01, orig_func),
    "GP_learning_rate_decay": (0.99, 1.0, orig_func),
    "RePlAce_LOWER_PCOF": (0.9, 0.99, orig_func),
    "RePlAce_UPPER_PCOF": (1.02, 1.15, orig_func),
    "RePlAce_ref_hpwl": (150000, 550000, round_func),
    "density_weight": (1e-06, 1e-04, orig_func),
    "gamma": (1, 4, orig_func),
    "stop_overflow": (0.06, 0.1, orig_func),
    "target_density": (0.8, 1.2, orig_func),
}


class HPOPlacer(BasicPlacer):
    DMP_CONFIG_PATH = "config/algorithm/dmp_config"
    DMP_TEMP_BENCHMARK_PATH = "benchmarks/.tmp/HPO"

    def __init__(self, 
                 args, 
                 placedb,
                 eval_metrics: List[str] = ["hpwl", ]):
        """
        初始化 HPO Placer
        
        Args:
            args: 参数对象
            placedb: 布局数据库
            n_workers: worker 进程数量
        """
        super().__init__(args, placedb, eval_metrics)
        self.args = args
        self.placedb = placedb

        num_cpus = getattr(self.args, 'num_cpus', 1)
        num_gpus = getattr(self.args, 'num_gpus', 0)
        # Reserve some CPUs for driver and tasks
        # If we have N CPUs, we can use roughly N/2 workers to allow N/2 concurrent tasks
        # Ensure at least 1 worker if possible
        n_workers = max(1, num_cpus // 2 - 1)
        # Calculate GPU resources per actor
        self.gpu_resources = 0
        if num_gpus > 0:
            # Distribute workers across GPUs
            actors_per_gpu = math.ceil(n_workers / num_gpus)
            # Set resource requirement slightly less than 1/N to avoid floating point issues preventing packing
            self.gpu_resources = 0.99 / actors_per_gpu
        
        print(f"Initializing {n_workers} DREAMPlace Actors for HPO Placer with {self.gpu_resources:.4f} GPU each...")        
        # 加载 DMP 配置
        self.params = DMPParams()
        self._load_dmp_config()
        
        # 转换 args 为字典
        self.args_dict = vars(args) if hasattr(args, '__dict__') else args
        
        self.actors = [self._create_actor() for _ in range(n_workers)]
        # 将 actors 赋值给 gp_evaluators 以复用 BasicPlacer 的逻辑
        self.gp_evaluators = self.actors
        

    def _create_actor(self):
        return DREAMPlaceActor.options(
            num_cpus=1, 
            num_gpus=self.gpu_resources,
            max_restarts=-1  # Infinite restarts allowed
        ).remote(
            self.args_dict, 
            self.placedb.canvas_width, 
            self.placedb.canvas_height,
            temp_benchmark_path=self._temp_benchmark_path,
            verbose=getattr(self.args, 'verbose', False),
        )
        

    @property
    def param_dims(self) -> int:
        return len(params_space.items())
        
    @property
    def _temp_benchmark_path(self):
        ROOT_DIR = self.args.ROOT_DIR
        return os.path.join(
            ROOT_DIR,
            HPOPlacer.DMP_TEMP_BENCHMARK_PATH,
            "%(benchmark)s" % self.args.__dict__
        )

    def _prepare_benchmark_aux(self):
        os.makedirs(self._temp_benchmark_path, exist_ok=True)
        self._link_files(HPOPlacer.AUX_FILES)
        
        # prepare .pl
        pl_file_path = os.path.join(
            self._temp_benchmark_path,
            "%(benchmark)s.pl" % self.args.__dict__
        )
        
        if os.path.exists(pl_file_path):
            return

        with open(pl_file_path, "w") as pl_file:
            pl_file.write(self.placedb.to_pl(fix_macro=False))

    def _prepare_benchmark_openroad_def(self):
        pass

    def _prepare_benchmark_def(self):
        os.makedirs(self._temp_benchmark_path, exist_ok=True)
        self._link_files(HPOPlacer.DEF_FILES)
        
        # prepare .def
        def_file_path = os.path.join(
            self._temp_benchmark_path,
            "%(benchmark)s.def" % self.args.__dict__
        )
        
        if os.path.exists(def_file_path):
            return

        with open(def_file_path, "w") as def_file:
            def_file.write(self.placedb.to_def(fix_macro=False))

    def _load_dmp_config(self):
        """加载 DREAMPlace 配置"""
        ROOT_DIR = self.args.ROOT_DIR
        json_file = f"{self.args.benchmark}.json"
        json_path = os.path.join(
            ROOT_DIR,
            self.DMP_CONFIG_PATH,
            json_file
        )
        self.params.load(json_path)

    def _load_genotype(self, x: Dict[str, Numeric]) -> Dict:
        """
        加载基因型（超参数）并转换为 DMP 参数
        """
        params_to_update = {}
        
        for param_name, value in x.items():
            lb, ub, tf = params_space[param_name]
            # 稍微放宽边界检查，避免浮点误差
            if not (lb - EPS < value < ub + EPS):
                 pass
            
            param_value = tf(value)
            
            if param_name.startswith("GP_"):
                # Global placement 阶段参数
                params_to_update.setdefault("global_place_stages", [{}])
                subject = params_to_update["global_place_stages"][0]
                entry_name = param_name.replace("GP_", "")
            else:
                # 其他参数
                subject = params_to_update
                entry_name = param_name
            
            subject[entry_name] = param_value
        
        return params_to_update

    def save_solution(self, macro_pos, n_eval, params=None):
        if self.args.benchmark_type == "openroad_def" and params is not None:
             logging.info("Creating temporary actor for saving OpenROAD solution (HPO)...")
             actor = self._create_actor()
             try:
                 suffix = "def"
                 placement_file = os.path.join(self.placement_save_path, f'{n_eval}.{suffix}')
                 figure_file = os.path.join(self.fig_save_path, f"{n_eval}.png")
                 
                 params_update = None
                 if isinstance(params, (list, np.ndarray)):
                     params_name = list(params_space.keys())
                     xi_dict = dict(zip(params_name, list(params)))
                     params_update = self._load_genotype(xi_dict)
                 elif isinstance(params, dict):
                     # Check if it has keys from params_space
                     if set(params.keys()).intersection(params_space.keys()):
                         params_update = self._load_genotype(params)
                     else:
                         params_update = params

                 if params_update:
                     ray.get(actor.evaluate_hyper_params.remote(
                         params_update, 
                         self.placedb.macro_lst, 
                         placement_file=placement_file, 
                         figure_file=figure_file, 
                         save_result=True
                     ))
                 else:
                     ray.get(actor.evaluate_macro_pos.remote(
                         macro_pos, 
                         placement_file=placement_file, 
                         figure_file=figure_file, 
                         save_result=True
                     ))
                 
             except Exception as e:
                 logging.error(f"Error saving OpenROAD HPO solution: {e}")
             finally:
                 ray.kill(actor)
        else:
            super().save_solution(macro_pos, n_eval, params)

    def _genotype2phenotype(self, x: Union[List, Dict]):
        """
        将基因型转换为表现型（宏单元位置）
        """
        params_name = list(params_space.keys())
        x = dict(tuple(zip(params_name, list(x))))
        
        # 加载参数
        params_update = self._load_genotype(x)
        
        # 使用第一个 Actor 执行布局
        actor = self.actors[0]
        
        try:
            result = ray.get(actor.evaluate_hyper_params.remote(
                params_update=params_update,
                macro_lst=self.placedb.macro_lst
            ))
            
            if result is None:
                return {}
            
            # 转换格式 (Actor 已经返回了正确的格式，这里做个保险)
            final_pos = {}
            for k, v in result["macro_pos"].items():
                try:
                    final_pos[k] = (v[0], v[1])
                except:
                    final_pos[k] = tuple(v)
            
            return final_pos, result
        except Exception as e:
            print(f"Error in HPO placement: {e}")
            return {}, {}

    def evaluate(self, x):
        """
        并行评估种群 - 使用 Actor Pool (Robust Version)
        """
        t_start = time.time()
        
        start_idx = self.counter
        self.counter += len(x)
        
        suffix_map = {"aux" : "pl", "def" : "def" , "openroad_def": "def"}
        suffix = suffix_map[self.args.benchmark_type]
        
        # Prepare tasks
        tasks = []
        for i, xi in enumerate(x):
            n_eval = start_idx + i + 1
            placement_file = os.path.join(self.placement_save_path, f'{n_eval}.{suffix}')
            figure_file = os.path.join(self.fig_save_path, f"{n_eval}.png")
            
            params_name = list(params_space.keys())
            xi_list = list(xi)
            xi_dict = dict(zip(params_name, xi_list))
            params_update = self._load_genotype(xi_dict)
            
            tasks.append({
                "params_update": params_update,
                "placement_file": placement_file,
                "figure_file": figure_file,
                "index": i
            })

        results = [None] * len(x)
        
        idle_actors = list(range(len(self.actors)))
        restarting_actors = {} # future (ping) -> actor_index
        busy_actors = {} # future (work) -> (actor_index, task_index, start_time)
        
        pending_tasks = tasks[:] 
        

        # Configure timeout and retries
        TASK_TIMEOUT = getattr(self.args, "task_timeout", 900) # 15 minutes default
        MAX_RETRIES = 5 # Maximum number of retries per task

        while len(pending_tasks) > 0 or len(busy_actors) > 0 or len(restarting_actors) > 0:
            current_time = time.time()
            
            # 0. Check for timed out actors
            timed_out_futures = []
            for f, (actor_idx, task_idx, start_time) in busy_actors.items():
                if current_time - start_time > TASK_TIMEOUT:
                    timed_out_futures.append(f)
            
            for f in timed_out_futures:
                actor_idx, task_idx, _ = busy_actors.pop(f)
                
                # Check retry count
                if tasks[task_idx].get("retry_count", 0) >= MAX_RETRIES:
                    logging.error(f"Task {task_idx} failed/timed out {MAX_RETRIES} times. Skipping task.")
                    # Mark as failed (results[task_idx] remains None or set to empty dict to trigger INF)
                    results[task_idx] = {} 
                    
                    # Kill the actor as it's stuck
                    logging.warning(f"Killing stuck actor {actor_idx}...")
                    ray.kill(self.actors[actor_idx])
                    # Wait for resources to be released
                    time.sleep(3) 

                    # Recreate actor for future use
                    new_actor = self._create_actor()
                    self.actors[actor_idx] = new_actor
                    
                    # Ping to wait for initialization
                    ping_future = new_actor.evaluate_macro_pos.remote({})
                    restarting_actors[ping_future] = actor_idx
                    
                else:
                    logging.warning(f"Actor {actor_idx} timed out processing task {task_idx} (> {TASK_TIMEOUT}s). Killing and retrying (Attempt {tasks[task_idx].get('retry_count', 0) + 1}/{MAX_RETRIES})...")
                    
                    # Update retry count
                    tasks[task_idx]["retry_count"] = tasks[task_idx].get("retry_count", 0) + 1
                    
                    # Kill and recreate actor
                    ray.kill(self.actors[actor_idx])
                    # Wait for resources to be released
                    time.sleep(3) 

                    new_actor = self._create_actor()
                    self.actors[actor_idx] = new_actor
                    
                    # Ping to wait for initialization
                    ping_future = new_actor.evaluate_macro_pos.remote({})
                    restarting_actors[ping_future] = actor_idx
                    
                    # Requeue task
                    pending_tasks.insert(0, tasks[task_idx])

            # 1. Check for completed restarts
            if restarting_actors:
                ready_futures, _ = ray.wait(list(restarting_actors.keys()), num_returns=len(restarting_actors), timeout=0)
                for f in ready_futures:
                    actor_idx = restarting_actors.pop(f)
                    try:
                        ray.get(f)
                        idle_actors.append(actor_idx)
                    except Exception as e:
                        print(f"Actor {actor_idx} restart failed: {e}. Retrying...")
                        ray.kill(self.actors[actor_idx])
                        time.sleep(3) # Wait before retry
                        new_actor = self._create_actor()
                        self.actors[actor_idx] = new_actor
                        ping_future = new_actor.evaluate_macro_pos.remote({}) # Ping
                        restarting_actors[ping_future] = actor_idx
            
            # 2. Assign tasks to idle actors
            while pending_tasks and idle_actors:
                task = pending_tasks.pop(0)
                # Check for cancelled tasks (if any mechanism existed, but here we handled it in timeout block)
                
                actor_idx = idle_actors.pop(0)
                actor = self.actors[actor_idx]
                
                future = actor.evaluate_hyper_params.remote(
                    params_update=task["params_update"],
                    macro_lst=self.placedb.macro_lst,
                    placement_file=task["placement_file"],
                    figure_file=task["figure_file"]
                )
                busy_actors[future] = (actor_idx, task["index"], time.time())
            
            # 3. Wait for work to complete
            wait_list = list(busy_actors.keys()) + list(restarting_actors.keys())
            if not wait_list:
                if not pending_tasks:
                    break
                # Only restarting actors are left, wait a bit
                time.sleep(0.1) 
                continue

            # Use timeout to allow checking for timeouts in the loop
            done_futures, _ = ray.wait(wait_list, num_returns=1, timeout=1.0)
            
            for f in done_futures:
                if f in busy_actors:
                    # Work completed
                    actor_idx, task_idx, _ = busy_actors.pop(f)
                    
                    try:
                        result = ray.get(f)
                        results[task_idx] = result
                        
                        # Check if we need to restart the actor
                        # HPO Placer only needs restart on crash, not on dirty state
                        idle_actors.append(actor_idx)
                    except Exception as e:
                        if tasks[task_idx].get("retry_count", 0) >= MAX_RETRIES:
                             logging.error(f"Task {task_idx} failed with error {MAX_RETRIES} times: {e}. Skipping.")
                             results[task_idx] = {}
                             
                             # Kill and recreate actor (it might be in bad state)
                             ray.kill(self.actors[actor_idx])
                             time.sleep(3)
                             new_actor = self._create_actor()
                             self.actors[actor_idx] = new_actor
                             
                             ping_future = new_actor.evaluate_macro_pos.remote({})
                             restarting_actors[ping_future] = actor_idx
                        else:
                            logging.warning(f"Actor {actor_idx} failed processing task {task_idx}: {e}. Retrying (Attempt {tasks[task_idx].get('retry_count', 0) + 1}/{MAX_RETRIES})...")
                            
                            tasks[task_idx]["retry_count"] = tasks[task_idx].get("retry_count", 0) + 1

                            # Kill and recreate actor
                            ray.kill(self.actors[actor_idx])
                            time.sleep(3)
                            
                            new_actor = self._create_actor()
                            self.actors[actor_idx] = new_actor
                            
                            # Ping to wait for initialization
                            ping_future = new_actor.evaluate_macro_pos.remote({})
                            restarting_actors[ping_future] = actor_idx
                            
                            # Requeue task
                            pending_tasks.insert(0, tasks[task_idx])
                    
                elif f in restarting_actors:
                    if f in restarting_actors: 
                        actor_idx = restarting_actors.pop(f)
                        try:
                            ray.get(f)
                            idle_actors.append(actor_idx)
                        except Exception as e:
                            print(f"Actor {actor_idx} restart failed: {e}. Retrying...")
                            ray.kill(self.actors[actor_idx])
                            time.sleep(3)
                            new_actor = self._create_actor()
                            self.actors[actor_idx] = new_actor
                            ping_future = new_actor.evaluate_macro_pos.remote({})
                            restarting_actors[ping_future] = actor_idx
        
        # 3. 后处理和计算其他指标
        macro_pos_list = []
        metric_lists = {k: [] for k in self.eval_metrics}
        
        for res in results:
            # 提取 macro_pos
            macro_pos = {}
            if res and "macro_pos" in res:
                for k, v in res["macro_pos"].items():
                    # 确保格式正确
                    if hasattr(v, '__iter__'):
                        macro_pos[k] = (float(v[0]), float(v[1]))
                    else:
                        macro_pos[k] = v
            else:
                # Handle failure case
                macro_pos = {}

            macro_pos_list.append(macro_pos)
            
            # 计算常规指标 (hpwl, regularity 等)
            computed_metrics = comp_res(macros_pos=macro_pos, placedb=self.placedb, eval_metrics=self.eval_metrics)
            
            if res:
                computed_metrics.update(res)
            
            # 收集结果
            for metric in self.eval_metrics:
                val = computed_metrics.get(metric, 0)
                metric_lists[metric].append(val)
                
        # 转换为 numpy array
        final_results = {k: np.array(v) for k, v in metric_lists.items()}
        
        t_eval_solution = time.time() - t_start
        self.t_eval_solution_total += t_eval_solution
        # self._manage_saved_files(self.placement_save_path, self.n_max_saving_placement)
        # self._manage_saved_files(self.fig_save_path, self.n_max_saving_placement)
        return final_results, macro_pos_list

    def __deepcopy__(self, memo=None):
        """防止深拷贝"""
        return self
    
