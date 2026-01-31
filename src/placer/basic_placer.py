from abc import abstractmethod
from tabnanny import verbose

import numpy as np
import math
from src.placer.dmp_actor import DREAMPlaceActor
from src.utils.debug import *
from src.utils.compute_res import comp_res
from src.utils.read_benchmark.read_aux import write_pl
from src.utils.read_benchmark.read_def import write_def, write_openroad_def
from src.utils.constant import INF, get_n_power


import os
import csv
import ray
import time
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

@ray.remote(num_cpus=1)
def evaluate_placer(placer_ref, x0, actor=None, placement_file=None, figure_file=None):
    return placer_ref._evaluate(x0, actor, placement_file, figure_file)

class BasicPlacer:
    DMP_TEMP_BENCHMARK_PATH = "benchmarks/.tmp"
    AUX_FILES = [
        "%(benchmark)s.aux",
        "%(benchmark)s.scl",
        "%(benchmark)s.wts",
        "%(benchmark)s.nets",
        "%(benchmark)s.nodes"
    ]
    DEF_FILES = [
        "%(benchmark)s.lef",
        "%(benchmark)s.v",
        "%(benchmark)s.sdc",
        "%(benchmark)s_Early.lib",
        "%(benchmark)s_Late.lib"
    ]

    def __init__(self, args, placedb, eval_metrics= ["hpwl"]) -> None:
        self.args = args
        self.placedb = placedb
        self.eval_metrics = eval_metrics

        self.canvas_width  = placedb.canvas_width
        self.canvas_height = placedb.canvas_height
        
        self._prepare_benchmark()
        
        self.fig_save_path       = os.path.join(args.result_path, "figures")
        self.placement_save_path = os.path.join(args.result_path, "placements")
        os.makedirs(self.fig_save_path, exist_ok=True)
        os.makedirs(self.placement_save_path, exist_ok=True)

        self.metrics_file = os.path.join(args.result_path, "metrics.csv")
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            header = ["n_eval"] + [f"{prefix}_{metric}" for prefix in ["current","his_best", "pop_best", "pop_avg", "pop_std"] for metric in self.eval_metrics] + \
                        ["t_each_eval", "avg_t_each_eval", "avg_t_algo_optimization", "avg_t_eval_solution"]
            writer.writerow(header)
        
        self.placement_saving_lst = []
        self.figure_saving_lst = []
        self.n_max_saving_placement = args.n_max_saving_placement
        self.t_eval_solution_total = 0
        self.counter = 0
        
        self.gp_evaluators = []
        self.poster = None
        
        # Initialize resources info
        num_cpus = getattr(self.args, 'num_cpus', 1)
        num_gpus = getattr(self.args, 'num_gpus', 0)
        n_workers = max(1, num_cpus // 2 - 1)
        self.gpu_resources = 0
        if num_gpus > 0:
            actors_per_gpu = math.ceil(n_workers / num_gpus)
            self.gpu_resources = 0.99 / actors_per_gpu

        if self.args.eval_gp_hpwl and self.args.placer != 'hpo':
            print(f"Initializing {n_workers} DREAMPlace Actors for BasicPlacer with {self.gpu_resources:.4f} GPU each...")
            # Use max_restarts=-1 (infinite restarts) and max_task_retries=-1
            # But crucially, use max_calls to restart actor after N calls to clear memory leaks
            self.gp_evaluators = [self._create_actor() for _ in range(n_workers)]
            
        if self.args.benchmark_type == "openroad_def" and self.args.placer != 'hpo':
            if len(self.gp_evaluators) > 0:
                self.poster = self.gp_evaluators[0]
            else:
                logging.info("Initializing discrete DMP Actor for saving OpenROAD results...")
                self.poster = self._create_actor()

    def _create_actor(self):
        return DREAMPlaceActor.options(
            num_cpus=1, 
            num_gpus=self.gpu_resources,
        ).remote(
            vars(self.args), 
            self.placedb.canvas_width, 
            self.placedb.canvas_height,
            temp_benchmark_path=self._temp_benchmark_path,
            verbose=getattr(self.args, 'verbose', False),
        )
        
    @property
    def _orig_benchmark_path(self):
        ROOT_DIR = self.args.ROOT_DIR
        return os.path.join(
            ROOT_DIR,
            self.args.benchmark_path
        )

    @property
    def _temp_benchmark_path(self):
        ROOT_DIR = self.args.ROOT_DIR
        return os.path.join(
            ROOT_DIR,
            self.DMP_TEMP_BENCHMARK_PATH,
            "%(benchmark)s" % self.args.__dict__
        )

    def _link_files(self, files):
        for file_name in files:
            orig = os.path.join(
                self._orig_benchmark_path,
                file_name % self.args.__dict__)

            if not os.path.exists(orig):
                continue

            link = os.path.join(
                self._temp_benchmark_path,
                file_name % self.args.__dict__)
            
            os.system(f"ln -sfr {orig} {link}")

    def _generate_random_initial_placement(self):
        n_grid_x = self.args.n_grid_x
        n_grid_y = self.args.n_grid_y
        grid_width = self.canvas_width / n_grid_x
        grid_height = self.canvas_height / n_grid_y
        
        macro_pos = {} # macro_name -> (x, y) (physical coords)
        
        for macro in self.placedb.macro_lst:
            size_x = self.placedb.node_info[macro]["size_x"]
            size_y = self.placedb.node_info[macro]["size_y"]
            scaled_size_x = math.ceil(size_x / grid_width)
            scaled_size_y = math.ceil(size_y / grid_height)
            
            # Boundary check: max valid x index is n_grid_x - scaled_size_x
            max_x = max(0, n_grid_x - scaled_size_x)
            max_y = max(0, n_grid_y - scaled_size_y)
            
            chosen_x = np.random.randint(0, max_x + 1)
            chosen_y = np.random.randint(0, max_y + 1)
                
            macro_pos[macro] = (chosen_x * grid_width, chosen_y * grid_height)
            
        return macro_pos

    def _prepare_benchmark_aux(self):
        os.makedirs(self._temp_benchmark_path, exist_ok=True)
        self._link_files(self.AUX_FILES)
        
        # prepare .pl
        pl_file_path = os.path.join(
            self._temp_benchmark_path,
            "%(benchmark)s.pl" % self.args.__dict__
        )
        
        if os.path.exists(pl_file_path):
            return

        macro_pos = self._generate_random_initial_placement()
        write_pl(pl_file_path, macro_pos, self.placedb)

    def _prepare_benchmark_def(self):
        os.makedirs(self._temp_benchmark_path, exist_ok=True)
        self._link_files(self.DEF_FILES)
        
        # prepare .def
        def_file_path = os.path.join(
            self._temp_benchmark_path,
            "%(benchmark)s.def" % self.args.__dict__
        )
        
        if os.path.exists(def_file_path):
            return
        # only generate random placement for temp benchmark, not for real use

          #generate random placement for temp benchmark
        macro_pos = self._generate_random_initial_placement()
        write_def(def_file_path, macro_pos, self.placedb)

    def _prepare_benchmark_openroad_def(self):
        pass

    def _prepare_benchmark(self):
        type_mapping = {
            "aux": self._prepare_benchmark_aux,
            "def": self._prepare_benchmark_def,
            "openroad_def": self._prepare_benchmark_openroad_def,
        }
        if self.args.benchmark_type in type_mapping:
            type_mapping[self.args.benchmark_type]()
        else:
            raise NotImplementedError


    def _evaluate(self, x, actor=None, placement_file=None, figure_file=None):
        # 单个评估逻辑，主要用于非批量场景或 fallback
        res = {}
        macro_pos = {}
        macro_pos, info = self._genotype2phenotype(x)
        res = comp_res(macros_pos=macro_pos, placedb=self.placedb, eval_metrics=self.eval_metrics)
        
        gp_res = {} 
        if macro_pos and len(macro_pos) > 0: # 非空
            if self.args.eval_gp_hpwl and actor:
                gp_res = ray.get(actor.evaluate_macro_pos.remote(macro_pos, placement_file, figure_file)) # {macro_pos: {}, eval_metric: value, ...}
                del gp_res["macro_pos"] # { eval_metric: value, ...}
            res.update(gp_res)
            
        else:
            for metric in self.eval_metrics:
                res[metric] = INF
        return res, macro_pos
    
    
    def evaluate(self, x):
        t = time.time()
        
        start_idx = self.counter
        self.counter += len(x)
        
        suffix_map = {"aux" : "pl", "def" : "def", "openroad_def": "def"}
        suffix = suffix_map[self.args.benchmark_type]
        placer_ref = ray.put(self)

        # Prepare tasks
        tasks = []
        for i, x0 in enumerate(x):
            n_eval = start_idx + i + 1
            placement_file = os.path.join(self.placement_save_path, f'{n_eval}.{suffix}')
            figure_file = os.path.join(self.fig_save_path, f"{n_eval}.png")
            tasks.append({
                "x0": x0,
                "placement_file": placement_file,
                "figure_file": figure_file,
                "index": i
            })

        results = [None] * len(x)

        if not self.gp_evaluators:
            futures = []
            for task in tasks:
                futures.append(evaluate_placer.remote(placer_ref, task["x0"], None, task["placement_file"], task["figure_file"]))
            results = ray.get(futures)
        else:
            idle_actors = list(range(len(self.gp_evaluators)))
            restarting_actors = {} # future (ping) -> actor_index
            busy_actors = {} # future (work) -> actor_index
            
            pending_tasks = tasks[:] 
            
            while len(pending_tasks) > 0 or len(busy_actors) > 0 or len(restarting_actors) > 0:
                # 1. Check for completed restarts
                if restarting_actors:
                    ready_futures, _ = ray.wait(list(restarting_actors.keys()), num_returns=len(restarting_actors), timeout=0)
                    for f in ready_futures:
                        actor_idx = restarting_actors.pop(f)
                        idle_actors.append(actor_idx)
                
                # 2. Assign tasks to idle actors
                while pending_tasks and idle_actors:
                    task = pending_tasks.pop(0)
                    actor_idx = idle_actors.pop(0)
                    actor = self.gp_evaluators[actor_idx]
                    
                    future = evaluate_placer.remote(
                        placer_ref, 
                        task["x0"], 
                        actor, 
                        task["placement_file"], 
                        task["figure_file"]
                    )
                    busy_actors[future] = (actor_idx, task["index"])
                
                # 3. Wait for work to complete
                wait_list = list(busy_actors.keys()) + list(restarting_actors.keys())
                if not wait_list:
                    if not pending_tasks:
                        break
                    continue

                done_futures, _ = ray.wait(wait_list, num_returns=1)
                
                for f in done_futures:
                    if f in busy_actors:
                        # Work completed
                        actor_idx, task_idx = busy_actors.pop(f)
                        
                        try:
                            result = ray.get(f)
                            results[task_idx] = result
                            
                            # Check if we need to restart the actor
                            # result is (res_dict, macro_pos)
                            # If macro_pos is empty, the actor wasn't used for evaluation, so no memory leak
                            macro_pos_result = result[1]
                            
                            if not macro_pos_result or len(macro_pos_result) == 0:
                                # Actor clean, return to pool immediately
                                idle_actors.append(actor_idx)
                            else:
                                # Actor dirty, restart it
                                ray.kill(self.gp_evaluators[actor_idx])
                                
                                new_actor = self._create_actor()
                                self.gp_evaluators[actor_idx] = new_actor
                                
                                # Ping to wait for initialization
                                ping_future = new_actor.evaluate_macro_pos.remote({})
                                restarting_actors[ping_future] = actor_idx
                        except Exception as e:
                            logging.warning(f"Actor {actor_idx} failed processing task {task_idx}: {e}. Retrying...")
                            
                            # Kill and recreate actor
                            ray.kill(self.gp_evaluators[actor_idx])
                            new_actor = self._create_actor()
                            self.gp_evaluators[actor_idx] = new_actor
                            
                            # Ping to wait for initialization
                            ping_future = new_actor.evaluate_macro_pos.remote({})
                            restarting_actors[ping_future] = actor_idx
                            
                            # Requeue task
                            pending_tasks.insert(0, tasks[task_idx])
                        
                    elif f in restarting_actors:
                        # Restart completed
                        actor_idx = restarting_actors.pop(f)
                        try:
                            ray.get(f)
                            idle_actors.append(actor_idx)
                        except Exception as e:
                            print(f"Actor {actor_idx} restart failed: {e}. Retrying...")
                            ray.kill(self.gp_evaluators[actor_idx])
                            new_actor = self._create_actor()
                            self.gp_evaluators[actor_idx] = new_actor
                            ping_future = new_actor.evaluate_macro_pos.remote({})
                            restarting_actors[ping_future] = actor_idx

        t_eval_solution = time.time() - t
        self.t_eval_solution_total += t_eval_solution

        res = {}
        for eval_metric in self.eval_metrics:
            res[eval_metric] = np.array([result[0][eval_metric] for result in results])
        macro_pos_list = [result[1] for result in results]

        # mangage figure
        # self._manage_saved_files(self.fig_save_path, self.n_max_saving_placement)
        # self._manage_saved_files(self.placement_save_path, self.n_max_saving_placement)
        
        return res, macro_pos_list


    def save_placement(self, macro_pos, n_eval):
        logging.info("Placer saving placement")
        suffix_map = {
            "aux" : "pl",
            "def" : "def",
            "openroad_def": "def",
        }
        suffix = suffix_map[self.args.benchmark_type]
        file_name = os.path.join(self.placement_save_path, 
                                f'{n_eval}.{suffix}')
        
        if self.args.benchmark_type == "openroad_def" and self.poster:
             ray.get(self.poster.update_and_save.remote(macro_pos, placement_file=file_name))
             return

        type_map = {
            "aux" : write_pl,
            "def" : write_def,
            "openroad_def": write_openroad_def,
        }
        type_map[self.args.benchmark_type](file_name, macro_pos, self.placedb)
        
    
    def plot(self, macro_pos:dict, n_eval:int):
        logging.info("Placer plotting figure")
        file_name = os.path.join(self.fig_save_path, f"{n_eval}.png")

        if self.args.benchmark_type == "openroad_def" and self.poster:
             ray.get(self.poster.update_and_save.remote(macro_pos, figure_file=file_name))
             return

        self._plot_macro(macro_pos, file_name)


    def _manage_saved_files(self, directory, max_files):
        """
        管理保存的文件，保留最新的 max_files 个文件
        """
        try:
            files = [os.path.join(directory, f) for f in os.listdir(directory)]
            files = [f for f in files if os.path.isfile(f)]
            
            if len(files) <= max_files:
                return
                
            # 按修改时间排序 (最新的在最后)
            files.sort(key=os.path.getmtime)
            
            # 删除旧文件
            files_to_delete = files[:-max_files]
            for f in files_to_delete:
                try:
                    os.remove(f)
                except OSError as e:
                    logging.warning(f"Error deleting file {f}: {e}")
                    
        except Exception as e:
            logging.warning(f"Error managing saved files in {directory}: {e}")

    def _plot_macro(self, macro_pos, file_name):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="auto")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        for macro in macro_pos:
            pos_x, pos_y = macro_pos[macro]
            size_x, size_y = self.placedb.node_info[macro]["size_x"], self.placedb.node_info[macro]["size_y"]

            pos_x = pos_x / self.placedb.canvas_ux
            pos_y = pos_y / self.placedb.canvas_uy
            size_x = size_x / self.placedb.canvas_ux
            size_y = size_y / self.placedb.canvas_uy
            ax.add_patch(
                patches.Rectangle(
                    (pos_x, pos_y),
                    size_x, size_y,
                    linewidth=1, edgecolor='k'
                )
            )

        fig.savefig(file_name, dpi=90, bbox_inches='tight')
        plt.close()
        
    def save_metrics(
            self, 
            n_eval, 
            current_Y,
            his_best_Y, 
            pop_best_Y, 
            pop_avg_Y, 
            pop_std_Y,
            t_each_eval=0,
            avg_t_each_eval=0,
            avg_t_eval_solution=0,
            ):
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            content = [n_eval] + [value for Y in [current_Y, his_best_Y, pop_best_Y, pop_avg_Y, pop_std_Y] for value in Y] + [t_each_eval, avg_t_each_eval, avg_t_each_eval - avg_t_eval_solution, avg_t_eval_solution ]
            writer.writerow(content)

    def _save_checkpoint(self, checkpoint_path):
        def save_and_delete(set_new, set_old, is_placement=True):
            if is_placement:
                suffix = "placements"
            else:
                suffix = "figures"
            for file_name in set_old - set_new:
                os.system(f"rm {os.path.join(checkpoint_path, suffix, file_name)}")
            
            for file_name in set_new - set_old:
                if is_placement:
                    os.system(f"cp {os.path.join(self.placement_save_path, file_name)} "+\
                              f"{os.path.join(checkpoint_path, suffix, file_name)}")
                else:
                    os.system(f"cp {os.path.join(self.fig_save_path, file_name)} "+\
                              f"{os.path.join(checkpoint_path, suffix, file_name)}")

        os.makedirs(os.path.join(checkpoint_path, "placements"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_path, "figures"), exist_ok=True)
        
        placement_set_old = set(os.listdir(os.path.join(checkpoint_path, "placements")))
        placement_set_new = set([os.path.basename(file_name) for file_name in self.placement_saving_lst])

        figure_set_old = set(os.listdir(os.path.join(checkpoint_path, "figures")))
        figure_set_new = set([os.path.basename(file_name) for file_name in self.figure_saving_lst])

        save_and_delete(placement_set_new, placement_set_old, is_placement=True)
        save_and_delete(figure_set_new, figure_set_old, is_placement=False)
        
        # Save counter
        import json
        with open(os.path.join(checkpoint_path, "placer_state.json"), "w") as f:
            json.dump({"counter": self.counter}, f)

    def _load_checkpoint(self, checkpoint_path):
        # Load counter
        import json
        state_file = os.path.join(checkpoint_path, "placer_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
                self.counter = state.get("counter", 0)
        
        for file_name in os.listdir(os.path.join(checkpoint_path, "placements")):
            self.placement_saving_lst.append(os.path.join(self.placement_save_path, file_name))
            os.system(f"cp {os.path.join(checkpoint_path, 'placements', file_name)} "+\
                      f"{os.path.join(self.placement_save_path, file_name)}")

        for file_name in os.listdir(os.path.join(checkpoint_path, "figures")):
            self.figure_saving_lst.append(os.path.join(self.fig_save_path, file_name))
            os.system(f"cp {os.path.join(checkpoint_path, 'figures', file_name)} "+\
                      f"{os.path.join(self.fig_save_path, file_name)}")

        self.placement_saving_lst = sorted(self.placement_saving_lst, key=lambda x:int(os.path.basename(x).split('_')[0]))
        self.figure_saving_lst = sorted(self.figure_saving_lst, key=lambda x:int(os.path.basename(x).split('_')[0]))
        


    @abstractmethod
    def _genotype2phenotype(self, x):
        raise NotImplementedError

    def __deepcopy__(self, memo=None):
        return self
    
