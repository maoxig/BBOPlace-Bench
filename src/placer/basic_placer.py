from abc import abstractmethod

import numpy as np
from src.placer.dmp_actor import DREAMPlaceActor
from src.utils.debug import *
from src.utils.compute_res import comp_res
from src.utils.read_benchmark.read_aux import write_pl
from src.utils.read_benchmark.read_def import write_def
from src.utils.constant import INF, get_n_power

from typing import overload

import os
import csv
import ray
import time
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

@ray.remote(num_cpus=1, num_gpus=0.1)
def evaluate_placer(placer: 'BasicPlacer', x0):
    return placer._evaluate(x0)

@ray.remote(num_cpus=1, num_gpus=0.1)
def save_placement_remote(placer: 'BasicPlacer', macro_pos, n_eval):
    placer.save_placement(macro_pos, n_eval)


@ray.remote(num_cpus=1, num_gpus=0.1)
def plot_placement_remote(placer: 'BasicPlacer', macro_pos, n_eval):
    placer.plot(macro_pos, n_eval)

class BasicPlacer:
    def __init__(self, args, placedb, eval_metrics= ["hpwl"]) -> None:
        self.args = args
        self.placedb = placedb
        self.eval_metrics = eval_metrics

        self.canvas_width  = placedb.canvas_width
        self.canvas_height = placedb.canvas_height
        
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
        # TODO，怎么画图？
        if self.args.eval_gp_hpwl:
            self.gp_evaluator = DREAMPlaceActor.remote(
                vars(self.args), 
                placedb.canvas_width, 
                placedb.canvas_height
            ) 
        else:
            self.gp_evaluator = None
    def _evaluate(self, x):
        # 单个评估逻辑，主要用于非批量场景或 fallback
        res = {}
        macro_pos, info = self._genotype2phenotype(x)
        res = comp_res(macros_pos=macro_pos, placedb=self.placedb, eval_metrics=self.eval_metrics)
        
        if self.args.placer == 'hpo':
            res.update(info) # gp_hpwl, or tns, or wns
        else:
            gp_res = {}
            if self.args.eval_gp_hpwl:
                gp_res = ray.get(self.gp_evaluator.evaluate_macro_pos.remote(macro_pos))
            res.update(gp_res)
            
        # 初始化评估器，根据需求评估，并且返回对应指标
        # 总共4种情况：
        # 1. macro placer，仅评估macro级别指标
        # 2. macro placer + gp evaluator， 仅gp级别指标
        # 3. macro placer，评估macro 级别指标 + gp级别指标
        # 4. global placer， 仅评估macro级别指标
        # 5. global placer + gp evaluator，仅评估gp级别指标
        # 6. global placer + gp evaluator，评估macro 级别指标 + gp级别指标

        return res, macro_pos
    
    
    def evaluate(self, x):
        t = time.time()
        futures = [evaluate_placer.remote(self, x0) for x0 in x]
        results = ray.get(futures) # {eval_metric: value, ...}, macro_pos
        #print(results)
        t_eval_solution = time.time() - t
        self.t_eval_solution_total += t_eval_solution

        res = {}
        for eval_metric in self.eval_metrics:
            res[eval_metric] = np.array([result[0][eval_metric] for result in results])
        macro_pos_list = [result[1] for result in results]
        return res, macro_pos_list

    def save_placement_batch(self, 
                            macro_pos_list: list,
                            n_eval_list: list) -> bool:
        """
        批量保存多个布局文件
        """
        futures = []
        for macro_pos, n_eval in zip(macro_pos_list, n_eval_list):
            futures.append(save_placement_remote.remote(self, macro_pos, n_eval))
        try:
            ray.get(futures)
            return True 
        except Exception as e:
            print(f"Error in batch save placement: {e}")
            return False


    def save_placement(self, macro_pos, n_eval):
        logging.info("Placer saving placement")

        suffix_map = {
            "aux" : "pl",
            "def" : "def"
        }
        suffix = suffix_map[self.args.benchmark_type]
        file_name = os.path.join(self.placement_save_path, 
                                f'{n_eval}.{suffix}')
        
        if self.args.eval_gp_hpwl:
            ray.get(self.gp_evaluator.save_placement.remote(placement_name=file_name))
        else:
            type_map = {
                "aux" : write_pl,
                "def" : write_def
            }
            type_map[self.args.benchmark_type](file_name, macro_pos, self.placedb)
        
        self._manage_saved_files(self.placement_save_path, self.n_max_saving_placement)

    def plot_batch(self, macro_pos_list: list, n_eval_list: list) -> bool:
        futures = []
        for macro_pos, n_eval in zip(macro_pos_list, n_eval_list):
            futures.append(plot_placement_remote.remote(self, macro_pos, n_eval))
        try:
            ray.get(futures)
            return True
        except Exception as e:
            print(f"Error in batch plot: {e}")
            return False
    
    def plot(self, macro_pos:dict, n_eval:int):
        logging.info("Placer plotting figure")

        file_name = os.path.join(self.fig_save_path, f"{n_eval}.png")
        if self.args.eval_gp_hpwl:
            ray.get(self.gp_evaluator.plot.remote(figure_name=file_name))
        else:
            self._plot_macro(macro_pos, file_name)

        self._manage_saved_files(self.fig_save_path, self.n_max_saving_placement)

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

    def _load_checkpoint(self, checkpoint_path):
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
