from abc import abstractmethod

import numpy as np
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
            header = ["n_eval"] + [f"{prefix}_{metric}" for prefix in ["his_best", "pop_best", "pop_avg", "pop_std"] for metric in self.eval_metrics] + \
                        ["t_each_eval", "avg_t_each_eval", "avg_t_algo_optimization", "avg_t_eval_solution"]
            writer.writerow(header)
        
        self.placement_saving_lst = []
        self.figure_saving_lst = []
        self.n_max_saving_placement = args.n_max_saving_placement
        if self.args.placer!="hpo":
            if 'gp_hpwl' in self.eval_metrics or args.eval_gp_hpwl :
                from gp_evaluator import GPEvaluator
                self.gp_evaluator = GPEvaluator(args=args,
                                                placedb=placedb)
            else:
                self.gp_evaluator = None
        else:
            self.gp_evaluator = None
        self.t_eval_solution_total = 0

    def _evaluate(self, x):
        # 单个评估逻辑，主要用于非批量场景或 fallback
        macro_pos = self._genotype2phenotype(x)
        res = comp_res(
            placedb=self.placedb,
            macros_pos=macro_pos,
            eval_metrics=self.eval_metrics,
            gp_evaluator=self.gp_evaluator  
        )
        return res, macro_pos
    
    
    def evaluate(self, x):
        t = time.time()
        
        # 1. 批量转换基因型到表现型
        macro_pos_list = [self._genotype2phenotype(x0) for x0 in x]
        
        # 2. 批量评估 GP HPWL (如果启用)
        gp_hpwl_results = []
        if self.gp_evaluator is not None and ('gp_hpwl' in self.eval_metrics or self.args.eval_gp_hpwl):
            # 使用 Actor 的批量评估接口
            gp_hpwl_results = self.gp_evaluator.evaluate_batch(macro_pos_list)
        
        # 3. 计算其他指标 (本地计算)
        results = []
        for i, macro_pos in enumerate(macro_pos_list):
            # 临时禁用 gp_evaluator 以避免 comp_res 再次调用它
            # 我们手动注入 gp_hpwl 结果
            res = comp_res(
                placedb=self.placedb,
                macros_pos=macro_pos,
                eval_metrics=[m for m in self.eval_metrics if m != 'gp_hpwl'],
                gp_evaluator=None 
            )
            
            if gp_hpwl_results:
                res['gp_hpwl'] = gp_hpwl_results[i]
            elif 'gp_hpwl' in self.eval_metrics:
                # 如果需要 gp_hpwl 但没有 evaluator (不应该发生)，设为 INF
                res['gp_hpwl'] = float('inf')
                
            results.append(res)

        t_eval_solution = time.time() - t
        self.t_eval_solution_total += t_eval_solution

        res = {}
        for eval_metric in self.eval_metrics:
            res[eval_metric] = np.array([result[eval_metric] for result in results])
        return res, macro_pos_list



    def save_placement(self, macro_pos, n_eval, hpwl):
        logging.info("Placer saving placement")
        scale_hpwl, n_power = get_n_power(hpwl)

        suffix_map = {
            "aux" : "pl",
            "def" : "def"
        }
        suffix = suffix_map[self.args.benchmark_type]
        file_name = os.path.join(self.placement_save_path, 
                                f'{n_eval}_{scale_hpwl:.2f}e{n_power}.{suffix}')
        
        if self.args.eval_gp_hpwl:
            self.gp_evaluator.save_placement(placement_name=file_name, macro_pos=macro_pos)
        else:
            type_map = {
                "aux" : write_pl,
                "def" : write_def
            }
            type_map[self.args.benchmark_type](file_name, macro_pos, self.placedb)
        
        self._manage_saved_files(self.placement_save_path, self.n_max_saving_placement)

    def plot(self, macro_pos:dict, n_eval:int, hpwl:float):
        logging.info("Placer ploting figure")
        scale_hpwl, n_power = get_n_power(hpwl)

        file_name = os.path.join(self.fig_save_path, f"{n_eval}_{scale_hpwl:.2f}e{n_power}.png")
        if self.args.eval_gp_hpwl:
            self.gp_evaluator.plot(figure_name=file_name, macro_pos=macro_pos)
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
            content = [n_eval] + [value for Y in [his_best_Y, pop_best_Y, pop_avg_Y, pop_std_Y] for value in Y] + [t_each_eval, avg_t_each_eval, avg_t_each_eval - avg_t_eval_solution, avg_t_eval_solution ]
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
        pass

    def __deepcopy__(self, memo=None):
        return self
