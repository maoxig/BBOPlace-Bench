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
    """超参数优化 Placer - 使用 Ray Actor"""
    
    DMP_CONFIG_PATH = "config/algorithm/dmp_config"
    DMP_RESULT_DIR = os.path.join(
        "results",
        "%(name)s",
        "%(benchmark)s",
        "%(unique_token)s",
        "dmp_results"
    )

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
        self.n_workers = 4
        
        # 加载 DMP 配置
        self.params = DMPParams()
        self._load_dmp_config()
        
        # 转换 args 为字典
        self.args_dict = vars(args) if hasattr(args, '__dict__') else args
        
        # 初始化 Ray Actors
        print(f"Initializing {self.n_workers } DREAMPlace Actors for HPO...")
        self.actors = [
            DREAMPlaceActor.remote(
                self.args_dict, 
                placedb.canvas_width, 
                placedb.canvas_height
            ) 
            for _ in range(self.n_workers)
        ]
        

    @property
    def param_dims(self) -> int:
        """参数维度"""
        return len(params_space.items())

    @property
    def _result_dir(self) -> str:
        """结果目录"""
        ROOT_DIR = self.args.ROOT_DIR
        return os.path.join(
            ROOT_DIR,
            self.DMP_RESULT_DIR % self.args.__dict__
        )

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
                print(f"Warning: Parameter {param_name} (={value}) is out of bound [{lb}, {ub}].")
            
            param_value = tf(value)
            
            if param_name.startswith("GP_"):
                # Global placement 阶段参数
                params_to_update.setdefault("global_place_stages", [{}])
                subject = params_to_update["global_place_stages"][0]
                entry_name = param_name.lstrip("GP_")
            else:
                # 其他参数
                subject = params_to_update
                entry_name = param_name
            
            subject[entry_name] = param_value
        
        return params_to_update

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
        并行评估种群 - 使用 Actor Pool
        """
        t_start = time.time()
        
        # 1. 分发任务给 Actors
        futures = []
        for i, xi in enumerate(x):
            actor = self.actors[i % len(self.actors)]
            
            # 转换参数
            params_name = list(params_space.keys())
            xi_list = list(xi)
            xi_dict = dict(zip(params_name, xi_list))
            
            params_update = self._load_genotype(xi_dict)
            
            futures.append(actor.evaluate_hyper_params.remote(
                params_update=params_update,
                macro_lst=self.placedb.macro_lst
            ))
            
        # 2. 获取结果
        results = ray.get(futures)
        
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
            
            # 合并 Actor 返回的指标 (gp_hpwl, tns, wns)
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
        
        return final_results, macro_pos_list

    def save_placement(self, macro_pos, n_eval):
        """
        覆盖 BasicPlacer 的 save_placement，使用 HPO 的 actor
        """
        logging.info("HPO Placer saving placement")
        
        suffix_map = {
            "aux" : "pl",
            "def" : "def"
        }
        suffix = suffix_map[self.args.benchmark_type]
        file_name = os.path.join(self.placement_save_path, 
                                f'{n_eval}.{suffix}')
        
        # 使用轮询方式选择 actor 进行保存，均衡负载
        actor_idx = n_eval % len(self.actors)
        ray.get(self.actors[actor_idx].save_placement.remote(file_name))
        
        self._manage_saved_files(self.placement_save_path, self.n_max_saving_placement)

    def plot(self, macro_pos:dict, n_eval:int):
        """
        覆盖 BasicPlacer 的 plot，使用 HPO 的 actor
        """
        logging.info("HPO Placer plotting figure")

        file_name = os.path.join(self.fig_save_path, f"{n_eval}.png")
        
        actor_idx = n_eval % len(self.actors)
        ray.get(self.actors[actor_idx].plot.remote(file_name))

        self._manage_saved_files(self.fig_save_path, self.n_max_saving_placement)

    def __deepcopy__(self, memo=None):
        """防止深拷贝"""
        return self
    
