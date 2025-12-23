"""
重构后的 HPOPlacer - 使用 Ray Actor 并行优化超参数
"""

import os
import math
import ray
from typing import Dict, Union, List
from src.placer.basic_placer import BasicPlacer
from src.utils.constant import EPS
from src.placer.dmp_actor import DREAMPlaceActor

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
                 eval_metrics: List[str] = ["hpwl", ],
                 n_workers: int = 4):
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
        self.n_workers = n_workers
        
        # 加载 DMP 配置
        self.params = DMPParams()
        self._load_dmp_config()
        
        # 转换 args 为字典
        self.args_dict = vars(args) if hasattr(args, '__dict__') else args
        
        # 初始化 Ray Actors
        print(f"Initializing {n_workers} DREAMPlace Actors for HPO...")
        self.actors = [
            DREAMPlaceActor.remote(
                self.args_dict, 
                placedb.canvas_width, 
                placedb.canvas_height
            ) 
            for _ in range(n_workers)
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

    def _genotype2phenotype(self, x: Union[List, Dict]) -> Dict[str, tuple]:
        """
        将基因型转换为表现型（宏单元位置）
        """
        # 将列表转换为字典
        if isinstance(x, (list, tuple)):
            params_name = list(params_space.keys())
            x = dict(zip(params_name, x))
        
        # 加载参数
        params_update = self._load_genotype(x)
        
        # 使用第一个 Actor 执行布局
        actor = self.actors[0]
        
        try:
            macro_pos = ray.get(actor.place.remote(
                params_update=params_update,
                macro_lst=self.placedb.macro_lst
            ))
            
            if macro_pos is None:
                return {}
            
            # 转换格式 (Actor 已经返回了正确的格式，这里做个保险)
            final_pos = {}
            for k, v in macro_pos.items():
                try:
                    final_pos[k] = (v[0], v[1])
                except:
                    final_pos[k] = tuple(v)
            
            return final_pos
        except Exception as e:
            print(f"Error in HPO placement: {e}")
            return {}

    def optimize_batch(self, genotypes: List) -> List[Dict]:
        """
        批量优化（并行）
        """
        # 并行提交所有任务
        futures = []
        for i, genotype in enumerate(genotypes):
            # 将列表转换为字典
            if isinstance(genotype, (list, tuple)):
                params_name = list(params_space.keys())
                genotype = dict(zip(params_name, genotype))
            
            params_update = self._load_genotype(genotype)
            actor = self.actors[i % self.n_workers]
            futures.append(actor.place.remote(
                params_update=params_update,
                macro_lst=self.placedb.macro_lst
            ))
        
        # 收集结果
        results = []
        try:
            batch_results = ray.get(futures)
            for macro_pos in batch_results:
                if macro_pos:
                    # 转换格式
                    final_pos = {}
                    for k, v in macro_pos.items():
                        try:
                            final_pos[k] = (v[0], v[1])
                        except:
                            final_pos[k] = tuple(v)
                    results.append(final_pos)
                else:
                    results.append({})
        except Exception as e:
            print(f"Batch optimization error: {e}")
            results = [{}] * len(genotypes)
        
        return results

    def __deepcopy__(self, memo=None):
        """防止深拷贝"""
        return self
    
