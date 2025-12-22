"""
重构后的 HPOPlacer - 使用进程池并行优化超参数
"""

import os
import math
from typing import Dict, Union, List
from src.placer.basic_placer import BasicPlacer
from src.utils.constant import EPS
from src.placer.dmp_worker_pool import DMPWorkerPool

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
    """超参数优化 Placer - 使用进程池"""
    
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
        
        # 初始化进程池
        worker_path = os.path.join(self.args.SOURCE_DIR, "placer/dmp_worker.py")
        self.worker_pool = DMPWorkerPool(
            n_workers=1,
            args=args,
            placedb=placedb,
            worker_path=worker_path,
            timeout_seconds=args.timeout_seconds
        )

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
        
        Args:
            x: 超参数字典
            
        Returns:
            DMP 参数更新字典
        """
        params_to_update = {}
        
        for param_name, value in x.items():
            lb, ub, tf = params_space[param_name]
            assert lb - EPS < value < ub + EPS, \
                f"Parameter {param_name} (={value}) is out of bound [{lb}, {ub}]."
            
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
        
        Args:
            x: 超参数（列表或字典）
            
        Returns:
            宏单元位置字典 {name: (x, y)} 或空字典（失败）
        """
        # 将列表转换为字典
        if isinstance(x, (list, tuple)):
            params_name = list(params_space.keys())
            x = dict(zip(params_name, x))
        
        # 加载参数
        params_update = self._load_genotype(x)
        
        # 通过进程池执行布局
        macro_pos = self.worker_pool.place(
            params_update=params_update,
            macro_lst=self.placedb.macro_lst
        )
        
        if macro_pos is None:
            return {}
        
        # 转换格式
        for k, v in list(macro_pos.items()):
            try:
                macro_pos[k] = (v[0], v[1])
            except:
                macro_pos[k] = tuple(v)
        
        return macro_pos

    def optimize_batch(self, genotypes: List) -> List[Dict]:
        """
        批量优化（并行）
        
        Args:
            genotypes: 超参数列表
            
        Returns:
            宏单元位置列表
        """
        # 并行提交所有任务
        futures = []
        for genotype in genotypes:
            # 将列表转换为字典
            if isinstance(genotype, (list, tuple)):
                params_name = list(params_space.keys())
                genotype = dict(zip(params_name, genotype))
            
            params_update = self._load_genotype(genotype)
            future = self.worker_pool.submit_place(
                params_update=params_update,
                macro_lst=self.placedb.macro_lst
            )
            futures.append(future)
        
        # 收集结果
        results = []
        for future in futures:
            try:
                macro_pos = future.result(timeout=self.args.timeout_seconds)
                if macro_pos:
                    # 转换格式
                    for k, v in list(macro_pos.items()):
                        try:
                            macro_pos[k] = (v[0], v[1])
                        except:
                            macro_pos[k] = tuple(v)
                    results.append(macro_pos)
                else:
                    results.append({})
            except Exception as e:
                print(f"Batch optimization error: {e}")
                results.append({})
        
        return results

    def __deepcopy__(self, memo=None):
        """防止深拷贝"""
        return self
    
    def __del__(self):
        """析构函数 - 清理资源"""
        if hasattr(self, 'worker_pool'):
            self.worker_pool.shutdown()


# 使用示例
if __name__ == "__main__":
    # 创建 HPO Placer（8个并行 worker）
    
    placer = HPOPlacer(args, placedb, n_workers=8)
    
    # 单个参数配置优化
    genotype = [1.5, 2.0, 0.005, 0.995, 0.95, 1.1, 300000, 5e-5, 2.5, 0.08, 1.0]
    macro_pos = placer._genotype2phenotype(genotype)
    print(f"Macro positions: {macro_pos}")
    
    # 批量并行优化
    genotypes = [
        [1.5, 2.0, 0.005, 0.995, 0.95, 1.1, 300000, 5e-5, 2.5, 0.08, 1.0],
        [2.0, 2.5, 0.006, 0.990, 0.93, 1.12, 350000, 6e-5, 3.0, 0.07, 1.1],
        [1.0, 1.5, 0.004, 0.998, 0.97, 1.08, 280000, 4e-5, 2.0, 0.09, 0.9],
    ]
    results = placer.optimize_batch(genotypes)
    print(f"Batch results: {len(results)} configurations")
    
    # 清理
    del placer