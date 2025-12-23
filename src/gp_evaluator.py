
"""
重构后的 GPEvaluator - 使用 Ray Actor 并行评估
"""

import os
import ray
import numpy as np
from typing import Dict, List
from utils.constant import INF
from src.placer.dmp_actor import DREAMPlaceActor

class GPEvaluator:
    """全局布局评估器 - 使用 Ray Actor"""
    
    def __init__(self, 
                 args, 
                 placedb,
                 n_workers: int = 5) -> None:
        """
        初始化评估器
        
        Args:
            args: 参数对象
            placedb: 布局数据库
            n_workers: worker 进程数量
        """
        self.args = args
        self.placedb = placedb
        self.n_workers = n_workers
        self.n_eval = 0
        
        # 转换 args 为字典，确保可序列化且解耦
        self.args_dict = vars(args) if hasattr(args, '__dict__') else args
        
    def evaluate(self, macro_pos: Dict[str, tuple]) -> float:
        """
        评估宏单元布局的 HPWL
        
        Args:
            macro_pos: 宏单元位置字典 {name: (x, y)}
            
        Returns:
            HPWL 值，失败返回 INF
        """

    
    def save_placement(self, 
                      macro_pos: Dict[str, tuple],
                      placement_name: str) -> bool:
        """
        保存单个布局文件
        """
        output_dir = os.path.dirname(placement_name)
        actor = self.actors[0]
        
        try:
            ray.get(actor.save_results.remote(
                macro_pos, 
                output_dir, 
                save_placement=True, 
                save_plot=False
            ))
            return True
        except Exception as e:
            print(f"Error saving placement: {e}")
            return False

    
    def plot(self, 
            macro_pos: Dict[str, tuple],
            figure_name: str) -> bool:
        """
        生成单个布局可视化图
        """
        actor = self.actors[0]
        
        try:
            ray.get(actor.plot.remote(
                macro_pos, 
                figure_name
            ))
            return True
        except Exception as e:
            print(f"Error plotting: {e}")
            return False


    def __deepcopy__(self, memo=None):
        """
        防止深拷贝
        """
        return self
