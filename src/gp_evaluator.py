
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
                 n_workers: int = 4) -> None:
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
        
        # 获取 verbose 设置，默认为 False
        verbose = getattr(args, 'verbose', False)
        
        # 初始化 Ray Actors
        # 注意：假设 Ray 已经在主进程中初始化 (ray.init())
        print(f"Initializing {n_workers} DREAMPlace Actors...")
        self.actors = [
            DREAMPlaceActor.remote(
                self.args_dict, 
                placedb.canvas_width, 
                placedb.canvas_height,
                verbose=verbose
            ) 
            for _ in range(n_workers)
        ]
        
        # 预热 Actors (可选，确保它们都加载完毕)
        # ray.get([actor.evaluate.remote({}) for actor in self.actors])
        print("DREAMPlace Actors initialized.")

    def evaluate(self, macro_pos: Dict[str, tuple]) -> float:
        """
        评估宏单元布局的 HPWL
        
        Args:
            macro_pos: 宏单元位置字典 {name: (x, y)}
            
        Returns:
            HPWL 值，失败返回 INF
        """
        if not macro_pos:
            return INF
        
        self.n_eval += 1
        
        # 简单的负载均衡：轮询
        actor = self.actors[self.n_eval % self.n_workers]
        
        try:
            # 同步等待结果
            hpwl = ray.get(actor.evaluate.remote(macro_pos))
            return float(hpwl)
        except Exception as e:
            print(f"Error in GPEvaluator evaluate: {e}")
            return INF
    def evaluate_batch(self, macro_pos_list: list) -> list:
        """
        批量评估多个布局（并行）
        
        Args:
            macro_pos_list: 布局列表
            
        Returns:
            HPWL 列表
        """
        futures = []
        for i, macro_pos in enumerate(macro_pos_list):
            actor = self.actors[i % self.n_workers]
            futures.append(actor.evaluate.remote(macro_pos))
        
        try:
            results = ray.get(futures)
            # 处理可能的 None 或异常值
            return [float(r) if r is not None else INF for r in results]
        except Exception as e:
            print(f"Error in batch evaluation: {e}")
            return [INF] * len(macro_pos_list)
    
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

    def save_placement_batch(self, 
                           macro_pos_list: List[Dict[str, tuple]],
                           placement_names: List[str]) -> List[bool]:
        """
        批量保存布局文件（并行）
        """
        if len(macro_pos_list) != len(placement_names):
            print("Error: macro_pos_list and placement_names must have same length")
            return [False] * len(macro_pos_list)

        futures = []
        for i, (macro_pos, name) in enumerate(zip(macro_pos_list, placement_names)):
            actor = self.actors[i % self.n_workers]
            output_dir = os.path.dirname(name)
            futures.append(actor.save_results.remote(
                macro_pos, 
                output_dir, 
                save_placement=True, 
                save_plot=False
            ))
        
        try:
            # 等待所有保存任务完成
            ray.get(futures)
            return [True] * len(macro_pos_list)
        except Exception as e:
            print(f"Error in batch save_placement: {e}")
            return [False] * len(macro_pos_list)
    
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

    def plot_batch(self, 
                  macro_pos_list: List[Dict[str, tuple]],
                  figure_names: List[str]) -> List[bool]:
        """
        批量生成布局可视化图（并行）
        """
        if len(macro_pos_list) != len(figure_names):
            print("Error: macro_pos_list and figure_names must have same length")
            return [False] * len(macro_pos_list)

        futures = []
        for i, (macro_pos, name) in enumerate(zip(macro_pos_list, figure_names)):
            actor = self.actors[i % self.n_workers]
            futures.append(actor.plot.remote(
                macro_pos, 
                name
            ))
        
        try:
            ray.get(futures)
            return [True] * len(macro_pos_list)
        except Exception as e:
            print(f"Error in batch plotting: {e}")
            return [False] * len(macro_pos_list)


    def __deepcopy__(self, memo=None):
        """
        防止深拷贝
        """
        return self
