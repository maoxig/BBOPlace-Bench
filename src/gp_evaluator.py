"""
重构后的 GPEvaluator - 使用进程池并行评估
"""

import os
import numpy as np
from typing import Dict, Optional
from utils.constant import INF
from PIL import Image
from dmp_worker_pool import DMPWorkerPool

try:
    from thirdparty.dreamplace.Params import Params as DMPParams
except ImportError:
    DMPParams = None


class GPEvaluator:
    """全局布局评估器 - 使用进程池"""
    
    DMP_CONFIG_PATH = "config/algorithm/dmp_config"
    
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
        
        # 加载 DMP 配置
        self.dmp_params = DMPParams()
        self._load_dmp_config()
        
        # 准备基准测试环境
        self._prepare_benchmark()
        
        # 初始化进程池
        worker_path = os.path.join(self.args.SOURCE_DIR, "placer/dmp_worker.py")
        self.worker_pool = DMPWorkerPool(
            n_workers=n_workers,
            args=args,
            placedb=placedb,
            worker_path=worker_path,
            timeout_seconds=args.timeout_seconds
        )
        
        # 数据缓存
        self.empty_saving_data()
        self.n_eval = 0
    
    def evaluate(self, macro_pos: Dict[str, tuple]) -> float:
        """
        评估宏单元布局的 HPWL
        
        Args:
            macro_pos: 宏单元位置字典 {name: (x, y)}
            
        Returns:
            HPWL 值，失败返回 INF
        """
        if len(macro_pos) == 0:
            return INF
        
        self.n_eval += 1
        
        # 通过进程池评估
        hpwl = self.worker_pool.eval_hpwl(macro_pos)
        
        if hpwl is None:
            return INF
        
        return hpwl
    
    def evaluate_batch(self, macro_pos_list: list) -> list:
        """
        批量评估多个布局（并行）
        
        Args:
            macro_pos_list: 布局列表
            
        Returns:
            HPWL 列表
        """
        # 并行提交所有任务
        futures = []
        for macro_pos in macro_pos_list:
            future = self.worker_pool.submit_eval_hpwl(macro_pos)
            futures.append(future)
        
        # 收集结果
        results = []
        for future in futures:
            try:
                hpwl = future.result(timeout=self.args.timeout_seconds)
                results.append(hpwl if hpwl is not None else INF)
            except Exception as e:
                print(f"Batch evaluation error: {e}")
                results.append(INF)
        
        return results
    
    def save_placement(self, 
                      macro_pos: Dict[str, tuple],
                      placement_name: str) -> bool:
        """
        保存布局文件
        
        Args:
            macro_pos: 宏单元位置
            placement_name: 输出文件名（不含扩展名）
            
        Returns:
            是否成功
        """
        output_dir = os.path.dirname(placement_name)
        os.makedirs(output_dir, exist_ok=True)
        
        return self.worker_pool.save_results(
            macro_pos=macro_pos,
            output_dir=output_dir,
            save_placement=True,
            save_plot=False
        )
    
    def plot(self, 
            macro_pos: Dict[str, tuple],
            figure_name: str) -> bool:
        """
        生成布局可视化图
        
        Args:
            macro_pos: 宏单元位置
            figure_name: 输出图片文件名
            
        Returns:
            是否成功
        """
        output_dir = os.path.dirname(figure_name)
        os.makedirs(output_dir, exist_ok=True)
        
        success = self.worker_pool.save_results(
            macro_pos=macro_pos,
            output_dir=output_dir,
            save_placement=False,
            save_plot=True
        )
        
        if success:
            # 翻转图像（与原实现保持一致）
            try:
                img = Image.open(figure_name)
                out = img.transpose(Image.FLIP_TOP_BOTTOM)
                img.close()
                out.save(figure_name)
            except Exception as e:
                print(f"Error flipping image: {e}")
                return False
        
        return success
    
    def _load_dmp_config(self):
        """加载 DREAMPlace 配置"""
        ROOT_DIR = self.args.ROOT_DIR
        json_file = f"{self.args.benchmark}.json"
        json_path = os.path.join(
            ROOT_DIR,
            self.DMP_CONFIG_PATH,
            json_file
        )
        self.dmp_params.load(json_path)
        self.dmp_params.benchmark = self.args.benchmark
        self.dmp_params.random_center_init_flag = 1
    
    def _prepare_benchmark(self):
        """准备基准测试环境"""
        # 注意：进程池架构下不需要准备临时文件
        # Worker 会直接接收 macro_pos 数据
        pass
    
    def empty_saving_data(self):
        """清空保存的数据"""
        self.saving_data = {
            "placement": {},
            "figure": {}
        }
    
    def __deepcopy__(self, memo=None):
        """防止深拷贝"""
        return self
    
    def __del__(self):
        """析构函数 - 清理资源"""
        if hasattr(self, 'worker_pool'):
            self.worker_pool.shutdown()


# 使用示例
if __name__ == "__main__":
    # 创建评估器（4个并行 worker）
    evaluator = GPEvaluator(args, placedb, n_workers=4)
    
    # 单个评估
    macro_pos = {"macro1": (100, 200), "macro2": (300, 400)}
    hpwl = evaluator.evaluate(macro_pos)
    print(f"HPWL: {hpwl}")
    
    # 批量并行评估
    macro_pos_list = [
        {"macro1": (100, 200), "macro2": (300, 400)},
        {"macro1": (150, 250), "macro2": (350, 450)},
        {"macro1": (200, 300), "macro2": (400, 500)},
    ]
    hpwl_list = evaluator.evaluate_batch(macro_pos_list)
    print(f"Batch HPWL: {hpwl_list}")
    
    # 保存结果
    evaluator.save_placement(macro_pos, "output/placement")
    evaluator.plot(macro_pos, "output/layout.png")
    
    # 清理
    del evaluator