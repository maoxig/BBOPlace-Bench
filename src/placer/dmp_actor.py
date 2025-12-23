
import os
import sys
import time
import torch as th
import numpy as np
import ray
import logging
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 引入项目路径以便加载配置
sys.path.append(os.path.abspath("."))
from config.benchmark import benchmark_dict

# 尝试导入 DREAMPlace

import thirdparty.dreamplace.ops.place_io.place_io as place_io
from thirdparty.dreamplace.Params import Params as DMPParams
from thirdparty.dreamplace.PlaceDB import PlaceDB as DMPPlaceDB
from thirdparty.dreamplace.NonLinearPlace import NonLinearPlace
import thirdparty.dreamplace.Timer as Timer


@ray.remote(num_cpus=1, num_gpus=0.2) # 默认每个Actor占用0.2个GPU，可根据显存调整
class DREAMPlaceActor:
    def __init__(self, args_dict, canvas_width, canvas_height, verbose=False):
        """
        初始化 DREAMPlace Actor
        args_dict: 包含 args 的字典 (避免传递复杂对象)
        """
        self.args_dict = args_dict
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.verbose = verbose
        
        # 初始化环境
        sys.path.append(os.path.join(args_dict["ROOT_DIR"], "thirdparty"))
        sys.path.append(os.path.join(args_dict["ROOT_DIR"], "thirdparty", "dreamplace"))
        
        # 设置日志级别
        if not verbose:
            logging.getLogger().setLevel(logging.ERROR)
        
        if DMPParams is None:
            raise ImportError("DREAMPlace not found!")

        self.params = DMPParams()
        self.placedb = DMPPlaceDB()
        # 设置参数并加载 DB
        self._setup_inputs(self.args_dict)
        self.placedb(self.params)

        # 预处理节点名称 (用于后续可能的映射)
        self.node_names = self.placedb.node_names.astype(np.str_)
        mask = np.char.find(self.node_names, "DREAMPlace") != -1
        modified = np.char.split(self.node_names[mask], '.').tolist()
        self.node_names[mask] = [n[0] for n in modified]
        self.placer = NonLinearPlace(self.params, self.placedb, timer=None)

        self.reset_cache()

    def reset_cache(self):
        self.cached_data = {
            "placement": None,
            "figure": None,}

    def evaluate_macro_pos(self, macro_pos):
        """
        评估 HPWL
        macro_pos: {macro_name: (x, y)}
        """

        # 重置缓存
        self.reset_cache()
        # 1. 更新宏单元位置
        self._update_macro_pos(macro_pos)
        
        # 2. 运行评估
        metrics = self.placer(self.params, self.placedb)
      
        try:
            hpwl = metrics[-1].hpwl.cpu().item()
        except:
            hpwl = metrics[-1].hpwl.item()
       
        #print(metrics)
        # 确保返回 float
        if isinstance(hpwl, list) or isinstance(hpwl, tuple):
             hpwl = hpwl[0]
        self.cached_data["placement"] = (self.placedb.node_x.copy(), self.placedb.node_y.copy())
        self.cached_data["figure"] = np.copy(self.placer.pos[0].data.clone().cpu().numpy())
        return float(hpwl)

    def evaluate_hyper_params(self, params_update: dict, macro_lst: list):
        if isinstance(params_update, dict):
            self.params.fromJson(params_update)
        with th.no_grad():
            self.placer.pos[0].data.copy_(
                th.from_numpy(self.placer._initialize_position(self.params, self.placedb)).to(self.placer.device)
            )
        metrics = self.placer(self.params, self.placedb)

        macro_pos = self.placedb.export(self.params, macro_lst)
        for node_name in list(macro_pos.keys()):
            x = macro_pos[node_name][0] / (self.placedb.xh - self.placedb.xl) * self.canvas_width
            y = macro_pos[node_name][1] / (self.placedb.yh - self.placedb.yl) * self.canvas_height
            macro_pos[node_name] = [x, y]

        return macro_pos

    def evaluate_timing(self):
        self.timer = Timer.Timer()
        self.timer(self.params, self.placedb)
        # This must be done to explicitly execute the parser builders.
        # The parsers in OpenTimer are all in lazy mode.
        self.timer.update_timing()
        timing_op = self.placer.op_collections.timing_op
        time_unit = timing_op.timer.time_unit()

        # Perform timing analysis on current placement
        # The timing operator takes the current position as input
        timing_op(self.placer.pos[0].data.clone().cpu())
        timing_op.timer.update_timing()

        # Report TNS and WNS
        # Note: OpenTimer considers early,late,rise,fall for tns/wns
        # The following values are normalized by time units
        tns = timing_op.timer.report_tns_elw(split=1) / (time_unit * 1e17)
        wns = timing_op.timer.report_wns(split=1) / (time_unit * 1e15)
        
        return float(tns), float(wns)

    def plot(self, figure_name):
        """
        @brief plot layout
        @param macro_pos locations of cells
        @param figname output figure name
        """
        
        os.makedirs(os.path.dirname(figure_name), exist_ok=True)
        
        pos = self.cached_data["figure"]
        self.placer.plot(
            self.params,
            None,
            None,
            pos,
            figure_name, 
        )

        img = Image.open(figure_name)
        out = img.transpose(Image.FLIP_TOP_BOTTOM) # type: ignore
        img.close()
        out.save(figure_name)
        return True
    
    def save_results(self, macro_pos, output_dir, save_placement=True, save_plot=True):
        """
        保存结果
        """
        return True

    def _update_macro_pos(self, macro_pos):
        """
        更新 DB 中的宏单元位置
        """
        if self.args_dict["benchmark_type"] == "aux":
            dmp_scale_factor_x, dmp_scale_factor_y = 1.0, 1.0
        else:
            # (xh - xl) / canvas_width
            dmp_scale_factor_x = (self.placedb.xh - self.placedb.xl) / self.canvas_width # type: ignore
            dmp_scale_factor_y = (self.placedb.yh - self.placedb.yl)/ self.canvas_height # type: ignore

        for macro, pos in macro_pos.items():
            # 使用 numpy 查找索引，与 dmp_worker.py 保持一致
            index = np.where(self.node_names == macro)
            if len(index[0]) == 0:
                continue
                
            # 使用 round
            pos_x = round(pos[0] * dmp_scale_factor_x)
            pos_y = round(pos[1] * dmp_scale_factor_y)
            
            self.placedb.node_x[index] = pos_x
            self.placedb.node_y[index] = pos_y
        node_x, node_y = self.placedb.unscale_pl(self.params.shift_factor, self.params.scale_factor)
        place_io.PlaceIOFunction.apply(self.placedb.rawdb, node_x, node_y, all_movable=True)

        with th.no_grad():
            self.placer.pos[0].data.copy_(
                th.from_numpy(self.placer._initialize_position(self.params, self.placedb)).to(self.placer.device)
            )

    

    def _setup_inputs(self, args_dict):
        root_dir = args_dict["ROOT_DIR"]
        benchmark = args_dict["benchmark"]
        benchmark_type = args_dict["benchmark_type"]
        
        # 加载 JSON 配置
        json_path = os.path.join(
            root_dir,
            "config/algorithm/dmp_config",
            f"{benchmark}.json",
        )
        self.params.load(json_path)
        ##logging.info(f"Loaded DMP config from {json_path}")
        # 确定 benchmark 文件夹
        benchmark_folder = None
        for key, value in benchmark_dict.items():
            if benchmark in value:
                benchmark_folder = key
                break
        
        benchmark_path = os.path.join(
            root_dir,
            "benchmarks",
            benchmark_folder,
            benchmark,
        )

        def suffix2path(suffix: str) -> str:
            return os.path.join(benchmark_path, f"{benchmark}") + suffix

        if benchmark_type == "aux":
            self.params.fromJson({
                "aux_input": suffix2path(".aux")
            })
        elif benchmark_type == "def":
            self.params.fromJson({
                "def_input": suffix2path(".def"),
                "lef_input": suffix2path(".lef"),
                "verilog_input": suffix2path(".v"),
                "early_lib_input": suffix2path("_Early.lib"),
                "late_lib_input": suffix2path("_Late.lib"),
                "sdc_input": suffix2path(".sdc"),
            })
        
        # 设置输出目录和其他参数
        self.params.fromJson({
            "plot_flag": 0,
            "timing_opt_flag": 0,
            "random_seed": args_dict["seed"],
            "result_dir": os.path.join(
                root_dir,
                "results",
                benchmark,
                f"{args_dict['name']}",
                f"{args_dict['unique_token']}",
                "dmp_results",
            ),
            "random_center_init_flag": 1,
            "figure_path": None
        })
