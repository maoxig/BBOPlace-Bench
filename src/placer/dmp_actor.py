
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

from src.utils.constant import INF

# 引入项目路径以便加载配置
sys.path.append(os.path.abspath("."))
from config.benchmark import benchmark_dict

# 尝试导入 DREAMPlace

import thirdparty.dreamplace.ops.place_io.place_io as place_io
from thirdparty.dreamplace.Params import Params as DMPParams
from thirdparty.dreamplace.PlaceDB import PlaceDB as DMPPlaceDB
from thirdparty.dreamplace.NonLinearPlace import NonLinearPlace
import thirdparty.dreamplace.Timer as Timer


@ray.remote
class DREAMPlaceActor:
    def __init__(self, args_dict, canvas_width, canvas_height, temp_benchmark_path, verbose=False):
        """
        初始化 DREAMPlace Actor
        args_dict: 包含 args 的字典
        temp_benchmark_path: 临时 benchmark 路径 (用于 HPO)
        """
        # 重定向输出

        self.args_dict = args_dict
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.temp_benchmark_path = temp_benchmark_path
        self.verbose = verbose
        
        # 初始化环境
        sys.path.append(os.path.join(args_dict["ROOT_DIR"], "thirdparty"))
        sys.path.append(os.path.join(args_dict["ROOT_DIR"], "thirdparty", "dreamplace"))
        
        # 设置日志级别
        if not verbose:
            # Redirect Python stdout/stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            logging.getLogger().setLevel(logging.ERROR)
            
            # Redirect C-level stdout/stderr (for DREAMPlace C++ ops)
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                os.close(devnull)
            except Exception as e:
                # Fallback or ignore if redirection fails (e.g. in some restricted envs)
                pass
        
        if DMPParams is None:
            raise ImportError("DREAMPlace not found!")


        self.params = DMPParams()
        self.placedb = DMPPlaceDB()
        self._setup_inputs(self.args_dict)
        self.placedb(self.params)

        if "n_wns" in [m.lower() for m in self.args_dict.get("eval_metrics", [])] or \
           "n_tns" in [m.lower() for m in self.args_dict.get("eval_metrics", [])]:
            self.eval_timing = True
            self.timer = Timer.Timer()
            self.timer(self.params, self.placedb)
            self.timer.update_timing()
        else:
            self.eval_timing = False
            self.timer = None
        self.placer = NonLinearPlace(self.params, self.placedb, timer=self.timer)
        # cache node_names for evaluator
        self.node_names = self.placedb.node_names.astype('U')
        mask = np.char.find(self.node_names, "DREAMPlace") != -1
        modified = np.char.split(self.node_names[mask], '.').tolist()
        self.node_names[mask] = [n[0] for n in modified]


    def evaluate_macro_pos(self, macro_pos, placement_file=None, figure_file=None):
        """
        评估 HPWL
        macro_pos: {macro_name: (x, y)}
        """
        if not macro_pos or len(macro_pos) == 0:
            return {
                "macro_pos": {},
                "gp_hpwl": INF
            }
        # 1. 更新宏单元位置
        self._update_macro_pos(macro_pos)
        if self.timer:
            self.timer(self.params, self.placedb)
            self.timer.update_timing()
        self.placer = NonLinearPlace(self.params, self.placedb, timer=self.timer)
        self._update_dmp_placer()
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
        if placement_file:
            self.save_placement(placement_file)
        if figure_file:
            self.plot(figure_file)

        result ={ 
            "macro_pos": macro_pos,
            "gp_hpwl": float(hpwl)
        }
        if self.eval_timing:
            timing_res = self.evaluate_timing()
            result.update(timing_res)
        return result

    def evaluate_hyper_params(self, params_update: dict, macro_lst: list, placement_file=None, figure_file=None):
        
        if isinstance(params_update, dict):
            self.params.fromJson(params_update)
            
        with th.no_grad():
            self.placer.pos[0].data.copy_(
                th.from_numpy(self.placer._initialize_position(self.params, self.placedb)).to(self.placer.device)
            )
        metrics = self.placer(self.params, self.placedb)
        
        if placement_file:
            self.save_placement(placement_file)
        if figure_file:
            self.plot(figure_file)
        
        # 处理 macro_lst 可能存在的名称不匹配问题 (bytes vs str)
        if macro_lst and len(macro_lst) > 0:
            sample_macro = macro_lst[0]
            if sample_macro not in self.placedb.node_name2id_map:
                # 尝试检测 map 中的 key 类型
                first_key = next(iter(self.placedb.node_name2id_map))
                if isinstance(first_key, bytes) and isinstance(sample_macro, str):
                    macro_lst = [m.encode('utf-8') for m in macro_lst]
                elif isinstance(first_key, str) and isinstance(sample_macro, bytes):
                    macro_lst = [m.decode('utf-8') for m in macro_lst]
        
        macro_pos = self.placedb.export(self.params, macro_lst)
                    
        for node_name in list(macro_pos.keys()):
            x = macro_pos[node_name][0] / (self.placedb.xh - self.placedb.xl) * self.canvas_width
            y = macro_pos[node_name][1] / (self.placedb.yh - self.placedb.yl) * self.canvas_height
            macro_pos[node_name] = [x, y]

        result = {
            "macro_pos": macro_pos,
            "gp_hpwl": float(metrics[-1].hpwl.cpu().item())
        }
        if self.eval_timing:
            timing_res = self.evaluate_timing()
            result.update(timing_res)
        return result

    def evaluate_timing(self):
        timing_op = self.placer.op_collections.timing_op
        time_unit = timing_op.timer.time_unit()

        # Perform timing analysis on current placement
        # The timing operator takes the current position as input
        pos_data = self.placer.pos[0].data.clone().cpu()
        timing_op(pos_data)
        timing_op.timer.update_timing()

        # Report TNS and WNS
        # Note: OpenTimer considers early,late,rise,fall for tns/wns
        # The following values are normalized by time units
        tns = timing_op.timer.report_tns_elw(split=1) / (time_unit * 1e17)
        wns = timing_op.timer.report_wns(split=1) / (time_unit * 1e15)
        
        result = {
            "n_tns":  -float(tns),
            "n_wns":  -float(wns)
        }
        return result

    def plot(self, figure_name):
        """
        @brief plot layout
        @param macro_pos locations of cells
        @param figname output figure name
        """
        os.makedirs(os.path.dirname(figure_name), exist_ok=True)
        # Convert numpy array back to tensor for plot function
        pos_tensor = self.placer.pos[0].data.clone().cpu()# RuntimeError: !pos.is_cuda() INTERNAL ASSERT FAILED at "dreamplace/ops/draw_place/src/draw_place.cpp":39, please report a bug to PyTorch. pos must be a tensor on CPU
        
        self.placer.plot(
            self.params,
            self.placedb,
            None,
            pos_tensor,
            figure_name, 
        )
        try:
            img = Image.open(figure_name)
            out = img.transpose(Image.FLIP_TOP_BOTTOM) # type: ignore
            img.close()
            out.save(figure_name)
        except Exception as e:
            print(f"Error processing image {figure_name}: {e}")
            
        return True
    
    def save_placement(self, placement_name):

        # unscale locations
        node_x, node_y = self.placedb.unscale_pl(self.params.shift_factor, 
                                                     self.params.scale_factor)
        # update raw database
        place_io.PlaceIOFunction.apply(self.placedb.rawdb, node_x, node_y, all_movable=True)

        self.placedb.write(
            self.params, 
            placement_name
        )
        return True

    def _update_macro_pos(self, macro_pos):
        if self.args_dict["benchmark_type"] == "aux":
            dmp_scale_factor_x, dmp_scale_factor_y = 1.0, 1.0
        else:
            # (xh - xl) / canvas_width
            dmp_scale_factor_x = (self.placedb.xh ) / self.canvas_width # type: ignore
            dmp_scale_factor_y = (self.placedb.yh )/ self.canvas_height # type: ignore

        for macro, pos in macro_pos.items():
            index = np.where(self.node_names == macro)
            pos_x = round(pos[0] * dmp_scale_factor_x)
            pos_y = round(pos[1] * dmp_scale_factor_y)
            self.placedb.node_x[index] = pos_x
            self.placedb.node_y[index] = pos_y
        node_x, node_y = self.placedb.unscale_pl(self.params.shift_factor, self.params.scale_factor)
        place_io.PlaceIOFunction.apply(self.placedb.rawdb, node_x, node_y, all_movable=True)
        self.placedb.initialize_from_rawdb(self.params)
        self.placedb.initialize(self.params)
        #self._update_dmp_placer()


    def _update_dmp_placer(self):
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
        benchmark_path = self.temp_benchmark_path


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
