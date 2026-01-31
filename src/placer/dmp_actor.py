
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
import gc
import resource
from src.utils.constant import INF

# 引入项目路径以便加载配置
sys.path.append(os.path.abspath("."))

import thirdparty.dreamplace.ops.place_io.place_io as place_io
from thirdparty.dreamplace.Params import Params as DMPParams
from thirdparty.dreamplace.PlaceDB import PlaceDB as DMPPlaceDB
from thirdparty.dreamplace.NonLinearPlace import NonLinearPlace
import thirdparty.dreamplace.Timer as Timer
import thirdparty.dreamplace.EvalMetrics as EvalMetrics
import thirdparty.dreamplace.PlaceObj as PlaceObj
import thirdparty.dreamplace.ops.rudy.rudy as rudy
import thirdparty.dreamplace.ops.pin_utilization.pin_utilization as pin_utilization

@ray.remote(max_restarts=-1)
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
                pass
        
        self.params = DMPParams()
        self.placedb = DMPPlaceDB()
        self._setup_inputs(self.args_dict)
        
        

        if "n_wns" in [m.lower() for m in self.args_dict.get("eval_metrics", [])] or \
           "n_tns" in [m.lower() for m in self.args_dict.get("eval_metrics", [])]:
            self.eval_timing = True
        else:
            self.eval_timing = False
        
        self.placer = None 
        if self.args_dict.get('placer', "") == 'hpo': # prepare everything
            self.placedb(self.params)
            self.__init_placer()
        else:
            self.placedb.read(self.params) # only read rawdb and initialize pydb since we will modify rawdb and pydb
    
        # cache node_names for evaluator
        self.node_names = self.placedb.node_names.astype("U")
        self.node_names = np.char.replace(self.node_names, ".DREAMPlace.Shape0", "")

    def __init_placer(self):
        if self.eval_timing:
            timer = Timer.Timer()
            timer(self.params, self.placedb)
            timer.update_timing()
        else:
            timer = None
            
        self.placer = NonLinearPlace(self.params, self.placedb, timer=timer)
        

    def evaluate_macro_pos(self, macro_pos, placement_file=None, figure_file=None, save_result = False):
        """
        评估 HPWL
        macro_pos: {macro_name: (x, y)}
        """
        if not macro_pos or len(macro_pos) == 0:
            return {
                "macro_pos": {},
                "gp_hpwl": INF
            }
        self._setup_dmp_scale_factor()
        self._update_macro_pos(macro_pos)
        if self.placer is None:
            self.__init_placer()

        self._update_dmp_placer()
        # 2. 运行评估
        metrics = self.placer(self.params, self.placedb)
      
        if save_result:
            if placement_file:
                self.save_placement(placement_file)
            if figure_file:
                self.plot(figure_file)
        
        hpwl = metrics[-1].hpwl.cpu().item()

        #print(metrics)
        if isinstance(hpwl, list) or isinstance(hpwl, tuple):
             hpwl = hpwl[0]
        if save_result:
            if placement_file:
                self.save_placement(placement_file)
            if figure_file:
                self.plot(figure_file)

        result ={ 
            "macro_pos": macro_pos,
            "gp_hpwl": float(hpwl)
        }
        
        # Evaluate additional metrics (density, overflow, route_utilization)
        extra_metrics = self._evaluate_metrics_from_placer()
        result.update(extra_metrics)

        if self.eval_timing:
            timing_res = self.evaluate_timing()
            result.update(timing_res)
        return result

    def evaluate_hyper_params(self, params_update: dict, macro_lst: list, placement_file=None, figure_file=None, save_result = False):
        
        if isinstance(params_update, dict):
            self.params.fromJson(params_update)

        self._update_dmp_placer()
        
        metrics = self.placer(self.params, self.placedb)
        
        if save_result:
            if placement_file:
                self.save_placement(placement_file)
            if figure_file:
                self.plot(figure_file)
        
        # 处理 macro_lst 可能存在的名称不匹配问题 (bytes vs str)
        if macro_lst and len(macro_lst) > 0:
            sample_macro = macro_lst[0]
            if sample_macro not in self.placedb.node_name2id_map:
                first_key = next(iter(self.placedb.node_name2id_map))
                if isinstance(first_key, bytes) and isinstance(sample_macro, str):
                    macro_lst = [m.encode('utf-8') for m in macro_lst]
                elif isinstance(first_key, str) and isinstance(sample_macro, bytes):
                    macro_lst = [m.decode('utf-8') for m in macro_lst]
        
        macro_pos = self.export_macro_pos(macro_lst)
                    
        for node_name in list(macro_pos.keys()):
            x = macro_pos[node_name][0] / (self.placedb.xh - self.placedb.xl) * self.canvas_width
            y = macro_pos[node_name][1] / (self.placedb.yh - self.placedb.yl) * self.canvas_height
            macro_pos[node_name] = [x, y]

        result = {
            "macro_pos": macro_pos,
            "gp_hpwl": float(metrics[-1].hpwl.cpu().item())
        }

        # Evaluate additional metrics
        extra_metrics = self._evaluate_metrics_from_placer()
        result.update(extra_metrics)

        if self.eval_timing:
            timing_res = self.evaluate_timing()
            result.update(timing_res)

        return result

    def evaluate_timing(self):
        timing_op = self.placer.op_collections.timing_op
        time_unit = timing_op.timer.time_unit()

        # Perform timing analysis on current placement
        # The timing operator takes the current position as input
        #pos_data = self.placer.pos[0].data.clone().cpu()
        timing_op(self.placer.pos[0].data.cpu())
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

    def _evaluate_metrics_from_placer(self):
        """
        Evaluate additional metrics (density, overflow, route_utilization) from the current placer state.
        This is called after placement has run, so the position and data collections are populated.
        """
        if self.placer is None:
            return {}
        
        # Ensure ops exist, particularly for routability if not enabled globally but requested
        if self.placer.op_collections.density_op is None:
             # This might happen if density weight was 0 and optimization skipped density? Unlikely in DMP logic.
             # But if routability op is missing and we want it, we need to build it.
             pass

        routability_needed = False
        if self.args_dict.get("eval_metrics", []) and "route_utilization" in self.args_dict.get("eval_metrics", []):
            routability_needed = True
        
        pin_utilization_needed = False
        if self.args_dict.get("eval_metrics", []) and "pin_utilization" in self.args_dict.get("eval_metrics", []):
            pin_utilization_needed = True
        
        if routability_needed and self.placer.op_collections.pin_pos_op is None:
             try:
                 import thirdparty.dreamplace.ops.pin_pos.pin_pos as pin_pos
                 self.placer.op_collections.pin_pos_op = pin_pos.PinPos(
                     pin_offset_x=self.placer.data_collections.pin_offset_x,
                     pin_offset_y=self.placer.data_collections.pin_offset_y,
                     pin2node_map=self.placer.data_collections.pin2node_map,
                     flat_node2pin_map=self.placer.data_collections.flat_node2pin_map,
                     flat_node2pin_start_map=self.placer.data_collections.flat_node2pin_start_map,
                     num_physical_nodes=self.placedb.num_movable_nodes,
                     algorithm="segment"
                 ).to(self.placer.data_collections.pos[0].device)
             except Exception as e:
                 print(f"Warning: Failed to build PinPos op manually: {e}")

        if routability_needed and self.placer.op_collections.route_utilization_map_op is None:
             # Manually build RUDY op if needed, instead of re-instantiating PlaceObj
             try:
                 params = self.params
                 placedb = self.placedb
                 data_collections = self.placer.data_collections
                 
                 self.placer.op_collections.route_utilization_map_op = rudy.Rudy(
                    netpin_start=data_collections.flat_net2pin_start_map,
                    flat_netpin=data_collections.flat_net2pin_map,
                    net_weights=data_collections.net_weights,
                    xl=placedb.xl,
                    xh=placedb.xh,
                    yl=placedb.yl,
                    yh=placedb.yh,
                    num_bins_x=placedb.num_routing_grids_x,
                    num_bins_y=placedb.num_routing_grids_y,
                    unit_horizontal_capacity=float(placedb.unit_horizontal_capacity),
                    unit_vertical_capacity=float(placedb.unit_vertical_capacity),
                    deterministic_flag=params.deterministic_flag if hasattr(params, 'deterministic_flag') else True,
                    initial_horizontal_utilization_map=None,
                    initial_vertical_utilization_map=None
                 ).to(data_collections.pos[0].device)
             except Exception as e:
                 print(f"Warning: Failed to build RUDY op manually: {e}")

        if pin_utilization_needed and self.placer.op_collections.pin_utilization_map_op is None:
             # Manually build Pin Utilization op
             try:
                params = self.params
                placedb = self.placedb
                data_collections = self.placer.data_collections

                self.placer.op_collections.pin_utilization_map_op = pin_utilization.PinUtilization(
                    node_size_x=data_collections.node_size_x,
                    node_size_y=data_collections.node_size_y,
                    pin_weights=data_collections.pin_weights,
                    flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
                    xl=placedb.xl,
                    yl=placedb.yl,
                    xh=placedb.xh,
                    yh=placedb.yh,
                    num_movable_nodes=placedb.num_movable_nodes,
                    num_filler_nodes=placedb.num_filler_nodes,
                    num_bins_x=placedb.num_routing_grids_x,
                    num_bins_y=placedb.num_routing_grids_y,
                    unit_pin_capacity=data_collections.unit_pin_capacity,
                    pin_stretch_ratio=params.pin_stretch_ratio if hasattr(params, 'pin_stretch_ratio') else 1.414213562,
                    deterministic_flag=params.deterministic_flag if hasattr(params, 'deterministic_flag') else True,
                ).to(data_collections.pos[0].device)
             except Exception as e:
                 print(f"Warning: Failed to build Pin Utilization op manually: {e}")

        ops = {
            # "hpwl": self.placer.op_collections.hpwl_op, # Already extracted from metrics
            "density": self.placer.op_collections.density_op,
            "overflow": self.placer.op_collections.density_overflow_op,
        }
        
        if self.placer.op_collections.route_utilization_map_op:
            if self.placer.op_collections.pin_pos_op:
                def route_utilization_wrapper(pos):
                    # Compute pin pos from node pos
                    pin_pos = self.placer.op_collections.pin_pos_op(pos)
                    return self.placer.op_collections.route_utilization_map_op(pin_pos)
                ops["route_utilization"] = route_utilization_wrapper
            else:
                ops["route_utilization"] = self.placer.op_collections.route_utilization_map_op
        
        if self.placer.op_collections.pin_utilization_map_op:
            ops["pin_utilization"] = self.placer.op_collections.pin_utilization_map_op

        metric = EvalMetrics.EvalMetrics()
        # Evaluate metrics on current position (self.placer.pos[0])
        metric.evaluate(self.placedb, ops, self.placer.pos[0], self.placer.data_collections)
        
        results = {
            "density": float(metric.density) if metric.density is not None and metric.density.numel() == 1 else None,
            "overflow": float(metric.overflow) if metric.overflow is not None else None,
        }
        
        if metric.route_utilization is not None:
             results["route_utilization"] = float(metric.route_utilization)
        
        if metric.pin_utilization is not None:
             results["pin_utilization"] = float(metric.pin_utilization)
        
        return results

    def plot(self, figure_name):
        """
        @brief plot layout
        @param macro_pos locations of cells
        @param figname output figure name
        """
        os.makedirs(os.path.dirname(figure_name), exist_ok=True)
        # Convert numpy array back to tensor for plot function
        pos_tensor = self.placer.pos[0].data.cpu()# RuntimeError: !pos.is_cuda() INTERNAL ASSERT FAILED at "dreamplace/ops/draw_place/src/draw_place.cpp":39, please report a bug to PyTorch. pos must be a tensor on CPU
        
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
            dmp_scale_factor_x = (self.placedb.xh - self.placedb.xl ) / self.canvas_width # type: ignore
            dmp_scale_factor_y = (self.placedb.yh - self.placedb.yl) / self.canvas_height # type: ignore

        for i, (macro, pos) in enumerate(macro_pos.items()):
            index = np.where(self.node_names == macro)
            assert len(index[0]) == 1, f"({i+1}/{len(macro_pos)}) Macro {macro} not found in placedb"
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

    def _setup_dmp_scale_factor(self):
        # shift and scale
        # adjust shift_factor and scale_factor if not set
        self.params.shift_factor[0] = self.placedb.xl
        self.params.shift_factor[1] = self.placedb.yl
        
        if self.params.scale_factor == 0.0 or self.placedb.site_width != 1.0:
            self.params.scale_factor = 1.0 / self.placedb.site_width
        
        self.placedb.scale(self.params.shift_factor, self.params.scale_factor)

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
        elif self.args_dict.get("benchmark_type", "") == "openroad_def" and self.args_dict.get("placer", "") == "hpo":
            self.params.def_input = self.params.def_input.replace("replace.def","raw.def") # raw.def: all free ; replace.def: with macros fixed
        elif self.args_dict.get("benchmark_type", "") == "openroad_def" and self.args_dict.get("placer", "") != "hpo":
            self.params.def_input = self.params.def_input.replace("raw.def","replace.def") # raw.def: all free ; replace.def: with macros fixed
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

    def export_macro_pos(self, macro_lst: list):
        node_x, node_y = self.placedb.node_x, self.placedb.node_y
        macro_pos = {}
        for macro_name in macro_lst:
            id = self.placedb.node_name2id_map.get(macro_name, None)
            if id is None:
                id = self.placedb.node_name2id_map.get(macro_name.replace(".DREAMPlace.Shape0",""))
            macro_pos[macro_name] = (node_x[id], node_y[id])
        return macro_pos
    
