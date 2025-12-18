"""
增强版 DMP Worker
支持进程池模式，新增命令：update_macro_pos, save_results
"""

import json
import sys
import os
import torch as th
import socket
import numpy as np
import signal
import time
import threading
import psutil

try:
    import thirdparty.dreamplace.ops.place_io.place_io as place_io
    from thirdparty.dreamplace.Params import Params as DMPParams
    from thirdparty.dreamplace.PlaceDB import PlaceDB as DMPPlaceDB
    from thirdparty.dreamplace.NonLinearPlace import NonLinearPlace
except ImportError as e:
    place_io = None
    DMPParams = None
    DMPPlaceDB = None
    NonLinearPlace = None


def _setup_inputs(params: DMPParams, args: dict):
    """设置输入文件路径"""
    json_path = os.path.join(
        args["ROOT_DIR"],
        "config/algorithm/dmp_config",
        f"{args['benchmark']}.json",
    )
    params.load(json_path)

    temp_subdir = args.get("temp_subdir", "WorkerPool")
    temp_benchmark_path = os.path.join(
        args["ROOT_DIR"],
        f"benchmarks/.tmp/{temp_subdir}",
        f"{args['benchmark']}_{args['unique_token']}",
    )

    def suffix2path(suffix: str) -> str:
        return os.path.join(temp_benchmark_path, f"{args['benchmark']}") + suffix

    if args["benchmark_type"] == "aux":
        params.fromJson({
            "aux_input": suffix2path(".aux")
        })
    elif args["benchmark_type"] == "def":
        params.fromJson({
            "def_input": suffix2path(".def"),
            "lef_input": suffix2path(".lef"),
            "verilog_input": suffix2path(".v"),
            "early_lib_input": suffix2path("_Early.lib"),
            "late_lib_input": suffix2path("_Late.lib"),
            "sdc_input": suffix2path(".sdc"),
        })
    else:
        raise RuntimeError("Unsupported benchmark_type")

    params.fromJson({
        "plot_flag": 0,
        "timing_opt_flag": 0,
        "random_seed": args["seed"],
        "result_dir": os.path.join(
            args["ROOT_DIR"],
            "results",
            f"{args['name']}",
            f"{args['benchmark']}",
            f"{args['unique_token']}",
            "dmp_results",
        ),
        "random_center_init_flag": 1,
    })


class DMPWorker:
    """DREAMPlace Worker - 增强版"""
    
    def __init__(self):
        self.init = False
        self.args = None
        self.canvas_width = None
        self.canvas_height = None

        self.params = DMPParams()
        self.placedb = DMPPlaceDB()
        self.placer = None

        self.node_names = None
        self.server = None
        self.sock_path = None
        
        self._parent_pid = os.getppid()
        self._monitor_thread = None
    
    def _init_data(self, args: dict, canvas_width: float, canvas_height: float):
        """初始化数据"""
        self.args = args
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        _setup_inputs(self.params, self.args)

        self.placedb(self.params)
        self.placer = NonLinearPlace(self.params, self.placedb, timer=None)

        # 缓存节点名称
        self.node_names = self.placedb.node_names.astype('U')
        mask = np.char.find(self.node_names, "DREAMPlace") != -1
        modified = np.char.split(self.node_names[mask], '.').tolist()
        self.node_names[mask] = [n[0] for n in modified]

        self.init = True

    def _start_parent_monitor(self):
        """启动父进程监控线程"""
        def cleanup_worker():
            if self.server:
                self.server.close()
            if self.sock_path is not None and os.path.exists(self.sock_path):
                os.unlink(self.sock_path)
            os._exit(0)
        
        def monitor_parent():
            while True:
                try:
                    if not psutil.pid_exists(self._parent_pid):
                        cleanup_worker()
                    
                    try:
                        parent = psutil.Process(self._parent_pid)
                        if parent.create_time() != psutil.Process(os.getppid()).create_time():
                            cleanup_worker()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        cleanup_worker()
                        
                except Exception:
                    cleanup_worker()
                    
                time.sleep(1)
        
        self._monitor_thread = threading.Thread(target=monitor_parent, daemon=True)
        self._monitor_thread.start()

    def handle_place(self, msg: dict) -> dict:
        """
        处理布局命令
        
        Args:
            msg: 包含 params_update 和 macro_lst 的消息
            
        Returns:
            包含 macro_pos 的响应
        """
        params_update = msg.get("params_update")
        if isinstance(params_update, dict):
            self.params.fromJson(params_update)
        
        with th.no_grad():
            self.placer.pos[0].data.copy_(
                th.from_numpy(self.placer._initialize_position(self.params, self.placedb)).to(self.placer.device)
            )
        
        _ = self.placer(self.params, self.placedb)
        macro_pos = self.placedb.export(self.params, msg["macro_lst"])
        
        # 转换坐标
        for node_name in list(macro_pos.keys()):
            x = macro_pos[node_name][0] / (self.placedb.xh - self.placedb.xl) * self.canvas_width
            y = macro_pos[node_name][1] / (self.placedb.yh - self.placedb.yl) * self.canvas_height
            macro_pos[node_name] = [x, y]
        
        return {"ok": True, "macro_pos": macro_pos}

    def handle_eval_hpwl(self, msg: dict) -> dict:
        """
        处理 HPWL 评估命令
        
        Args:
            msg: 包含 macro_pos 的消息
            
        Returns:
            包含 hpwl 的响应
        """
        macro_pos = msg["macro_pos"]
        
        # 计算缩放因子
        if self.args["benchmark_type"] == "aux":
            dmp_scale_factor_x, dmp_scale_factor_y = 1, 1
        else:
            dmp_scale_factor_x = self.placedb.xh / self.canvas_width
            dmp_scale_factor_y = self.placedb.yh / self.canvas_height
        
        # 更新宏单元位置
        for macro, pos in macro_pos.items():
            index = np.where(self.node_names == macro)
            pos_x = round(pos[0] * dmp_scale_factor_x)
            pos_y = round(pos[1] * dmp_scale_factor_y)
            self.placedb.node_x[index] = pos_x
            self.placedb.node_y[index] = pos_y
        
        # 更新 PlaceDB
        node_x, node_y = self.placedb.unscale_pl(self.params.shift_factor, self.params.scale_factor)
        place_io.PlaceIOFunction.apply(self.placedb.rawdb, node_x, node_y, all_movable=True)

        # 初始化位置并评估
        with th.no_grad():
            self.placer.pos[0].data.copy_(
                th.from_numpy(self.placer._initialize_position(self.params, self.placedb)).to(self.placer.device)
            )
        
        metrics = self.placer(self.params, self.placedb)
        
        # 提取结果
        try:
            hpwl = metrics[-1].hpwl.cpu().item()
        except Exception:
            hpwl = None
        
        try:
            pos_arr = self.placer.pos[0].data.detach().cpu().numpy().tolist()
        except Exception:
            pos_arr = None
        
        try:
            node_x_list = self.placedb.node_x.tolist()
            node_y_list = self.placedb.node_y.tolist()
        except Exception:
            node_x_list, node_y_list = None, None
        
        return {
            "ok": True,
            "hpwl": hpwl,
            "pos": pos_arr,
            "node_x": node_x_list,
            "node_y": node_y_list,
        }

    def handle_update_macro_pos(self, msg: dict) -> dict:
        """
        新命令：更新宏单元位置（不执行布局）
        
        Args:
            msg: 包含 macro_pos 的消息
            
        Returns:
            成功响应
        """
        macro_pos = msg["macro_pos"]
        
        # 计算缩放因子
        if self.args["benchmark_type"] == "aux":
            dmp_scale_factor_x, dmp_scale_factor_y = 1, 1
        else:
            dmp_scale_factor_x = self.placedb.xh / self.canvas_width
            dmp_scale_factor_y = self.placedb.yh / self.canvas_height
        
        # 更新位置
        for macro, pos in macro_pos.items():
            index = np.where(self.node_names == macro)
            pos_x = round(pos[0] * dmp_scale_factor_x)
            pos_y = round(pos[1] * dmp_scale_factor_y)
            self.placedb.node_x[index] = pos_x
            self.placedb.node_y[index] = pos_y
        
        # 更新 PlaceDB
        node_x, node_y = self.placedb.unscale_pl(self.params.shift_factor, self.params.scale_factor)
        place_io.PlaceIOFunction.apply(self.placedb.rawdb, node_x, node_y, all_movable=True)
        
        # 更新 placer 位置
        with th.no_grad():
            self.placer.pos[0].data.copy_(
                th.from_numpy(self.placer._initialize_position(self.params, self.placedb)).to(self.placer.device)
            )
        
        return {"ok": True, "action": "updated_macro_pos"}

    def handle_save_results(self, msg: dict) -> dict:
        """
        新命令：保存结果文件
        
        Args:
            msg: 包含 macro_pos, output_dir, save_placement, save_plot
            
        Returns:
            成功响应
        """
        macro_pos = msg["macro_pos"]
        output_dir = msg["output_dir"]
        save_placement = msg.get("save_placement", True)
        save_plot = msg.get("save_plot", True)
        
        # 首先更新宏单元位置
        self.handle_update_macro_pos({"macro_pos": macro_pos})
        
        # 保存布局文件
        if save_placement:
            try:
                os.makedirs(output_dir, exist_ok=True)
                placement_name = os.path.join(output_dir, "placement")
                self.placedb.write(self.params, placement_name)
            except Exception as e:
                return {"ok": False, "error": f"Failed to save placement: {str(e)}"}
        
        # 保存可视化图
        if save_plot:
            try:
                os.makedirs(output_dir, exist_ok=True)
                figure_name = os.path.join(output_dir, "layout.png")
                pos = self.placer.pos[0].data.clone().cpu().numpy()
                self.placer.plot(
                    self.params,
                    None,
                    None,
                    pos,
                    figure_name,
                )
            except Exception as e:
                return {"ok": False, "error": f"Failed to save plot: {str(e)}"}
        
        return {"ok": True, "action": "saved_results", "output_dir": output_dir}

    def handle_update_params(self, msg: dict) -> dict:
        """
        新命令：更新 DMP 参数
        
        Args:
            msg: 包含 params_update 的消息
            
        Returns:
            成功响应
        """
        params_update = msg.get("params_update")
        if isinstance(params_update, dict):
            self.params.fromJson(params_update)
            return {"ok": True, "action": "updated_params"}
        else:
            return {"ok": False, "error": "Invalid params_update"}

    def serve_forever(self, sock_path: str):
        """启动服务器，持续处理请求"""
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        self.sock_path = sock_path
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(sock_path)
        self.server.listen(1)

        self._start_parent_monitor()
        
        while True:
            try:
                conn, _ = self.server.accept()
            except Exception:
                continue
            
            with conn:
                buffer = b""
                while True:
                    try:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buffer += chunk
                        
                        while b"\n" in buffer:
                            line_bytes, buffer = buffer.split(b"\n", 1)
                            line = line_bytes.decode(errors="ignore").strip()
                            if not line:
                                continue
                            
                            try:
                                msg = json.loads(line)
                            except Exception as e:
                                try:
                                    conn.sendall((json.dumps({"ok": False, "error": str(e)}) + "\n").encode())
                                except Exception:
                                    pass
                                continue

                            try:
                                cmd = msg.get("cmd")
                                
                                if cmd == "init":
                                    if not self.init:
                                        args = msg.get("args")
                                        canvas_width = msg.get("canvas_width")
                                        canvas_height = msg.get("canvas_height")
                                        self._init_data(args, canvas_width, canvas_height)
                                        resp = {"ok": True, "action": "init"}
                                    else:
                                        resp = {"ok": True, "action": "already_init"}
                                
                                elif cmd == "place":
                                    resp = self.handle_place(msg)
                                
                                elif cmd == "eval_hpwl":
                                    resp = self.handle_eval_hpwl(msg)
                                
                                elif cmd == "update_macro_pos":
                                    resp = self.handle_update_macro_pos(msg)
                                
                                elif cmd == "save_results":
                                    resp = self.handle_save_results(msg)
                                
                                elif cmd == "update_params":
                                    resp = self.handle_update_params(msg)
                                
                                elif cmd == "ping":
                                    resp = {"ok": True, "status": "alive"}
                                
                                else:
                                    resp = {"ok": False, "error": f"unknown cmd: {cmd}"}
                            
                            except Exception as e:
                                resp = {"ok": False, "error": str(e)}
                            
                            try:
                                conn.sendall((json.dumps(resp) + "\n").encode())
                            except Exception:
                                break
                    except Exception:
                        break


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--sock":
        sock_path = sys.argv[2]
        worker = DMPWorker()
        worker.serve_forever(sock_path)
    else:
        print("Usage: python dmp_worker.py --sock <socket_path>")
        sys.exit(1)