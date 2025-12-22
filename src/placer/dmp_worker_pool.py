"""
DREAMPlace Worker 进程池管理器
用于管理多个 dmp_worker 进程，实现并行布局评估
"""

import os
import time
import json
import socket
import signal
import subprocess
import threading
import queue
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
import select


@dataclass
class WorkerInfo:
    """Worker 信息"""
    worker_id: int
    process: subprocess.Popen
    sock: socket.socket
    sock_path: str
    is_busy: bool = False
    last_used: float = 0
    error_count: int = 0


class DMPWorkerPool:
    """DREAMPlace Worker 进程池"""
    
    MAX_RETRIES = 3
    WORKER_TIMEOUT = 300  # 5分钟超时
    INIT_TIMEOUT = 30
    
    def __init__(self, 
                 n_workers: int,
                 args,
                 placedb,
                 worker_path: str,
                 timeout_seconds: int = 300):
        """
        初始化进程池
        
        Args:
            n_workers: worker 进程数量
            args: 参数对象
            placedb: 布局数据库
            worker_path: worker 脚本路径
            timeout_seconds: 任务超时时间
        """
        self.n_workers = n_workers
        self.args = args
        self.placedb = placedb
        self.worker_path = worker_path
        self.timeout_seconds = timeout_seconds
        
        # Worker 管理
        self.workers: Dict[int, WorkerInfo] = {}
        self.available_workers = queue.Queue(maxsize=n_workers)
        self.lock = threading.RLock()
        
        # 任务队列
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        
        # 初始化所有 workers
        self._initialize_workers()
    
    def _initialize_workers(self):
        """初始化所有 worker 进程"""
        print(f"Initializing {self.n_workers} DMP workers...")
        
        for i in range(self.n_workers):
            worker_info = self._spawn_worker(i)
            if worker_info:
                self.workers[i] = worker_info
                self.available_workers.put(i)
            else:
                raise RuntimeError(f"Failed to initialize worker {i}")
        
        print(f"Successfully initialized {len(self.workers)} workers")
    
    def _spawn_worker(self, worker_id: int) -> Optional[WorkerInfo]:
        """
        启动单个 worker 进程
        
        Args:
            worker_id: Worker ID
            
        Returns:
            WorkerInfo 或 None（如果失败）
        """
        sock_path = os.path.join(
            self.args.ROOT_DIR,
            "sock_path",
            f"dmp_worker_pool_{self.args.unique_token}_{worker_id}.sock"
        )
        
        # 清理旧的 socket 文件
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        
        if True:
            # 启动 worker 进程
            process = subprocess.Popen(
                ["python3", self.worker_path, "--sock", sock_path],
                stdin=subprocess.DEVNULL,
                # stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL,
                cwd=self.args.ROOT_DIR,
                preexec_fn=os.setsid
            )
            
            # 连接 socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            deadline = time.time() + self.INIT_TIMEOUT
            connected = False
            
            while time.time() < deadline:
                try:
                    sock.connect(sock_path)
                    connected = True
                    break
                except Exception as e:
                    if process.poll() is not None:
                        # 进程已退出
                        break
                    time.sleep(0.05)
            
            if not connected:
                process.terminate()
                process.wait(timeout=5)
                return None
            
            sock.settimeout(self.timeout_seconds)
            
            # 发送初始化消息
            init_msg = {
                "cmd": "init",
                "args": {
                    "ROOT_DIR": self.args.ROOT_DIR,
                    "temp_subdir": "WorkerPool",
                    "name": getattr(self.args, "name", self.args.placer),
                    "benchmark": self.args.benchmark,
                    "benchmark_type": self.args.benchmark_type,
                    "unique_token": f"{self.args.unique_token}_w{worker_id}",
                    "seed": self.args.seed + worker_id,
                },
                "canvas_width": self.placedb.canvas_width,
                "canvas_height": self.placedb.canvas_height,
            }
            
            response = self._send_command(sock, init_msg)
            if not response or not response.get("ok"):
                sock.close()
                process.terminate()
                process.wait(timeout=5)
                return None
            
            return WorkerInfo(
                worker_id=worker_id,
                process=process,
                sock=sock,
                sock_path=sock_path,
                is_busy=False,
                last_used=time.time()
            )
            
        # except Exception as e:
        #     print(f"Error spawning worker {worker_id}: {e}")
            return None
    
    def _send_command(self, sock: socket.socket, cmd: dict) -> Optional[dict]:
        """
        向 worker 发送命令并接收响应
        
        Args:
            sock: Socket 连接
            cmd: 命令字典
            
        Returns:
            响应字典或 None
        """
        try:
            # 发送命令
            sock.sendall((json.dumps(cmd) + "\n").encode())
            
            # 接收响应
            data = b""
            deadline = time.time() + self.timeout_seconds
            
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                
                r, _, _ = select.select([sock], [], [], remaining)
                if not r:
                    return None
                
                chunk = sock.recv(65536)
                if not chunk:
                    break
                
                data += chunk
                if b"\n" in chunk:
                    break
            
            # 解析响应
            text = data.decode(errors="ignore")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                return None
            
            return json.loads(lines[-1])
            
        except Exception as e:
            print(f"Error sending command: {e}")
            return None
    
    def _get_worker(self, timeout: float = 60) -> Optional[int]:
        """
        获取可用的 worker
        
        Args:
            timeout: 等待超时时间
            
        Returns:
            Worker ID 或 None
        """
        try:
            worker_id = self.available_workers.get(timeout=timeout)
            with self.lock:
                if worker_id in self.workers:
                    self.workers[worker_id].is_busy = True
                    self.workers[worker_id].last_used = time.time()
            return worker_id
        except queue.Empty:
            return None
    
    def _release_worker(self, worker_id: int):
        """释放 worker"""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].is_busy = False
                self.available_workers.put(worker_id)
    
    def _restart_worker(self, worker_id: int) -> bool:
        """
        重启失败的 worker
        
        Args:
            worker_id: Worker ID
            
        Returns:
            是否成功
        """
        with self.lock:
            if worker_id in self.workers:
                old_worker = self.workers[worker_id]
                
                # 清理旧 worker
                try:
                    old_worker.sock.close()
                except:
                    pass
                
                try:
                    old_worker.process.terminate()
                    old_worker.process.wait(timeout=5)
                except:
                    try:
                        os.killpg(os.getpgid(old_worker.process.pid), signal.SIGKILL)
                    except:
                        pass
                
                if os.path.exists(old_worker.sock_path):
                    os.unlink(old_worker.sock_path)
                
                # 启动新 worker
                new_worker = self._spawn_worker(worker_id)
                if new_worker:
                    self.workers[worker_id] = new_worker
                    return True
                else:
                    del self.workers[worker_id]
                    return False
            return False
    
    def submit_eval_hpwl(self, macro_pos: Dict[str, tuple]) -> Future:
        """
        提交 HPWL 评估任务（异步）
        
        Args:
            macro_pos: 宏单元位置字典
            
        Returns:
            Future 对象
        """
        def task():
            return self.eval_hpwl(macro_pos)
        
        return self.executor.submit(task)
    
    def eval_hpwl(self, macro_pos: Dict[str, tuple]) -> Optional[float]:
        """
        评估 HPWL（同步）
        
        Args:
            macro_pos: 宏单元位置字典
            
        Returns:
            HPWL 值或 None
        """
        worker_id = self._get_worker()
        if worker_id is None:
            print("No available worker")
            return None
        
        try:
            worker = self.workers[worker_id]
            
            # 发送评估命令
            cmd = {
                "cmd": "eval_hpwl",
                "macro_pos": macro_pos
            }
            
            response = self._send_command(worker.sock, cmd)
            
            if response and response.get("ok") and response.get("hpwl") is not None:
                return response["hpwl"]
            else:
                # Worker 出错，尝试重启
                worker.error_count += 1
                if worker.error_count >= self.MAX_RETRIES:
                    print(f"Worker {worker_id} failed too many times, restarting...")
                    self._restart_worker(worker_id)
                return None
                
        except Exception as e:
            print(f"Error evaluating HPWL: {e}")
            return None
        finally:
            self._release_worker(worker_id)
    
    def submit_place(self, params_update: Dict = None, macro_lst: List[str] = None) -> Future:
        """
        提交布局任务（异步）
        
        Args:
            params_update: DMP 参数更新
            macro_lst: 宏单元列表
            
        Returns:
            Future 对象
        """
        def task():
            return self.place(params_update, macro_lst)
        
        return self.executor.submit(task)
    
    def place(self, params_update: Dict = None, macro_lst: List[str] = None) -> Optional[Dict]:
        """
        执行布局（同步）
        
        Args:
            params_update: DMP 参数更新
            macro_lst: 宏单元列表
            
        Returns:
            宏单元位置字典或 None
        """
        worker_id = self._get_worker()
        if worker_id is None:
            return None
        
        try:
            worker = self.workers[worker_id]
            
            cmd = {
                "cmd": "place",
                "params_update": params_update or {},
                "macro_lst": macro_lst or self.placedb.macro_lst
            }
            
            response = self._send_command(worker.sock, cmd)
            
            if response and response.get("ok"):
                return response.get("macro_pos")
            else:
                worker.error_count += 1
                if worker.error_count >= self.MAX_RETRIES:
                    self._restart_worker(worker_id)
                return None
                
        except Exception as e:
            print(f"Error placing: {e}")
            return None
        finally:
            self._release_worker(worker_id)
    
    def save_results(self, 
                     macro_pos: Dict[str, tuple],
                     output_dir: str,
                     save_placement: bool = True,
                     save_plot: bool = True) -> bool:
        """
        保存结果文件
        
        Args:
            macro_pos: 宏单元位置
            output_dir: 输出目录
            save_placement: 是否保存布局文件
            save_plot: 是否保存可视化图
            
        Returns:
            是否成功
        """
        worker_id = self._get_worker()
        if worker_id is None:
            return False
        
        try:
            worker = self.workers[worker_id]
            
            cmd = {
                "cmd": "save_results",
                "macro_pos": macro_pos,
                "output_dir": output_dir,
                "save_placement": save_placement,
                "save_plot": save_plot
            }
            
            response = self._send_command(worker.sock, cmd)
            return response and response.get("ok", False)
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
        finally:
            self._release_worker(worker_id)
    
    def shutdown(self):
        """关闭所有 worker"""
        print("Shutting down worker pool...")
        
        with self.lock:
            for worker_id, worker in self.workers.items():
                try:
                    worker.sock.close()
                except:
                    pass
                
                try:
                    worker.process.terminate()
                    worker.process.wait(timeout=5)
                except:
                    try:
                        os.killpg(os.getpgid(worker.process.pid), signal.SIGKILL)
                    except:
                        pass
                
                if os.path.exists(worker.sock_path):
                    try:
                        os.unlink(worker.sock_path)
                    except:
                        pass
        
        self.executor.shutdown(wait=True)
        print("Worker pool shut down")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False