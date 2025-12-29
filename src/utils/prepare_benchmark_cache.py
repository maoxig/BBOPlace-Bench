import argparse
import pickle
import sys
from matplotlib.pylab import f
import numpy as np
import os
import copy
import math
import re
from pathlib import Path
import time
import logging
import torch as th

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIRDPARTY_DIR = os.path.join(ROOT_DIR, "thirdparty")
SOURCE_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(ROOT_DIR)
sys.path.append(THIRDPARTY_DIR)
sys.path.append(SOURCE_DIR)
os.environ["PYTHONPATH"] = ":".join(sys.path)
from src.utils.read_benchmark.read_def import get_inv_scaling_ratio, get_scaling_ratio
from thirdparty.dreamplace.Params import Params
from thirdparty.dreamplace.PlaceDB import PlaceDB
from thirdparty.dreamplace.NonLinearPlace import NonLinearPlace
import thirdparty.dreamplace.ops.place_io.place_io as place_io
import thirdparty.dreamplace.Timer as Timer

from itertools import combinations
import pdb
from PIL import Image
import networkit as nk
from networkit import graphio
def to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


parser = argparse.ArgumentParser(description='prepare parser')
parser.add_argument("--benchmark", required=True, type=str)
parser.add_argument("--dataset", required=True, type=str, help="choose from ['ispd2005', 'iccad2015']")
parser.add_argument("--benchmark_path", type=str, default="")
#parser.add_argument("--ROOT_DIR", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
parser.add_argument("--n_macro", type=int, default=512)
parser.add_argument("--benchmark_type", type=str, default="def", help="choose from ['aux', 'def']")
parser.add_argument("--gamma", type=float, default=0.95, help="gamma for PLM community detection")
args = parser.parse_args()

class ProblemInstance():
    def __init__(self, args, benchmark, macro_placedb : 'MacroPlaceDB'):
        self.benchmark = benchmark
        self.args = args


        self.database = {}
        self.dmp_params = Params()
        self._setup_inputs(args.__dict__)
        self.dmp_placedb = PlaceDB()
        self.dmp_placedb.read(self.dmp_params)

        self.timer = Timer.Timer()
        self.timer(self.dmp_params, self.dmp_placedb)
        self.timer.update_timing()
        self.results = {}

        self.max_width = self.dmp_placedb.xh - self.dmp_placedb.xl
        self.max_height = self.dmp_placedb.yh - self.dmp_placedb.yl
        self.num_movable_nodes = self.dmp_placedb.num_movable_nodes

        # 检查这里的max_width和max_height
        # 对比macro placedb里面的canvas_width和canvas_height
        print(f"max_width: {self.max_width}, max_height: {self.max_height}")
        print(f"canvas_width: {macro_placedb.canvas_width}, canvas_height: {macro_placedb.canvas_height}")
        scale_x, ratio_y = get_scaling_ratio(macro_placedb.database)
        print(f"scaling ratio_x: {scale_x}, ratio_y: {ratio_y}")
        inv_ratio_x, inv_ratio_y = get_inv_scaling_ratio(macro_placedb.database)
        print(f"inv scaling ratio_x: {inv_ratio_x}, ratio_y: {inv_ratio_y}")


        # TODO: Macro Info: 
        self.macro_names = macro_placedb.macro_lst.copy()
        # TODO: Calculate average node area
        self.port_indices = []     
        total_area = 0

        self.macro_ids = []
        self.port_pos = []
        for node_name in self.macro_names:
            # Try direct lookup first
            node_id = self.dmp_placedb.node_name2id_map.get(node_name, -1)
            
            # If still failed, try appending .DREAMPlace.Shape0 suffix
            if node_id == -1:
                suffix = ".DREAMPlace.Shape0"
                node_name_with_suffix = node_name + suffix
                node_id = self.dmp_placedb.node_name2id_map.get(node_name_with_suffix, -1)
                if node_id != -1:
                    self.macro_names[self.macro_names.index(node_name)] = node_name_with_suffix

            if node_id == -1:
                print(f"Warning: Macro '{node_name}' not found in PlaceDB.")
                continue
            
            self.macro_ids.append(node_id)

        print("Total macros identified: ", len(self.macro_ids))
        # TODO: get port
        for node_name in self.dmp_placedb.node_names:
            node_name_str = node_name.decode('utf-8') if isinstance(node_name, bytes) else str(node_name)
            if node_name_str in self.dmp_placedb.node_name2id_map:
                node = self.dmp_placedb.node_name2id_map[node_name_str]
            else:
                continue
            if node < (self.dmp_placedb.num_physical_nodes - self.dmp_placedb.num_terminal_NIs):  # exclude IO ports
                continue
            else:
                self.port_indices.append(node)  # store the port indices
        # print port info
        print("Total ports identified: ", len(self.port_indices))
        scale_x, scale_y = get_inv_scaling_ratio(macro_placedb.database)

        self.macro_names = np.array(self.macro_names).astype(np.str_) # TODO
        self.n_macro = len(self.macro_names) # TODO
        self.node_names = self.dmp_placedb.node_names.astype(np.str_)
        for node in self.port_indices:
            raw_x = self.dmp_placedb.node_x[node]
            raw_y = self.dmp_placedb.node_y[node]

            pos_x = math.floor(max(0, (raw_x - scale_x) / scale_x))
            pos_y = math.floor(max(0, (raw_y - scale_y) / scale_y))
            self.port_pos.append((pos_x, pos_y))
        # TODO: compute macro clusters 
        self.macro_cluster_list = self.community_partition() # TODO
        print(f"Total macro clusters identified: {len(self.macro_cluster_list)}")
        # check macro clusters
        for i, cluster in enumerate(self.macro_cluster_list):
            print(f" Cluster {i}: {len(cluster)} macros")
            print("  Macros: ", [self.node_names[macro_id] for macro_id in cluster])
        # TODO: compute virtual connections based on dataflow
        res,  macro2index_map = self.compute_virtual_connections(self.macro_ids, self.dmp_placedb, k_max=5)
        print("Computed virtual connections based on dataflow.")
        # TODO: compute port pos rect
        # 检查一下macro_placedb里面的node_info={macro_name : {"raw_x": , "raw_y": , ."size_x": , "size_y": }} 画出来的bounding box
        # 然后检查port pos画出来的bounding box
        # 假设然后基于边界（macro_placedb里面的canvas_width, height）画一个边界
        DEBUG = False
        if DEBUG:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(12, 12))

            # Determine canvas size with margin
            margin = 0.1
            canvas_w = macro_placedb.canvas_width
            canvas_h = macro_placedb.canvas_height
            
            ax.set_xlim(-canvas_w * margin, canvas_w * (1 + margin))
            ax.set_ylim(-canvas_h * margin, canvas_h * (1 + margin))

            # Draw canvas boundary
            rect = patches.Rectangle((0, 0), canvas_w, canvas_h, linewidth=2, edgecolor='black', facecolor='none', label='Canvas')
            ax.add_patch(rect)

            # Draw macros
            print("Drawing macros...")
            for macro_name in self.macro_names:
                macro_name = macro_name.replace(".DREAMPlace.Shape0", "")  # Remove suffix for lookup
                if macro_name in macro_placedb.node_info:
                    info = macro_placedb.node_info[macro_name]
                    # Use raw coordinates from macro_placedb
                    mx, my = info["raw_x"], info["raw_y"]
                    mw, mh = info["size_x"], info["size_y"]
                    
                    rect = patches.Rectangle((mx, my), mw, mh, linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
                    ax.add_patch(rect)
                # ax.text(mx + mw/2, my + mh/2, macro_name, ha='center', va='center', fontsize=6, color='black')

            # Draw ports
            print("Drawing ports...")
            port_x = [p[0] for p in self.port_pos]
            port_y = [p[1] for p in self.port_pos]
            
            # Scatter plot for ports
            ax.scatter(port_x, port_y, c='red', s=10, marker='x', label='Ports')

            ax.set_aspect('equal')
            ax.legend()
            plt.title(f'Macro and Port Placement Check: {self.benchmark}')
            
            debug_dir = os.path.join(self.args.ROOT_DIR, "debug_plots")
            os.makedirs(debug_dir, exist_ok=True)
            save_path = os.path.join(debug_dir, f"{self.benchmark}_placement_check.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Debug plot saved to {save_path}")

        # TODO: 记得macro cluster里面的macro的name是由DREAMPlace.Shape0后缀，返回前需要处理一下
        return
        
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
        dataset = args_dict.get("dataset", "iccad2015")
        self.dmp_params.load(json_path)
        ##logging.info(f"Loaded DMP config from {json_path}")
        benchmark_path = os.path.join(
            root_dir,
            "benchmarks",
            dataset,
            benchmark,
        )


        def suffix2path(suffix: str) -> str:
            return os.path.join(benchmark_path, f"{benchmark}") + suffix

        if benchmark_type == "aux":
            self.dmp_params.fromJson({
                "aux_input": suffix2path(".aux")
            })
        elif benchmark_type == "def":
            self.dmp_params.fromJson({
                "def_input": suffix2path(".def"),
                "lef_input": suffix2path(".lef"),
                "verilog_input": suffix2path(".v"),
                "early_lib_input": suffix2path("_Early.lib"),
                "late_lib_input": suffix2path("_Late.lib"),
                "sdc_input": suffix2path(".sdc"),
            })
        
    
    
    def community_partition(self):
        cell_graph_builder = GraphBuilder(self.dmp_placedb)
        cell_graph = cell_graph_builder.build_graph()

        plm = nk.community.PLM(cell_graph, gamma=self.args.gamma, par="none")
        plm.run()
        communities = plm.getPartition()
        
        macro_cluster_dict = {i: [] for i in range(len(communities))}
        for node_id, node in enumerate(self.node_names):
            if node in self.macro_names:
                module = communities[node_id]
                macro_cluster_dict[module].append(self.dmp_placedb.node_name2id_map[node])

        # convert to List[List[node_id]]
        # filter empty clusters
        macro_cluster_list = [cluster for cluster in macro_cluster_dict.values() if len(cluster) > 0]
        return macro_cluster_list

    def get_node2type_map(self):
        # for dataflow
        gate2cell_map = self.timer.raw_timer.get_gate_cell_mapping()
        node2type_map = np.zeros(len(self.node_names), dtype=np.bool_)
        for node_id, node_name in enumerate(self.node_names):
            if node_name in self.macro_names:
                node2type_map[node_id] = True
            else:
                node_type = gate2cell_map.get(node_name, None)
                if node_type is not None:
                    if "DFF" in node_type:
                        node2type_map[node_id] = True
        return node2type_map

    def build_dataflow_graph(self, macros:list):
        graph_builder = DataflowGraphBuilder(self.dmp_placedb)
        if hasattr(self.args, "dataflow_graph_path"):
            graph_builder.load_graph(self.args.dataflow_graph_path)
            return graph_builder
            
        node2pin_map = self.dmp_placedb.node2pin_map
        pin2net_map = self.dmp_placedb.pin2net_map
        net2pin_map = self.dmp_placedb.net2pin_map
        pin2node_map = self.dmp_placedb.pin2node_map
        pin_direct = self.dmp_placedb.pin_direct.astype(np.str_)

        node2type_map = self.get_node2type_map()

        # get port
        port_names = self.node_names[
            self.num_movable_nodes + self.dmp_placedb.num_terminals: 
            self.num_movable_nodes + self.dmp_placedb.num_terminals + self.dmp_placedb.num_terminal_NIs
        ]
        port_ids = []
        pin_names = self.dmp_placedb.pin_names.astype(np.str_)
        for port_name in port_names:
            port_id = np.where(pin_names == port_name)[0].item()
            port_ids.append(port_id)

        # source node (connected to input port)
        source_nodes = set()
        for port_id in port_ids:
            if pin_direct[port_id] == "OUTPUT":
                port_net = pin2net_map[port_id]
                net_pins = net2pin_map[port_net]
                input_pins = net_pins[pin_direct[net_pins] == "INPUT"]
                input_nodes = pin2node_map[input_pins]
                source_nodes.update(input_nodes)

        def get_downstream_neighbors(node_id):
            neighbors = set()
            node_pin_ids = node2pin_map[node_id]
            node_pin_directs = pin_direct[node_pin_ids]
            nets = pin2net_map[node_pin_ids]
            net_pin_id_lists = net2pin_map[nets]
            for pin_dir, net_pin_ids in zip(node_pin_directs, net_pin_id_lists):
                if pin_dir != "OUTPUT":
                    continue
                net_pin_directs = pin_direct[net_pin_ids]
                input_mask = (net_pin_directs == "INPUT")
                if np.any(input_mask):
                    input_pins = net_pin_ids[input_mask]
                    input_nodes = pin2node_map[input_pins]
                    for nb in input_nodes:
                        nb = int(nb)
                        if nb != node_id:
                            neighbors.add(nb)
            return neighbors

        node2downstream_special_nodes_map = {}
        def dfs(node_id, last_special_node, path_set):
            if node_id in node2downstream_special_nodes_map:
                if last_special_node is not None:
                    for sink_node in node2downstream_special_nodes_map[node_id]:
                        if sink_node != last_special_node:
                            graph_builder.add_edge(last_special_node, sink_node)
                
                return node2downstream_special_nodes_map[node_id]

            if node_id in path_set:
                return []

            path_set.add(node_id)

            is_special = bool(node2type_map[node_id])
            next_last = last_special_node
            if is_special:
                node2downstream_special_nodes_map[node_id] = [node_id]
                if last_special_node is not None and node_id != last_special_node:
                    graph_builder.add_edge(last_special_node, node_id)
                next_last = node_id

            downstream_special_nodes = []
            for neighbor_node in get_downstream_neighbors(node_id):
                downstream_special_nodes.extend(dfs(neighbor_node, next_last, path_set))

            if node2downstream_special_nodes_map.get(node_id, None) is None:
                node2downstream_special_nodes_map[node_id] = list(set(downstream_special_nodes))

            path_set.remove(node_id)
            return node2downstream_special_nodes_map[node_id]

        def iterative_dfs(start_node_id, start_last_special_node):
            # Stack elements: (node_id, last_special_node, path_set, neighbors_iterator, downstream_special_nodes)
            stack = [(start_node_id, start_last_special_node, set(), None, [])]
            path_set = set()
            
            while stack:
                node_id, last_special_node, current_path, neighbors_iter, downstream_special_nodes = stack[-1]
                
                # First visit to this node
                if neighbors_iter is None:
                    # Check cache
                    if node_id in node2downstream_special_nodes_map:
                        if last_special_node is not None:
                            for sink_node in node2downstream_special_nodes_map[node_id]:
                                if sink_node != last_special_node:
                                    graph_builder.add_edge(last_special_node, sink_node)
                        stack.pop()
                        continue
                    
                    # Check for cycles
                    if node_id in path_set:
                        stack.pop()
                        continue
                    
                    # Enter node
                    path_set.add(node_id)
                    current_path.add(node_id)
                    
                    # Process special nodes
                    is_special = bool(node2type_map[node_id])
                    next_last = last_special_node
                    if is_special:
                        node2downstream_special_nodes_map[node_id] = [node_id]
                        if last_special_node is not None and node_id != last_special_node:
                            graph_builder.add_edge(last_special_node, node_id)
                        next_last = node_id
                    
                    # Get neighbors iterator
                    neighbors = list(get_downstream_neighbors(node_id))
                    stack[-1] = (node_id, last_special_node, current_path, iter(neighbors), [])
                    continue
                
                # Process neighbors
                try:
                    neighbor_node = next(neighbors_iter)
                    # Recursive call to neighbor (push to stack)
                    stack.append((neighbor_node, next_last, current_path.copy(), None, []))
                except StopIteration:
                    # All neighbors processed, compute downstream special nodes
                    if node_id not in node2downstream_special_nodes_map:
                        # Deduplicate and cache
                        unique_downstream = list(dict.fromkeys(downstream_special_nodes))
                        node2downstream_special_nodes_map[node_id] = unique_downstream
                    
                    # Exit node
                    path_set.remove(node_id)
                    current_path.remove(node_id)
                    
                    # Return result to parent node
                    if len(stack) > 1:
                        parent_node, parent_last, parent_path, parent_iter, parent_downstream = stack[-2]
                        parent_downstream.extend(node2downstream_special_nodes_map[node_id])
                        stack[-2] = (parent_node, parent_last, parent_path, parent_iter, parent_downstream)
                    
                    stack.pop()

        for i, s in enumerate(source_nodes):
            t = time.time()
            iterative_dfs(int(s), None)

            t_interval = time.time() - t
            if t_interval > 10:
                print(f"finish {i+1} in {t_interval} seconds")
        
        
        path = os.path.join(self.args.ROOT_DIR, "dataflow_graph", self.benchmark)
        if not os.path.exists(path):
            os.makedirs(path)
        graph_builder.save_graph(path)

        return graph_builder

    def compute_virtual_connections(self, macros, placedb, k_max=5):
        graph_builder = self.build_dataflow_graph(macros)
        dataflow_graph = graph_builder.convert2undirected()

        macro2index_map = {node: i for i, node in enumerate(macros)}
        
        def compute_dataflow_mat():
            res = np.zeros((self.n_macro, self.n_macro), dtype=np.float32)
            count = np.zeros((self.n_macro, self.n_macro), dtype=np.int32)
            for i, m in enumerate(macros):
                m_int = graph_builder.internal_id(m)
                if m_int == -1:
                    continue
                
                t = time.time()
                queue = [(m_int, 0, [m_int])]
                while len(queue) > 0:
                    node_int, k, path_history = queue.pop(0)
                    
                    neighbors_int = list(dataflow_graph.iterNeighbors(node_int))
                    for neighbor_int in neighbors_int:
                        neighbor_ext = graph_builder.external_id(neighbor_int)
                        if neighbor_ext in macro2index_map:
                            if m != neighbor_ext:
                                if neighbor_int not in path_history:
                                    res[macro2index_map[m], macro2index_map[neighbor_ext]] += 1 / ( 2 ** (k+1))
                                    count[macro2index_map[m], macro2index_map[neighbor_ext]] += 1
                        if k+1 < k_max and neighbor_int not in path_history:
                            queue.append((neighbor_int, k+1, path_history + [neighbor_int]))
                
                # print(f"finish {i+1} in {time.time() - t} seconds")
            
            return res

        
        for k_max in [2, 4]:
            res = compute_dataflow_mat()
            path = os.path.join(self.args.ROOT_DIR, "dataflow_info", self.benchmark, f"{k_max}")
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(os.path.join(path, "dataflow_mat.npy"), res)
            with open(os.path.join(path, "macro2index_map.pkl"), "wb") as f:
                pickle.dump(macro2index_map, f)

        return res, macro2index_map



class GraphBuilder:
    def __init__(self, placedb):
        self.net_names = placedb.net_names
        self.net_weights = placedb.net_weights
        self.net2pin_map = placedb.net2pin_map

        self.pin2node_map = placedb.pin2node_map
        self.node_names = placedb.node_names

        self.graph = nk.Graph(n=len(self.node_names), weighted=True, directed=False)

    def add_nodes(self):
        pass

    def add_edges(self):
        node_name_to_index = {name: i for i, name in enumerate(self.node_names)}

        for net_index, pins in enumerate(self.net2pin_map):
            weight = self.net_weights[net_index] if net_index < len(self.net_weights) else 1
            connected_nodes = set()
            for pin_id in pins:
                node_index = self.pin2node_map[pin_id]
                connected_nodes.add(node_index)

            if len(connected_nodes) > 2:
                virtual_node_index = self.graph.addNode()
                for node_index in connected_nodes:
                    self.graph.addEdge(virtual_node_index, node_index, weight)
            else:
                connected_nodes = list(connected_nodes)
                if len(connected_nodes) == 2:
                    self.graph.addEdge(connected_nodes[0], connected_nodes[1], weight)

    def build_graph(self):
        self.add_nodes()
        self.add_edges()
        return self.graph

    



class DataflowGraphBuilder:
    def __init__(self, placedb, directed=True):
        self.node_names = placedb.node_names.astype(np.str_)
        self.ext2int = {}
        self.int2ext = []
        self.graph = nk.Graph(n=0, weighted=False, directed=directed)

    def internal_id(self, ext_id):
        return self.ext2int.get(ext_id, -1)

    def external_id(self, int_id):
        if int_id < len(self.int2ext):
            return self.int2ext[int_id]
        else:
            return -1

    def add_edge(self, ext_id1, ext_id2):
        def to_internal(ext_id):
            if ext_id not in self.ext2int:
                self.ext2int[ext_id] = self.graph.addNode()
                self.int2ext.append(ext_id)
                assert len(self.int2ext) - 1 == self.ext2int[ext_id], (len(self.int2ext[-1]) - 1, self.ext2int[ext_id])
            return self.ext2int[ext_id]
        
        int_id1 = to_internal(ext_id1)
        int_id2 = to_internal(ext_id2)
        self.graph.addEdge(int_id1, int_id2)

        # if len(self.ext2int) <= 10:
        #     degree1 = self.graph.degree(int_id1)
        #     degree2 = self.graph.degree(int_id2)
        #     in_degree1 = self.graph.degreeIn(int_id1)
        #     in_degree2 = self.graph.degreeIn(int_id2)
        #     out_degree1 = self.graph.degreeOut(int_id1)
        #     out_degree2 = self.graph.degreeOut(int_id2)
        #     print(f"Edge {len(self.ext2int)}: {self.node_names[ext_id1]} -> {self.node_names[ext_id2]}")
        #     print(f"  source {ext_id1}: degree={degree1}, in_degree={in_degree1}, out_degree={out_degree1}")
        #     print(f"  sink {ext_id2}: degree={degree2}, in_degree={in_degree2}, out_degree={out_degree2}")
        #     print(f"  validation: degree == out_degree? source: {degree1 == out_degree1}, sink: {degree2 == out_degree2}")
        #     print()

    def save_graph(self, path):
        writer = graphio.EdgeListWriter("\t", 0)  # 分隔符、首节点编号
        writer.write(self.graph, os.path.join(path, "dataflow_graph"))

        content = {
            "ext2int": self.ext2int,
            "int2ext": self.int2ext
        }

        with open(os.path.join(path, "dataflow_graph_map.pkl"), "wb") as f:
            pickle.dump(content, f)

    def load_graph(self, path):
        reader = graphio.EdgeListReader(separator="\t", firstNode=0, directed=self.graph.isDirected())
        self.graph = reader.read(os.path.join(path, "dataflow_graph"))
    
        with open(os.path.join(path, "dataflow_graph_map.pkl"), "rb") as f:
            content = pickle.load(f)
        self.ext2int = content["ext2int"]
        self.int2ext = content["int2ext"]

    def convert2undirected(self):
        undirected_graph = nk.Graph(n=self.graph.numberOfNodes(), weighted=False, directed=False)
        for u, v in self.graph.iterEdges():
            if not undirected_graph.hasEdge(u, v):
                undirected_graph.addEdge(u, v)
        return undirected_graph
    
if __name__ == "__main__":
    from config.benchmark import (
    ROOT_DIR, BENCHMARK_DIR, benchmark_dict, 
    benchmark_type_dict, benchmark_n_macro_dict
)
    benchmark = args.benchmark
    dataset = args.dataset
    args.ROOT_DIR = ROOT_DIR
    if not args.benchmark_path:
        args.benchmark_path = os.path.join(
            args.ROOT_DIR,
            "benchmarks",
            dataset,
            benchmark,
        )
    from placedb import PlaceDB as MacroPlaceDB
    # load macro placedb
    macro_placedb = MacroPlaceDB(args)
        
    problem_instance = ProblemInstance(args, benchmark, macro_placedb=macro_placedb)