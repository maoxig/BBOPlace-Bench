from src.utils.constant import INF
from src.utils.debug import *

import numpy as np


def comp_res(macros_pos, placedb, eval_metrics=['hpwl'], ) -> dict:
    if len(macros_pos) == 0:
        print("Warning: No macro placed, return INF for all results")
        return {metric: INF for metric in eval_metrics}
    net_hpwl = _comp_net_hpwl(macros_pos, placedb)
    res = {}
    for metric in eval_metrics:
        if metric == 'hpwl':
            res['hpwl'] = _comp_res_hpwl(net_hpwl, placedb)
        elif metric == 'rudy':
            res['rudy'] = _comp_res_rudy(net_hpwl, placedb)
        elif metric == 'rudy2':
            res['rudy2'] = _comp_res_rudy_improved(net_hpwl, placedb, num_bins_x=224, num_bins_y=224)
        elif metric == 'regularity':
            res['regularity'] = _comp_res_regularity(macros_pos, placedb)
        elif metric == 'overlap':
            res['overlap'] = _comp_overlap(macros_pos, placedb)
        elif metric == 'dataflow_cost':
            res['dataflow_cost'] = _comp_dataflow_cost(macros_pos, placedb)
        elif metric == 'macro_grouping_cost':
            res['macro_grouping_cost'] = _comp_macro_grouping_cost(macros_pos, placedb)
        else:
            pass
    return res




def _comp_net_hpwl(macro_pos, placedb):
    assert len(macro_pos) > 0

    net_hwpl = {}
    for net_name in placedb.net_info:
        max_x = 0.0
        min_x = placedb.canvas_width * 1.1
        max_y = 0.0
        min_y = placedb.canvas_height * 1.1
        for macro in placedb.net_info[net_name]["nodes"]:
            size_x = placedb.node_info[macro]["size_x"]
            size_y = placedb.node_info[macro]["size_y"]
            pin_x = macro_pos[macro][0] + size_x / 2 + placedb.net_info[net_name]["nodes"][macro]["x_offset"]
            pin_y = macro_pos[macro][1] + size_y / 2 + placedb.net_info[net_name]["nodes"][macro]["y_offset"]
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)
        
        net_hwpl[net_name] = (min_x, min_y, max_x, max_y)
    
    return net_hwpl

def _comp_res_rudy_improved(net_hpwl, placedb, num_bins_x=64, num_bins_y=64):
    """
    改进的RUDY计算，参考DREAMPlace实现 - Vectorized Optimiazation
    
    关键改进：
    1. 使用粗粒度bin grid而非pixel-level
    2. 分别计算水平和垂直方向的拥塞
    3. 正确计算net bounding box与bin的重叠面积
    4. 考虑net权重
    5. 最终取两个方向的最大值
    6. 使用 NumPy 向量化替代 Python 循环以避免大规模网表时的卡顿
    """
    
    if not net_hpwl:
        return 0.0

    # 1. 准备数据: 将字典转换为 Numpy 数组
    net_names = list(net_hpwl.keys())
    coords = np.array(list(net_hpwl.values()), dtype=np.float32) # Shape: [N_nets, 4]
    
    min_x = coords[:, 0]
    min_y = coords[:, 1]
    max_x = coords[:, 2]
    max_y = coords[:, 3]
    
    # 获取权重
    weights = np.array([placedb.net_info[name].get("weight", 1.0) for name in net_names], dtype=np.float32)
    
    # 预计算分母 (防止除零)
    net_widths = max_x - min_x + 1e-6
    net_heights = max_y - min_y + 1e-6
    
    # 计算因子: weight / dimension
    # RUDY: horizontal_util += area / height * weight
    #       vertical_util   += area / width  * weight
    h_factor = weights / net_heights
    v_factor = weights / net_widths
    
    # 2. 向量化网格计算 (按列扫描以控制内存峰值)
    bin_size_x = placedb.canvas_width / num_bins_x
    bin_size_y = placedb.canvas_height / num_bins_y
    
    # 预先生成 Y 轴的 Bin 边界，用于广播计算
    # shape: (1, num_bins_y)
    y_bin_edges = np.arange(num_bins_y + 1, dtype=np.float32) * bin_size_y
    y_bin_starts = y_bin_edges[:-1][np.newaxis, :] 
    y_bin_ends = y_bin_edges[1:][np.newaxis, :]    
    
    horizontal_utilization = np.zeros((num_bins_x, num_bins_y), dtype=np.float32)
    vertical_utilization = np.zeros((num_bins_x, num_bins_y), dtype=np.float32)
    
    # 按列 (X轴 Bin) 遍历
    for bx in range(num_bins_x):
        bin_lx = bx * bin_size_x
        bin_ux = (bx + 1) * bin_size_x
        
        # Vectorized: 计算所有 nets 在当前 bin column (X方向) 的重叠长度
        # intersection of [min_x, max_x] and [bin_lx, bin_ux]
        overlap_x = np.maximum(0, np.minimum(max_x, bin_ux) - np.maximum(min_x, bin_lx))
        
        # 筛选：只处理涉及当前列的 nets (overlap_x > 0)
        mask = overlap_x > 1e-6
        if not np.any(mask):
            continue
            
        # 提取活跃 nets 的数据
        active_overlap_x = overlap_x[mask]       # (M,)
        active_min_y = min_y[mask][:, np.newaxis] # (M, 1) -> 用于广播
        active_max_y = max_y[mask][:, np.newaxis] # (M, 1)
        active_h_factor = h_factor[mask][:, np.newaxis]
        active_v_factor = v_factor[mask][:, np.newaxis]
        
        # Vectorized: 计算活跃 nets 在所有 Y Bins 的重叠长度
        # result shape: (M, num_bins_y)
        overlap_y = np.maximum(0, np.minimum(active_max_y, y_bin_ends) - np.maximum(active_min_y, y_bin_starts))
        
        # 综合计算 overlap_area 并加权
        # area = active_overlap_x * overlap_y
        # value = area * factor
        # 为了高效，先将 x_overlap 乘入 factor 或者 y_overlap
        
        weighted_overlap_y = overlap_y * active_overlap_x[:, np.newaxis] # (M, num_bins_y)
        
        # 累加到当前列的所有 Bin 中
        # sum over nets (axis 0)
        horizontal_utilization[bx, :] = np.sum(weighted_overlap_y * active_h_factor, axis=0)
        vertical_utilization[bx, :] = np.sum(weighted_overlap_y * active_v_factor, axis=0)
    
    # 归一化：除以bin面积
    bin_area = bin_size_x * bin_size_y
    horizontal_utilization /= bin_area
    vertical_utilization /= bin_area
    
    # 取两个方向的最大值作为routing utilization
    route_utilization = np.maximum(horizontal_utilization, vertical_utilization)
    
    # 返回top-k平均值（类似原实现）
    # 注意: partition 对 flatten 数组操作
    k = max(1, int(route_utilization.size * 0.1))
    rudy_score = np.partition(route_utilization.ravel(), -k)[-k:].mean()
    
    return float(rudy_score)

def _comp_res_hpwl(net_hpwl, placedb):
    hpwl = 0.0
    for net_name, bounding_box in net_hpwl.items():
        min_x, min_y, max_x, max_y = bounding_box 
        hpwl_temp = (max_x - min_x) + (max_y - min_y)
        
        if "weight" in placedb.net_info[net_name]:
            hpwl_temp *= placedb.net_info[net_name]["weight"]
        hpwl += hpwl_temp
    return hpwl


def _comp_res_rudy(net_hpwl, placedb):
    congestion = np.zeros((int(placedb.canvas_width), int(placedb.canvas_height)), dtype=np.float32)
    
    coords = np.array(list(net_hpwl.values()))
        
    coords[:, 0] = np.maximum(0, np.ceil(coords[:, 0]))  # min_x
    coords[:, 1] = np.maximum(0, np.ceil(coords[:, 1]))  # min_y
    coords[:, 2] = np.minimum(placedb.canvas_width, np.ceil(coords[:, 2]))   # max_x
    coords[:, 3] = np.minimum(placedb.canvas_height, np.ceil(coords[:, 3]))  # max_y
    
    delta_x = coords[:, 2] - coords[:, 0]
    delta_y = coords[:, 3] - coords[:, 1]
    
    valid_nets = (delta_x > 0) & (delta_y > 0)
    coords = coords[valid_nets]
    delta_x = delta_x[valid_nets]
    delta_y = delta_y[valid_nets]
    
    for i in range(len(coords)):
        min_x, min_y, max_x, max_y = coords[i].astype(int)
        congestion[min_x:max_x, min_y:max_y] += 1/delta_x[i] + 1/delta_y[i]
    
    k = max(1, int(congestion.size * 0.1))
    return np.partition(congestion.ravel(), -k)[-k:].mean()


def _comp_res_regularity(macro_pos, placedb):
    # FIXME: Warning, regularity is not feasible for sequence pair formulation
    # since macro will exceed the chip canvas 
    x_dis_from_edge = 0
    y_dis_from_edge = 0
    total_area = 0
    for macro_name, (macro_lx, macro_ly) in macro_pos.items():
        macro_ux = macro_lx + placedb.node_info[macro_name]["size_x"]
        macro_uy = macro_ly + placedb.node_info[macro_name]["size_y"]
        area = placedb.node_info[macro_name]["area"]
        total_area += area

        x_dis_from_edge += min(macro_lx, max(placedb.canvas_width  - macro_ux, 0)) * area
        y_dis_from_edge += min(macro_ly, max(placedb.canvas_height - macro_uy, 0)) * area
    
    return (x_dis_from_edge + y_dis_from_edge) / total_area


def _comp_overlap(macro_pos, placedb):
    overlap_area = 0
    macro_lst = list(macro_pos.keys())
    l = len(macro_lst)
    for idx in range(l):
        macro = macro_lst[idx]
        xl, yl = macro_pos[macro]
        xh = placedb.node_info[macro]["size_x"] + xl
        yh = placedb.node_info[macro]["size_y"] + yl
        for i in range(idx+1, l):
            m = macro_lst[i]
            m_xl, m_yl = macro_pos[m]
            m_xh = placedb.node_info[m]["size_x"] + m_xl
            m_yh = placedb.node_info[m]["size_y"] + m_yl
            if m_xh < xl or m_xl > xh or \
               m_yh < yl or m_yl > yh:
                continue
            
            delta_x = min(m_xh, xh) - max(m_xl, xl)
            delta_y = min(m_yh, yh) - max(m_yl, yl)

            if np.isnan(delta_x) or np.isnan(delta_y):
                return 0
            assert delta_x >= 0, (m_xh, xh, m_xl, xl)
            assert delta_y >= 0, (m_yh, yh, m_yl, yl)
            overlap_area += delta_x * delta_y
    
    return overlap_area / placedb.macro_area_sum

def _comp_macro_grouping_cost(macro_pos, placedb):
    if not hasattr(placedb, "macro_clusters"):
        return INF
    grouping_cost = 0.0
    macro_clusters = placedb.macro_clusters
    for cluster in macro_clusters:
        if not cluster:
            continue

        min_x = INF
        min_y = INF
        max_x = float('-inf')
        max_y = float('-inf')
        
        found_macro = False
        for macro_name in cluster:
            # Check for macro name or its variant without suffix
            clean_name = macro_name.replace(".DREAMPlace.Shape0", "")
            target_name = macro_name if macro_name in macro_pos else (clean_name if clean_name in macro_pos else None)
            
            if not target_name:
                continue

            found_macro = True
            lx, ly = macro_pos[target_name]
            # Use the original name to look up node info if possible, otherwise fallback
            info_name = macro_name if macro_name in placedb.node_info else clean_name
            
            ux = lx + placedb.node_info[info_name]["size_x"]
            uy = ly + placedb.node_info[info_name]["size_y"]

            min_x = min(min_x, lx)
            min_y = min(min_y, ly)
            max_x = max(max_x, ux)
            max_y = max(max_y, uy)
        
        if found_macro:
            grouping_cost += (max_x - min_x) * (max_y - min_y)

        return grouping_cost

def _comp_dataflow_cost(macro_pos, placedb):
    if not hasattr(placedb, "dataflow_mat") or not hasattr(placedb, "node_name2index_map"):
        return INF

    dataflow_mat = placedb.dataflow_mat
    node_name2index_map = placedb.node_name2index_map
    
    centers = []
    indices = []

    for macro_name, (x, y) in macro_pos.items():
        idx = -1
        suffix = ".DREAMPlace.Shape0"
        # Determine the key for index map
        idx_key = macro_name if macro_name in node_name2index_map else (
            macro_name + suffix if (macro_name + suffix) in node_name2index_map else None
        )

        if idx_key:
            idx = node_name2index_map[idx_key]
            # Get node info, preferring original name
            node_data = placedb.node_info.get(macro_name) or placedb.node_info.get(macro_name + suffix)

            if node_data:
                center_x = x + node_data["size_x"] / 2.0
                center_y = y + node_data["size_y"] / 2.0
                centers.append([center_x, center_y])
                indices.append(idx)
            
    if not centers:
        return 0.0

    centers = np.array(centers)
    indices = np.array(indices)

    # Extract submatrix of weights
    W = dataflow_mat[np.ix_(indices, indices)]

    # Compute pairwise L1 distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sum(np.abs(diff), axis=-1)

    # Compute weighted cost
    cost = np.sum(W * dists)

    return cost