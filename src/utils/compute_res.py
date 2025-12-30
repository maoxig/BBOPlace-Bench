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
        elif metric == 'congestion':
            res['congestion'] = _comp_res_congestion(net_hpwl, placedb)
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


def _comp_res_hpwl(net_hpwl, placedb):
    hpwl = 0.0
    for net_name, bounding_box in net_hpwl.items():
        min_x, min_y, max_x, max_y = bounding_box 
        hpwl_temp = (max_x - min_x) + (max_y - min_y)
        
        if "weight" in placedb.net_info[net_name]:
            hpwl_temp *= placedb.net_info[net_name]["weight"]
        hpwl += hpwl_temp
    return hpwl


def _comp_res_congestion(net_hpwl, placedb):
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