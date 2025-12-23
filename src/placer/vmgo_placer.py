import math
import numpy as np
import logging

from .basic_placer import BasicPlacer
from src.utils.constant import INF
from src.utils.debug import *

class VanillaMaskGuidedOptimizationPlacer(BasicPlacer):
    def __init__(self, args, placedb, eval_metrics= ['hpwl']) -> None:
        super(VanillaMaskGuidedOptimizationPlacer, self).__init__(args=args, placedb=placedb, eval_metrics=eval_metrics)

        self.n_grid_x = args.n_grid_x
        self.n_grid_y = args.n_grid_y
        self.canvas_width  = placedb.canvas_width
        self.canvas_height = placedb.canvas_height

        self.grid_width  = self.canvas_width / self.n_grid_x
        self.grid_height = self.canvas_height / self.n_grid_y
        
        self.ranked_macro = self._rank_macro()

        # scale size_x, size_y
        self.scale_size = {}
        for macro in self.ranked_macro:
            size_x = self.placedb.node_info[macro]["size_x"]
            size_y = self.placedb.node_info[macro]["size_y"]
            scaled_size_x = math.ceil(size_x / self.grid_width)
            scaled_size_y = math.ceil(size_y / self.grid_height)
            self.scale_size[macro] = {}
            self.scale_size[macro]["size_x"] = scaled_size_x
            self.scale_size[macro]["size_y"] = scaled_size_y
    

    def _genotype2phenotype(self, x):
        # get x_id, y_id for all macro based on genotype x
        macro_grid_pos = {}
        for idx, macro_name in zip(range(0, self.placedb.node_cnt), self.placedb.macro_lst):
            x_id = x[idx]
            y_id = x[idx + self.placedb.node_cnt]
            macro_grid_pos[macro_name] = (math.floor(x_id), math.floor(y_id))

        # greedy search legal grid location
        placed_macro_grid_pos = {}
        hpwl_info_for_each_net = {}
        hpwl = 0.0

        # 1. Maintain a global occupancy grid (0: Empty, 1: Occupied)
        occupancy_grid = np.zeros((self.n_grid_x, self.n_grid_y), dtype=int)

        # Pre-compute grid coordinates for vectorization
        x_grid_indices = np.arange(self.n_grid_x) * self.grid_width
        y_grid_indices = np.arange(self.n_grid_y) * self.grid_height

        for i, macro in enumerate(self.ranked_macro):
            size_x = self.placedb.node_info[macro]["size_x"]
            size_y = self.placedb.node_info[macro]["size_y"]
            scaled_size_x = self.scale_size[macro]["size_x"]
            scaled_size_y = self.scale_size[macro]["size_y"]

            # Check bounds - if macro is bigger than canvas, we must place it at (0,0) or similar
            # but let's keep the error if it's physically impossible to fit
            if scaled_size_x > self.n_grid_x or scaled_size_y > self.n_grid_y:
                logging.error(f"Macro {macro} too big.")
                return {}

            # --- OPTIMIZATION START: Integral Image for Position Mask ---
            
            valid_w = self.n_grid_x - scaled_size_x + 1
            valid_h = self.n_grid_y - scaled_size_y + 1

            # Compute 2D Integral Image (Summed Area Table)
            sat = np.pad(occupancy_grid.cumsum(axis=0).cumsum(axis=1), ((1,0), (1,0)), 'constant')

            # D: Bottom-Right, B: Top-Right, C: Bottom-Left, A: Top-Left
            sat_d = sat[scaled_size_x : scaled_size_x + valid_w, scaled_size_y : scaled_size_y + valid_h]
            sat_b = sat[0 : valid_w, scaled_size_y : scaled_size_y + valid_h]
            sat_c = sat[scaled_size_x : scaled_size_x + valid_w, 0 : valid_h]
            sat_a = sat[0 : valid_w, 0 : valid_h]

            # Calculate area sum (occupancy) for every possible position
            area_sums = sat_d - sat_b - sat_c + sat_a

            # Initialize position_mask with INF
            position_mask = np.full((self.n_grid_x, self.n_grid_y), INF)
            
            # Find valid positions (where sum is 0)
            valid_indices = (area_sums == 0)
            
            # Check if we have ANY legal position
            has_legal_pos = np.any(valid_indices)

            if has_legal_pos:
                # Normal case: Only allow placement in empty spots
                mask_view = position_mask[:valid_w, :valid_h]
                mask_view[valid_indices] = 0.0
            else:
                # Fallback case: No legal place found.
                # Instead of returning empty, we allow overlap.
                # We set the mask to 0.0 for ALL physically possible positions (within bounds),
                # effectively ignoring the occupancy grid.
                # Note: We still respect valid_w/valid_h to ensure the macro stays inside the canvas.
                position_mask[:valid_w, :valid_h] = 0.0
                
                # Optional: Log a warning if needed, but for optimization loop it might be too noisy
                #logging.debug(f"Macro {macro} forced overlap, index {i+1}/{len(self.ranked_macro)}")
                #print(f"Macro {macro} forced overlap, index {i+1}/{len(self.ranked_macro)}")

            # --- OPTIMIZATION END ---

            # 2. Calculate Wire Mask (Vectorized)
            cost_x = np.zeros(self.n_grid_x)
            cost_y = np.zeros(self.n_grid_y)

            for net_name in self.placedb.node_to_net_dict[macro]:
                if net_name in hpwl_info_for_each_net:
                    net_info = hpwl_info_for_each_net[net_name]
                    x_offset = self.placedb.net_info[net_name]["nodes"][macro]["x_offset"] + 0.5 * size_x
                    y_offset = self.placedb.net_info[net_name]["nodes"][macro]["y_offset"] + 0.5 * size_y

                    current_pin_x = x_grid_indices + x_offset
                    current_pin_y = y_grid_indices + y_offset

                    d_x = np.maximum(net_info["x_min"] - current_pin_x, 0) + np.maximum(current_pin_x - net_info["x_max"], 0)
                    d_y = np.maximum(net_info["y_min"] - current_pin_y, 0) + np.maximum(current_pin_y - net_info["y_max"], 0)

                    cost_x += d_x
                    cost_y += d_y

            wire_mask = cost_x[:, np.newaxis] + cost_y[np.newaxis, :]
            wire_mask += 0.1 

            # 3. Combine and Select
            total_cost_mask = position_mask + wire_mask
            min_val = np.min(total_cost_mask)
            

            hpwl += (min_val - 0.1)

            candidates_x, candidates_y = np.where(total_cost_mask == min_val)
            
            target_x, target_y = macro_grid_pos[macro]
            dist_sq = (candidates_x - target_x)**2 * (self.grid_width**2) + \
                      (candidates_y - target_y)**2 * (self.grid_height**2)
            
            best_idx = np.argmin(dist_sq)
            chosen_scale_x = candidates_x[best_idx]
            chosen_scale_y = candidates_y[best_idx]

            placed_macro_grid_pos[macro] = (chosen_scale_x, chosen_scale_y)

            # 4. Update Occupancy Grid
            # Even if we forced overlap, we still mark this place as occupied for FUTURE macros
            # This encourages subsequent macros to avoid this crowded spot if possible
            ox_end = min(self.n_grid_x, chosen_scale_x + scaled_size_x)
            oy_end = min(self.n_grid_y, chosen_scale_y + scaled_size_y)
            occupancy_grid[chosen_scale_x : ox_end, chosen_scale_y : oy_end] = 1

            # 5. Update Net Bounding Boxes
            center_x = self.grid_width * chosen_scale_x + 0.5 * size_x
            center_y = self.grid_height * chosen_scale_y + 0.5 * size_y
            
            for net_name in self.placedb.node_to_net_dict[macro]:
                x_offset = self.placedb.net_info[net_name]["nodes"][macro]["x_offset"]
                y_offset = self.placedb.net_info[net_name]["nodes"][macro]["y_offset"]
                pin_x = center_x + x_offset
                pin_y = center_y + y_offset
                
                if net_name not in hpwl_info_for_each_net:
                    hpwl_info_for_each_net[net_name] = {
                        "x_max" : pin_x, "x_min" : pin_x,
                        "y_max" : pin_y, "y_min" : pin_y,
                    }
                else:
                    net_entry = hpwl_info_for_each_net[net_name]
                    if net_entry["x_max"] < pin_x: net_entry["x_max"] = pin_x
                    if net_entry["x_min"] > pin_x: net_entry["x_min"] = pin_x
                    if net_entry["y_max"] < pin_y: net_entry["y_max"] = pin_y
                    if net_entry["y_min"] > pin_y: net_entry["y_min"] = pin_y
            
        # unscale
        macro_pos = {}
        for macro, (scaled_x, scaled_y) in placed_macro_grid_pos.items():
            x = scaled_x * self.grid_width 
            y = scaled_y * self.grid_height
            macro_pos[macro] = (x, y)

        return macro_pos, {}


    def _rank_macro(self):
        macro_lst = self.placedb.macro_lst.copy()
        net_lst = list(self.placedb.net_info.keys()).copy()

        # compute macro area per net
        for net_name in net_lst:
            area_sum = 0
            for macro in self.placedb.net_info[net_name]["nodes"].keys():
                area_sum += self.placedb.node_info[macro]["area"]
            self.placedb.net_info[net_name]["area"] = area_sum
        # compute area sum per macro
        for macro in macro_lst:
            self.placedb.node_info[macro]["area_sum"] = 0
            for net_name in net_lst:
                if macro in self.placedb.net_info[net_name]["nodes"].keys():
                    self.placedb.node_info[macro]["area_sum"] += self.placedb.net_info[net_name]["area"]

        macro_lst.sort(key=lambda x: self.placedb.node_info[x][self.args.rank_key], reverse=True)
        return macro_lst

    
