import numpy as np 
import ray
import sys
from pymoo.core.problem import Problem

from utils.debug import *


class PlacementProblem(Problem):
    def __init__(self, n_var, xl, xu, placer, n_obj=1):
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=n_obj,
            vtype=np.int64
        )
        self.n_var = n_var
        self.xl = xl
        self.xu = xu
        self.placer = placer

    def fitness_function(self, x):
        return self.placer.evaluate([x])
    
    def get_problem_dict(self):
        # assert0(self.xl, self.xu)
        return {
            "fitness_function" : self.fitness_function,
            "ndim_problem" : self.n_var,
            "upper_boundary" : self.xu,
            "lower_boundary" : self.xl
        }
        
    
    def _evaluate(self, x, out, *args, **kwargs):
        y, macro_pos_all = self.placer.evaluate([x])
            
        out["F"] = np.array(y)
        out["macro_pos"] = macro_pos_all
        out["X"] = x
    
class MaskGuidedOptimizationPlacementProblem(PlacementProblem):
    def __init__(self, n_grid_x, n_grid_y, placer, n_obj=1):
        self.node_cnt = placer.placedb.node_cnt
        self.n_grid_x = n_grid_x
        self.n_grid_y = n_grid_y
        super().__init__(
            n_var=self.node_cnt * 2,
            xl=np.zeros(self.node_cnt * 2),
            xu=np.array(
                ([self.n_grid_x] * self.node_cnt) + \
                ([self.n_grid_y] * self.node_cnt)
            ),
            n_obj=n_obj,
            placer=placer
        )

class SequencePairPlacementProblem(PlacementProblem):
    def __init__(self, placer, n_obj=1):
        self.node_cnt = placer.placedb.node_cnt
        super().__init__(
            n_var=self.node_cnt * 2,
            xl=np.zeros(self.node_cnt * 2),
            xu=np.array([self.node_cnt] * self.node_cnt * 2),
            n_obj=n_obj,
            placer=placer
        )

class HyperparameterPlacementProblem(PlacementProblem):
    def __init__(self, params_space, placer, n_obj=1):
        self.params_space = params_space
        self.n_var = len(self.params_space.keys())

        extract = lambda ent_i: \
            [entry[ent_i] for entry in self.params_space.values()]
        self.params_name = list(self.params_space.keys())
        self.xl = np.array(extract(0))
        self.xu = np.array(extract(1))

        super().__init__(
            n_var=self.n_var,
            xl=self.xl,
            xu=self.xu,
            placer=placer,
            n_obj=n_obj
        )