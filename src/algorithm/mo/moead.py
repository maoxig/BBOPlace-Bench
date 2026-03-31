import time 
import os 
import numpy as np 
from utils.debug import * 
from utils.constant import INF
from pymoo.core.population import Population
from placer.hpo_placer import params_space
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from ..basic_algo import BasicAlgo
import time
import os
import pickle
from operators import REGISTRY as OPS_REGISTRY

from problem.pymoo_problem import (
    MaskGuidedOptimizationPlacementProblem,
    SequencePairPlacementProblem,
    HyperparameterPlacementProblem
)


class AdaptiveDecomposition:
    def __init__(self, method, initial_nadir=None):
        self.method = method
        self.algo = None
        self.nadir = initial_nadir
        self.valid_upper = INF * 0.9

    def set_algo(self, algo):
        self.algo = algo
    
    def do(self, F, weights, **kwargs):
        ideal = kwargs.pop('ideal_point', None)
        if ideal is None:
            return self.method.do(F, weights, **kwargs)

        F_arr = np.asarray(F)
        F_2d = F_arr.reshape(1, -1) if F_arr.ndim == 1 else F_arr
        n_objs = F_2d.shape[1]

        if self.nadir is None or np.asarray(self.nadir).size != n_objs:
            self.nadir = np.ones(n_objs, dtype=F_2d.dtype)
        else:
            self.nadir = np.asarray(self.nadir, dtype=F_2d.dtype).reshape(-1)

        # Incremental nadir update from current F only (hot path optimization).
        valid_mask = np.isfinite(F_2d) & (F_2d < self.valid_upper)
        if np.any(valid_mask):
            cand = np.where(valid_mask, F_2d, -np.inf)
            col_max = np.max(cand, axis=0)
            has_valid = col_max > -np.inf
            if np.any(has_valid):
                self.nadir[has_valid] = np.maximum(self.nadir[has_valid], col_max[has_valid])

        local_ideal = np.asarray(ideal, dtype=F_2d.dtype).copy()
        ideal_valid = np.isfinite(local_ideal) & (local_ideal < self.valid_upper)
        local_ideal = np.where(ideal_valid, local_ideal, 0.0)

        diff = self.nadir - local_ideal
        diff = np.maximum(diff, 1e-6)

        F_norm = (F_arr - local_ideal) / diff

        return self.method.do(F_norm, weights, ideal_point=np.zeros_like(local_ideal), **kwargs)


class MOEADDE(BasicAlgo):
    def __init__(self, args, placer, logger):
        super(MOEADDE, self).__init__(args=args, placer=placer, logger=logger)
        self.node_cnt = placer.placedb.node_cnt
        self.eval_metrics = placer.eval_metrics
        if args.placer == "mgo":
            self.problem = MaskGuidedOptimizationPlacementProblem(
                n_grid_x=args.n_grid_x,
                n_grid_y=args.n_grid_y,
                placer=placer,
                n_obj=len(self.eval_metrics)
            )
            self.xl = np.zeros(self.node_cnt * 2)
            self.xu = np.array(
                ([args.n_grid_x] * self.node_cnt) + \
                    ([args.n_grid_y] * self.node_cnt)
            )
        elif args.placer == "vmgo":
            self.problem = MaskGuidedOptimizationPlacementProblem(
                n_grid_x=args.n_grid_x,
                n_grid_y=args.n_grid_y,
                placer=placer,
                n_obj=len(self.eval_metrics)
            )
            self.xl = np.zeros(self.node_cnt * 2)
            self.xu = np.array(
                ([args.n_grid_x] * self.node_cnt) + \
                    ([args.n_grid_y] * self.node_cnt)
            )
        elif args.placer == "sp":
            self.xl = np.zeros(self.node_cnt * 2)
            self.xu = np.array([self.node_cnt] * self.node_cnt * 2)
            self.problem = SequencePairPlacementProblem(
                placer=placer,
                n_obj=len(self.eval_metrics)
            )
        elif args.placer == "hpo":
            extract = lambda ent_i: \
                [entry[ent_i] for entry in params_space.values()]
            self.xl = np.array(extract(0))
            self.xu = np.array(extract(1))
            self.problem = HyperparameterPlacementProblem(
                params_space=params_space,
                placer=placer,
                n_obj=len(self.eval_metrics)
            )
        else:
            self.problem = None
            raise NotImplementedError
        
        # Generate reference directions for MOEAD
        self.ref_dirs = get_reference_directions(
            "energy", 
            len(self.eval_metrics), 
            n_points = self.args.n_population,
            seed=1
        )

        self.args.__dict__.update(
            {"logger": logger, "record_func": self._record_results}
        )  

    def run(self):
        checkpoint = self._load_checkpoint()
        initial_population = None
        initial_algo_n_gen = 0

        if checkpoint is not None:
            assert len(checkpoint["population"]) == self.args.n_population
            initial_population = Population.new(
                X=checkpoint["population"], F=checkpoint["fitness"]
            )
            initial_algo_n_gen = checkpoint["n_gen"] - 1
        else:
            pass

        max_n_gen = self.args.max_evals // self.args.n_population
        remaining_gen = max_n_gen - initial_algo_n_gen

        self.t = time.time()
        current_population = initial_population

        if current_population is None:
            x = OPS_REGISTRY["sampling"][self.args.placer][self.args.sampling]( 
                self.args, self.placer
            ).do(self.problem, self.args.n_population).get("X")
            sampling = Population.new(X=x) 
        else:
            sampling = current_population
        from pymoo.decomposition.pbi import PBI

        decomposition = PBI()
        adaptive_decomposition = None
        if getattr(self.args, "adaptive_normalization", False):
            if sampling.get("F") is None:
                from pymoo.core.evaluator import Evaluator
                Evaluator().eval(self.problem, sampling)
            
            initial_nadir = np.max(sampling.get("F"), axis=0)
            adaptive_decomposition = AdaptiveDecomposition(PBI(), initial_nadir=initial_nadir)
            decomposition = adaptive_decomposition

        self._algo = MOEAD(
            ref_dirs=self.ref_dirs,
            decomposition=decomposition,
            sampling=sampling,
            crossover=OPS_REGISTRY["crossover"][self.args.placer][self.args.crossover](self.args),
            mutation=OPS_REGISTRY["mutation"][self.args.placer][self.args.mutation](self.args),
            callback=self._save_callback,
        )

        if adaptive_decomposition is not None:
            adaptive_decomposition.set_algo(self._algo)

        res = minimize(
            problem=self.problem,
            algorithm=self._algo,
            termination=("n_gen", remaining_gen),
            verbose=True,
        )
        return res

    def _save_callback(self, algo):
        # compute time
        t_temp = time.time()
        t_eval = t_temp - self.t
        self.t_total += t_eval
        t_each_eval = t_eval / self.args.n_population
        avg_t_each_eval = self.t_total / (self.n_eval + self.args.n_population * 2)
        self.t = t_temp

        macro_pos_all = algo.pop.get("macro_pos")
        Y = algo.pop.get("F")
        X = algo.pop.get("X")

        if not self.start_from_checkpoint:
            self._record_results(Y=Y, 
                                macro_pos_all=macro_pos_all,
                                t_each_eval=t_each_eval, 
                                avg_t_each_eval=avg_t_each_eval,
                                X=X)
        else:
            self.start_from_checkpoint = False

        self._save_checkpoint(
            population=algo.pop.get("X"),
            fitness=algo.pop.get("F"),
            n_gen=self._algo.n_gen 
        )
    
    def _load_checkpoint(self):
        if hasattr(self.args, "checkpoint") and os.path.exists(self.args.checkpoint):
            super()._load_checkpoint()
            with open(os.path.join(self.args.checkpoint, "moead.pkl"), "rb") as f:
                checkpoint = pickle.load(f)
                self.start_from_checkpoint = True
        else:
            checkpoint = None
            self.start_from_checkpoint = False
        
        return checkpoint

    def _save_checkpoint(self, population, fitness, n_gen):
        super()._save_checkpoint()

        with open(os.path.join(self.checkpoint_path, "moead.pkl"), "wb") as f:
            pickle.dump(
                {
                    "population" : population,
                    "fitness" : fitness,
                    "n_gen" : n_gen
                },
                file=f
            )