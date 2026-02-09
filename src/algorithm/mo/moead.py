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

    def set_algo(self, algo):
        self.algo = algo
    
    def do(self, F, weights, **kwargs):
        ideal = kwargs.get('ideal_point')
        if ideal is None:
            return self.method.do(F, weights, **kwargs)

        # Update nadir
        # If algo is available, check pop
        current_F_pop = None
        if self.algo is not None and self.algo.pop is not None:
            current_F_pop = self.algo.pop.get("F")
        
        candidates = []
        if self.nadir is not None:
             candidates.append(np.atleast_2d(self.nadir))
        
        if current_F_pop is not None and len(current_F_pop) > 0:
             candidates.append(current_F_pop)
             
        if F.ndim == 1:
             candidates.append(F.reshape(1, -1))
        else:
             candidates.append(F)
             
        if candidates:
             # Update global nadir estimate
             stack = np.vstack(candidates)
             
             if self.nadir is None:
                 self.nadir = np.zeros(stack.shape[1])
             
             # Robust update ignoring INF values (invalid solutions)
             for i in range(stack.shape[1]):
                 col = stack[:, i]
                 # Filter out values close to INF
                 valid_col = col[col < (INF * 0.9)]
                 if valid_col.size > 0:
                     self.nadir[i] = np.max(valid_col)
                 elif self.nadir[i] == 0:
                     self.nadir[i] = 1.0 # Default fallback if no valid values seen
        
        if self.nadir is None:
             self.nadir = np.ones_like(ideal)
             
        # Normalize
        # Sanitize ideal if needed (though usually min won't pick INF unless all are INF)
        local_ideal = ideal.copy()
        local_ideal[local_ideal > (INF * 0.9)] = 0.0
        
        diff = self.nadir - local_ideal
        diff[diff < 1e-6] = 1e-6
        
        F_norm = (F - local_ideal) / diff
        
        return self.method.do(F_norm, weights, ideal_point=np.zeros_like(ideal), **kwargs)


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
        if getattr(self.args, "adaptive_normalization", False):
            if sampling.get("F") is None:
                from pymoo.core.evaluator import Evaluator
                Evaluator().eval(self.problem, sampling)
            
            initial_nadir = np.max(sampling.get("F"), axis=0)
            decomposition = AdaptiveDecomposition(PBI(), initial_nadir=initial_nadir)

        self._algo = MOEAD(
            ref_dirs=self.ref_dirs,
            decomposition=decomposition,
            sampling=sampling,
            crossover=OPS_REGISTRY["crossover"][self.args.placer][self.args.crossover](self.args),
            mutation=OPS_REGISTRY["mutation"][self.args.placer][self.args.mutation](self.args),
            callback=self._save_callback,
        )

        if getattr(self.args, "adaptive_normalization", False):
            decomposition.set_algo(self._algo)

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