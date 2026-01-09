import os
import time
import torch
import numpy as np
import logging
import pickle
import ray
from abc import abstractmethod
from utils.debug import *
from utils.constant import INF
from utils.random_parser import set_state
from src.placer.basic_placer import BasicPlacer
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

class BasicAlgo:
    def __init__(self, args, placer: BasicPlacer, logger) -> None:
        self.args = args
        self.eval_metrics = args.eval_metrics
        self.placer = placer
        self.logger = logger

        self.n_eval = 0
        self.population = None
        self.best_Y = np.zeros(len(self.eval_metrics)) + INF 

        self.nds = NonDominatedSorting()
        self.pareto_front = None

        self.t_total = 0
        self.max_eval_time_second = args.max_eval_time * 60 * 60 
        
        # N: number of final solutions to save for evaluation
        self.K_elite = getattr(args, "n_max_saving_placement", 10)

        self.checkpoint_path = os.path.join(args.result_path, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
    
    @abstractmethod
    def run(self):
        pass
    

    def _record_results(self, Y, macro_pos_all, t_each_eval=0, avg_t_each_eval=0, X=None):
        # Update historical best Y
        current_best_Y = np.min(Y, axis=0)
        self.best_Y = np.minimum(self.best_Y, current_best_Y)

        pop_best_Y = np.min(Y, axis=0)
        pop_avg_Y  = np.mean(Y, axis=0)
        pop_std_Y  = np.std(Y, axis=0)
            
        for idx, (y, m_pos) in enumerate(zip(Y, macro_pos_all)):
            self.n_eval += 1
            
            y_info = "\t".join(
                [f"{key}: {value}" for key, value in zip(self.eval_metrics, y)]
            )
            
            for i, metric in enumerate(self.eval_metrics):
                self.logger.add(f"{metric}/current", y[i])
                self.logger.add(f"{metric}/his_best", self.best_Y[i])
                self.logger.add(f"{metric}/pop_best", pop_best_Y[i])
                self.logger.add(f"{metric}/pop_avg",  pop_avg_Y[i])
                self.logger.add(f"{metric}/pop_std",  pop_std_Y[i])
            
            self.logger.add("Time/each_eval", t_each_eval)
            self.logger.add("Time/avg_each_eval", avg_t_each_eval)
            self.logger.step()

            self.placer.save_metrics(
                current_Y=y,
                n_eval=self.n_eval,
                his_best_Y=self.best_Y,
                pop_best_Y=pop_best_Y,
                pop_avg_Y=pop_avg_Y,
                pop_std_Y=pop_std_Y,
                t_each_eval=t_each_eval,
                avg_t_each_eval=avg_t_each_eval
            )
            
            if self.n_eval >= self.args.max_evals:

                break

    def select_final_solutions(self, population, N=None):
        """
        Select N solutions from the final population using Non-Dominated Sorting and Crowding Distance.
        """
        if N is None:
            N = self.K_elite
        
        # Handle pymoo Result object
        if hasattr(population, "pop"):
            population = population.pop

        # Normalize population input to a list of dicts with 'X', 'Y', 'macro_pos'
        candidates = []
        
        # Handle pymoo Population object (or anything with .get method returning arrays)
        # Note: dicts also have .get, so we check this first but carefully
        if hasattr(population, "get") and not isinstance(population, dict):
             xs = population.get("X")
             ys = population.get("F")
             if ys is None: ys = population.get("Y")
             mps = population.get("macro_pos")
             
             if ys is not None:
                 n = len(ys)
                 for i in range(n):
                     candidates.append({
                         'X': xs[i] if xs is not None else None,
                         'Y': ys[i],
                         'macro_pos': mps[i] if mps is not None else None
                     })

        elif isinstance(population, list):
             # Assume list of objects/dicts that have attributes or keys
             for ind in population:
                 # Check if it's a dict
                 if isinstance(ind, dict):
                     candidates.append(ind)
                 else:
                     # Assume Individual-like object with attributes
                     candidates.append({
                         'X': getattr(ind, 'X', None),
                         'Y': getattr(ind, 'F', getattr(ind, 'Y', None)), # pymoo uses F
                         'macro_pos': getattr(ind, 'macro_pos', None)
                     })
        elif isinstance(population, dict):
             # Assume dict of arrays/lists: {'X': [...], 'Y': [...], 'macro_pos': [...]}
             # Check lengths
             ys = population.get('Y')
             if ys is None:
                 ys = population.get('F')
             xs = population.get('X')
             mps = population.get('macro_pos')
             if ys is not None:
                 n = len(ys)
                 for i in range(n):
                     candidates.append({
                         'X': xs[i] if xs is not None else None,
                         'Y': ys[i],
                         'macro_pos': mps[i] if mps is not None else None
                     })
        else:
            # Fallback for empty or unknown types passed that weren't caught
            if not candidates:
                logging.warning(f"Unknown population format in select_final_solutions: {type(population)}. Return empty.")
                return []

        # Filter out invalid entries
        pool = [c for c in candidates if c['Y'] is not None and c['macro_pos'] is not None]
        
        if not pool:
            return []

        # Deduplicate based on Y
        unique_pool = []
        seen_Y = set()
        for cand in pool:
            y_tuple = tuple(cand['Y'])
            if y_tuple not in seen_Y:
                seen_Y.add(y_tuple)
                unique_pool.append(cand)
        pool = unique_pool
        
        if not pool:
            return []

        # Perform Selection
        Y_all = np.array([p['Y'] for p in pool])
        nds = NonDominatedSorting()
        fronts = nds.do(Y_all)
        
        selected_solutions = []
        for front in fronts:
            if len(selected_solutions) + len(front) <= N:
                for idx in front:
                    selected_solutions.append(pool[idx])
            else:
                # Split front using Crowding Distance
                n_needed = N - len(selected_solutions)
                if n_needed > 0:
                    front_Y = Y_all[front]
                    cd = calc_crowding_distance(front_Y)
                    # Descending sort
                    sorted_indices = np.argsort(-cd)
                    for i in range(n_needed):
                        original_idx = front[sorted_indices[i]]
                        selected_solutions.append(pool[original_idx])
                break
            
            if len(selected_solutions) >= N:
                break
                
        return selected_solutions


    def _save_final_solutions(self, final_solutions):
        if not final_solutions:
            logging.warning("No final solutions to save.")
            return

        logging.info(f"Saving {len(final_solutions)} final solutions to files...")
        
        # Save final solutions data to pickle for analysis
        final_solutions_path = os.path.join(self.checkpoint_path, "final_solutions.pkl")
        try:
            with open(final_solutions_path, 'wb') as f:
                pickle.dump(final_solutions, f)
            logging.info(f"Saved final solutions data to {final_solutions_path}")
        except Exception as e:
            logging.error(f"Failed to save final_solutions.pkl: {e}")

        for i, sol in enumerate(final_solutions):
            sol_id = i + 1
            macro_pos = sol['macro_pos']
            
            # 1. MP Saving: Save simple artifacts (PL, PNG)
            self.placer.save_placement(macro_pos, sol_id)
            self.placer.plot(macro_pos, sol_id)
            
            # 2. GP Saving: Re-run actor if available to generate GP artifacts
            if self.args.eval_gp_hpwl:
                try:
                    # kill existing actors to free resources
                    for actor in self.placer.gp_evaluators:
                        ray.kill(actor)
                    self.placer.gp_evaluators = []
                    # Create a temporary actor for saving elite solutions to ensure isolation
                    actor = self.placer._create_actor()
                    
                    suffix = "def" # GP output is usually DEF
                    if hasattr(self.args, 'benchmark_type'):
                        suffix = "def" if "def" in self.args.benchmark_type else "pl"

                    placement_file = os.path.join(self.placer.placement_save_path, f"gp_{sol_id}.{suffix}")
                    figure_file = os.path.join(self.placer.fig_save_path, f"gp_{sol_id}.png")

                    if self.args.placer == "hpo":
                        ray.get(actor.evaluate_hyper_params.remote(
                            sol['X'],
                            list(macro_pos.keys()),
                            placement_file=placement_file,
                            figure_file=figure_file,
                            save_result=True
                        ))
                    else:
                        ray.get(actor.evaluate_macro_pos.remote(
                            macro_pos,
                            placement_file=placement_file,
                            figure_file=figure_file,
                            save_result=True
                        ))
                    
                    ray.kill(actor)
                    
                except Exception as e:
                    logging.warning(f"Failed to save GP elite solution {sol_id}: {e}")

    def _save_checkpoint(self):
        logging.info("saving checkpoint")

        # logger checkpoint
        self.logger._save_checkpoint(path=self.checkpoint_path)

        # placement and corresponding figure checkpoint
        self.placer._save_checkpoint(checkpoint_path=self.checkpoint_path)

        if self.t_total >= self.max_eval_time_second:
            logging.info(f"Reaching maximun running time ({self.t_total:.2f} >= {self.max_eval_time_second}), exiting")
            exit(0)

        
    def _load_checkpoint(self):
        if hasattr(self.args, "checkpoint") and os.path.exists(self.args.checkpoint):
            logging.info(f"Loading checkpoint from {self.args.checkpoint}")
            log_file = os.path.join(self.args.checkpoint, "log.pkl")
            with open(log_file, 'rb') as log_f:
                log_data = pickle.load(log_f)
            
            
            self.n_eval = len(log_data[f"{self.eval_metrics[0]}/current"])
            
            set_state(log_data)

            for i_eval in range(0, self.n_eval):
                for key, value_lst in log_data.items():
                    if key in ["random", "np_random", "th_random" ,"th_cuda_random"]:
                        continue
                    self.logger.add(key, value_lst[i_eval])
                self.logger.step()

                self.placer.save_metrics(
                    n_eval=i_eval+1,
                    current_Y=np.array([log_data[f"{metric}/current"][i_eval] for metric in self.eval_metrics]),
                    his_best_Y=np.array([log_data[f"{metric}/his_best"][i_eval] for metric in self.eval_metrics]), 
                    pop_best_Y=np.array([log_data[f"{metric}/pop_best"][i_eval] for metric in self.eval_metrics]), 
                    pop_avg_Y=np.array([log_data[f"{metric}/pop_avg"][i_eval] for metric in self.eval_metrics]), 
                    pop_std_Y=np.array([log_data[f"{metric}/pop_std"][i_eval] for metric in self.eval_metrics]),
                    t_each_eval=log_data["Time/each_eval"][i_eval],
                    avg_t_each_eval=log_data["Time/avg_each_eval"][i_eval]
                )

            self.best_Y = np.array([log_data[f"{metric}/his_best"][self.n_eval-1] for metric in self.eval_metrics])
            self.t_total   = sum(log_data["Time/each_eval"])

            self.placer._load_checkpoint(checkpoint_path=self.args.checkpoint)

