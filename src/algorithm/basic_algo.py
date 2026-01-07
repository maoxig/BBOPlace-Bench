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
        
        self.elite_pool = [] # storage for elite solutions: [{'X':, 'Y':, 'macro_pos':}]
        self.K_elite = getattr(args, "n_max_saving_placement", 10)

        self.checkpoint_path = os.path.join(args.result_path, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
    
    @abstractmethod
    def run(self):
        pass
    
    def _update_elite_pool(self, Y_batch, macro_pos_batch, X_batch=None):
        # Check inputs
        if Y_batch is None or len(Y_batch) == 0:
            return
        if macro_pos_batch is None:
            logging.warning("macro_pos_batch is None in _update_elite_pool. Skipping update.")
            return
            
        # 1. Create candidates
        candidates = []
        batch_size = len(Y_batch)
        
        # Robustness check for macro_pos length
        if len(macro_pos_batch) != batch_size:
            logging.warning(f"Length mismatch in _update_elite_pool: Y={len(Y_batch)}, macro_pos={len(macro_pos_batch)}. truncating to min.")
            batch_size = min(len(Y_batch), len(macro_pos_batch))

        for i in range(batch_size):
            cand = {
                'Y': Y_batch[i],
                'macro_pos': macro_pos_batch[i],
                'X': X_batch[i] if X_batch is not None else None
            }
            candidates.append(cand)
            
        # 2. Merge and Deduplicate
        full_pool = self.elite_pool + candidates
        
        unique_pool = []
        seen_Y = set()
        for cand in full_pool:
            y_tuple = tuple(cand['Y'])
            if y_tuple not in seen_Y:
                seen_Y.add(y_tuple)
                unique_pool.append(cand)

        pool = unique_pool
        
        if not pool:
            self.elite_pool = []
            return

        # 3. Filter (Non-dominated sorting + Crowding Distance)
        Y_all = np.array([p['Y'] for p in pool])

        nds = NonDominatedSorting()
        fronts = nds.do(Y_all)

        
        new_pool = []
        for front in fronts:
            if len(new_pool) + len(front) <= self.K_elite:
                for idx in front:
                    new_pool.append(pool[idx])
            else:
                # Split front using Crowding Distance
                n_needed = self.K_elite - len(new_pool)
                if n_needed > 0:
                    front_Y = Y_all[front]
                    cd = calc_crowding_distance(front_Y)

                    # Descending sort
                    sorted_indices = np.argsort(-cd)
                    
                    for i in range(n_needed):
                        original_idx = front[sorted_indices[i]]
                        new_pool.append(pool[original_idx])
                break
            
            if len(new_pool) >= self.K_elite:
                break
        
        self.elite_pool = new_pool

    def _record_results(self, Y, macro_pos_all, t_each_eval=0, avg_t_each_eval=0, X=None):
        # Update elite pool
        self._update_elite_pool(Y, macro_pos_all, X)

        pop_best_Y = np.min(Y, axis=0)
        pop_avg_Y  = np.mean(Y, axis=0)
        pop_std_Y  = np.std(Y, axis=0)

        if self.elite_pool:
            elite_Y = np.array([p['Y'] for p in self.elite_pool])
            self.best_Y = np.min(elite_Y, axis=0)
            
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
                self._save_elite_solutions()
                break


    def _save_elite_solutions(self):
        logging.info(f"Saving {len(self.elite_pool)} elite solutions to files...")
        
        for i, sol in enumerate(self.elite_pool):
            sol_id = i + 1
            macro_pos = sol['macro_pos']
            
            # 1. MP Saving: Save simple artifacts (PL, PNG)
            self.placer.save_placement(macro_pos, sol_id)
            self.placer.plot(macro_pos, sol_id)
            
            # 2. GP Saving: Re-run actor if available to generate GP artifacts
            if self.args.eval_gp_hpwl:
                try:
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

        # saving elite pool
        with open(os.path.join(self.checkpoint_path, "elite_pool.pkl"), 'wb') as f:
            pickle.dump(self.elite_pool, f)

        if self.t_total >= self.max_eval_time_second:
            logging.info(f"Reaching maximun running time ({self.t_total:.2f} >= {self.max_eval_time_second}), saving elite solutions and exiting")
            self._save_elite_solutions()
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
                    if key in ("random", "np_random"):
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


            # elite pool
            elite_pool_path = os.path.join(self.args.checkpoint, "elite_pool.pkl")
            if os.path.exists(elite_pool_path):
                with open(elite_pool_path, 'rb') as f:
                    self.elite_pool = pickle.load(f)

