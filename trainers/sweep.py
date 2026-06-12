import sys
import gc

sys.path.append('../')
import os
import json
import time
import threading
from math import prod
import torch
import wandb
import pandas as pd
import collections
import argparse
import warnings
import sklearn.exceptions

from algorithms.algorithms_sfunida import SF_METHODS
from configs.sweep_params import sweep_alg_hparams, sweep_train_hparams
from utils import fix_randomness, starting_logs

from utils import AverageMeter

from trainers.abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    IS_SWEEP = True

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # sweep parameters
        self.num_sweeps = args.num_sweeps
        self.sweep_project_wandb = args.sweep_project_wandb
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize
        self.method = args.da_method
        self.uniDA = args.uniDA

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # Set by sweep(), read by _increment_sweep_counter() inside train()
        self.sweep_registry = None
        self.sweep_key = None

    def _hp_cardinality(self):
        """Total number of unique HP combinations for the current method."""
        alg_card   = prod(len(cfg['values']) for cfg in sweep_alg_hparams[self.da_method].values())
        train_card = prod(len(cfg['values']) for cfg in sweep_train_hparams.values())
        return alg_card * train_card

    def sweep(self, sweep_id=None, name_suffix="", sweep_registry=None):
        self.sweep_key = f"{self.da_method}_{self.backbone}_{self.dataset}{name_suffix}"
        self.sweep_registry = sweep_registry

        # If the requested budget covers the whole HP grid, use grid search so
        # wandb exhausts all combinations without Bayesian early-stopping.
        cardinality = self._hp_cardinality()
        effective_num_sweeps = min(self.num_sweeps, cardinality)
        method = 'grid' if effective_num_sweeps >= cardinality else self.hp_search_strategy
        if method == 'grid':
            print(f"[sweep] HP cardinality={cardinality} ≤ budget={self.num_sweeps} "
                  f"→ switching to grid search (capped at {effective_num_sweeps})")

        sweep_config = {
            'method': method,
            'metric': {'name': self.metric_to_minimize, 'goal': 'maximize'},
            'name': self.sweep_key,
            'parameters': {**sweep_alg_hparams[self.da_method], **sweep_train_hparams}
        }

        # Load registry
        registry = {}
        if sweep_registry and os.path.exists(sweep_registry):
            with open(sweep_registry) as f:
                registry = json.load(f)

        # Priority: registry entry > provided sweep_id > create new sweep
        if sweep_registry and self.sweep_key in registry:
            self.sweep_id = registry[self.sweep_key]["sweep_id"]
            completed = registry[self.sweep_key]["completed"]
            print(f"Resuming: {self.sweep_key} → {self.sweep_id} ({completed}/{self.num_sweeps} done)")
        elif sweep_id is not None:
            self.sweep_id = sweep_id
            completed = 0
        else:
            self.sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)
            completed = 0

        # Write to registry immediately — before agent starts — so a hard shutdown doesn't lose the ID
        if sweep_registry:
            registry[self.sweep_key] = {"sweep_id": self.sweep_id, "completed": completed}
            os.makedirs(os.path.dirname(os.path.abspath(sweep_registry)), exist_ok=True)
            with open(sweep_registry, 'w') as f:
                json.dump(registry, f, indent=2)

        remaining = effective_num_sweeps - completed
        if remaining <= 0:
            print(f"Sweep budget exhausted: {completed}/{effective_num_sweeps} runs. Nothing to run.")
            return
        print(f"Sweep budget: {completed}/{effective_num_sweeps} runs done, launching {remaining} more.")

        # Run the agent in a daemon thread so we can exit if it stalls waiting for jobs.
        # _job_active suppresses the idle timer while a run is executing, so long
        # training runs are never interrupted — only genuine "waiting" periods are timed.
        AGENT_IDLE_TIMEOUT = 300  # seconds with no active job before giving up
        self._job_active = False
        self._last_job_time = time.time()

        agent_thread = threading.Thread(
            target=lambda: wandb.agent(
                self.sweep_id, self._train_tracked, count=remaining,
                entity=self.wandb_entity, project=self.sweep_project_wandb),
            daemon=True)
        agent_thread.start()

        while agent_thread.is_alive():
            agent_thread.join(timeout=30)
            if not self._job_active:
                idle = time.time() - self._last_job_time
                if idle > AGENT_IDLE_TIMEOUT:
                    print(f"[sweep] No new job for {idle:.0f}s — "
                          f"agent stuck (grid exhausted / Bayes converged). Moving on.")
                    break  # daemon thread dies with the process

        self._save_best_hparams()

    def _train_tracked(self):
        """Wraps train() to maintain the idle timer accurately."""
        self._job_active = True
        self._last_job_time = time.time()
        try:
            self.train()
        finally:
            self._job_active = False
            self._last_job_time = time.time()  # reset after completion; waiting period starts now

    def _increment_sweep_counter(self):
        if not self.sweep_registry or not self.sweep_key:
            return
        registry = {}
        if os.path.exists(self.sweep_registry):
            with open(self.sweep_registry) as f:
                registry = json.load(f)
        if self.sweep_key in registry:
            registry[self.sweep_key]["completed"] += 1
            with open(self.sweep_registry, 'w') as f:
                json.dump(registry, f, indent=2)

    def _save_best_hparams(self, save_dir=None):
        """After a sweep finishes, query wandb for the best run and persist its HP."""
        if save_dir is None:
            subfolder = "sfunida" if self.da_method in SF_METHODS else "unida"
            save_dir = os.path.join("configs/best_hparams", subfolder)
        out_dir = os.path.join(self.home_path, save_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"best_hparams_{self.backbone}.json")

        data = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                data = json.load(f)

        try:
            api = wandb.Api()
            sweep_path = f"{self.wandb_entity}/{self.sweep_project_wandb}/{self.sweep_id}"
            sweep_obj = api.sweep(sweep_path)
            best_run = sweep_obj.best_run()
        except Exception as e:
            print(f"[best_hparams] Could not retrieve sweep results from wandb: {e}")
            return

        if best_run is None:
            print(f"[best_hparams] No completed runs found for sweep {self.sweep_id}.")
            return

        best_score = best_run.summary.get(self.metric_to_minimize, None)
        best_hparams = dict(best_run.config)
        print(f"[best_hparams] Best run: {best_run.name}  {self.metric_to_minimize}={best_score}")

        if self.dataset not in data:
            data[self.dataset] = {}
        data[self.dataset][self.da_method] = best_hparams

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[best_hparams] Saved to {out_path}")


    def train(self):
        run = wandb.init(config=self.hparams)
        self.hparams = dict(wandb.config)
        print(self.hparams)
        # create tables for results and risks
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        if self.uniDA:
            results_columns.append('H-score')
            results_columns.append("acc_c")
            results_columns.append("acc_p")
            results_columns.append("acc_mix")
        risks_columns = ["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"]

        # table with metrics
        table_results = pd.DataFrame(columns=results_columns)
        # table with risks
        table_risks = pd.DataFrame(columns=risks_columns)


        for sc_id, (src_id, trg_id) in enumerate(self.dataset_configs.scenarios):
            for run_id in range(self.num_runs):
                # set random seed and create logger
                fix_randomness(run_id)
                self.logger, self.scenario_log_dir = starting_logs( self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id)

                # average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Free previous run's resources before loading new data
                for attr in ('src_train_dl', 'src_test_dl', 'trg_train_dl', 'trg_test_dl', 'algorithm'):
                    if hasattr(self, attr):
                        delattr(self, attr)
                gc.collect()
                torch.cuda.empty_cache()

                # load data and train model
                self.load_data(src_id, trg_id, sc_id)

                # initiate the domain adaptation algorithm
                self.initialize_algorithm()

                # Train the domain adaptation algorithm
                if self.da_method in SF_METHODS:
                    src_ckpt = self._get_source_ckpt_path(src_id)
                    if os.path.exists(src_ckpt):
                        self.algorithm.load_source_model(
                            torch.load(src_ckpt, map_location=self.device))
                        self.logger.info(f"Loaded source model from {src_ckpt}")
                    else:
                        self._train_source_model(src_ckpt)
                    self.last_model, self.best_model = self.algorithm.update(
                        None, self.trg_train_dl, self.loss_avg_meters, self.logger)
                else:
                    self.last_model, self.best_model = self.algorithm.update(
                        self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)

                # calculate metrics and risks
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)
                handlers = self.logger.handlers[:]
                for handler in handlers:
                    self.logger.removeHandler(handler)
                    handler.close()

        # calculate overall metrics and risks
        table_results = self.average_run_rusults(table_results)
        table_risks = self.average_run_rusults(table_risks)
        # Save tables to file if needed
        self.save_sweep_tables_to_file(table_results, self.sweep_id, run.name, 'results')
        self.save_sweep_tables_to_file(table_risks, self.sweep_id, run.name, 'risks')

        #print(table_results)
        table_results = wandb.Table(dataframe=table_results)
        #print(table_risks)
        table_risks = wandb.Table(dataframe=table_risks)

        total_results, summary_metrics = self.calculate_avg_std_wandb_table(table_results)
        total_risks, summary_risks = self.calculate_avg_std_wandb_table(table_risks)

        # log results to WandB
        self.wandb_logging(total_results, total_risks, summary_metrics, summary_risks)

        '''for artifact in run.logged_artifacts():
            artifact.delete()'''
        # increment counter only on clean completion (crash/interrupt never reach this line)
        self._increment_sweep_counter()
        run.finish()

