import importlib
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import wandb
from torch.optim import SGD, Adam, AdamW
from torchmetrics import MeanMetric
from tqdm.auto import tqdm

from utilities import SequentialSchedulers, GeneralUtility, RunTerminator


class Runner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config: Any, tmp_dir: str):
        """
        Initialize useful variables using config.
        :param config: wandb run config
        :type config: wandb.config.Config
        """
        self.config = config

        GeneralUtility.verify_config(config=self.config)  # Verify the config

        assert not (torch.cuda.device_count() > 1), "DataParallel is not supported yet."
        self.device = torch.device(config.device)
        if 'gpu' in config.device:
            torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True

        # Set a couple useful variables/functions/instances
        self.seed = None
        self.tmp_dir = tmp_dir      
        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device)} for mode in ['train', 'val', 'test']}
        self.run_terminator = RunTerminator(**config.kill_criterion)   
        if isinstance(self.config.training['batch_size'], int):
            self.effective_batch_size = self.config.training['batch_size']
            if self.config.training['sample_all_colours']:
                self.effective_batch_size *= self.config['n_colours']
        elif isinstance(self.config.training['batch_size'], list):
            na, nc, nd, np = self.config.training['batch_size']
            nc = self.config['n_colours'] if nc == 'all' else nc
            self.effective_batch_size = (na if isinstance(na, int) else na[0]) * (nc if isinstance(nc, int) else nc[1]) * (nd if isinstance(nd, int) else nd[2]) * (np if isinstance(np, int) else np[3])

        # Variables to be set
        self.problem = None
        self.optimizer = None
        self.problem_metrics = {}

        sys.stdout.write(f"Using temporary directory {self.tmp_dir}.\n")

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def get_metrics(self) -> dict:
        """
        Returns the metrics for the current epoch.
        :return: dict containing the metrics
        :rtype: dict
        """
        with torch.no_grad():
            n_total, n_nonzero = GeneralUtility.get_parameter_count(model=self.problem.model)

            loggingDict = dict(
                train={metric_name: metric.compute() for metric_name, metric in self.metrics['train'].items()},
                n_total_params=n_total,
                n_nonzero_params=n_nonzero,
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
                problem_metrics=self.problem_metrics,
            )

        return loggingDict

    def get_optimizer(self,
                      optimizer_config: dict,
                      initial_lr: float) -> torch.optim.Optimizer:
        """
        Returns the optimizer.
        :param optimizer_config: Kewords specifying the optimizer.
        :param initial_lr: The initial learning rate

        :return: The optimizer.
 
        """
        params = [param for (name, param) in self.problem.model.named_parameters()]

        wd = optimizer_config['weight_decay'] or 0.
        optimizer_kwargs = dict(params=params,
                                lr=initial_lr,
                                weight_decay=wd)

        if optimizer_config['name'] == 'SGD':
            optimizer = SGD(**optimizer_kwargs,
                            nesterov=wd > 0.,
                            momentum=0.9)
        elif optimizer_config['name'] == 'Adam':
            optimizer = Adam(**optimizer_kwargs)
        elif optimizer_config['name'] == 'AdamW':
            optimizer = AdamW(**optimizer_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_config['name']} not implemented.")
        return optimizer

    def define_optimizer_scheduler(self, optimizer_config: dict):
        """Defines the optimizer and scheduler."""
        assert optimizer_config['learning_rate'] > 0, "Learning rate must be specified as a positive floating number."
        initial_lr = float(optimizer_config['learning_rate'])

        # Define the optimizer
        optimizer = self.get_optimizer(optimizer_config=optimizer_config,
                                       initial_lr=initial_lr)

        # We define a scheduler. All schedulers work on a per-iteration basis
        n_total_iterations = self.config.training['n_steps']

        # Set the initial learning rate
        for param_group in optimizer.param_groups: param_group['lr'] = initial_lr

        # Define the warmup scheduler
        n_warmup_iterations = int(0.05 * n_total_iterations)  # This is now hardcoded to 5% of n_total_iterations
        # As a start factor we use 1e-20, to avoid division by zero when putting 0.
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                                start_factor=1e-20, end_factor=1.,
                                                                total_iters=n_warmup_iterations)
        milestone = n_warmup_iterations

        n_remaining_iterations = n_total_iterations - n_warmup_iterations
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                        start_factor=1.0, end_factor=0.,
                                                        total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [initial_lr for _ in optimizer.param_groups]

        # Define the Sequential Scheduler
        scheduler = SequentialSchedulers(optimizer=optimizer, schedulers=[warmup_scheduler, scheduler],
                                             milestones=[milestone])

        return optimizer, scheduler

    def define_problem(self) -> Any:
        """
        Defines the problem and algorithm to be tackled.
        :return: the problem and algorithm object
        :rtype: tuple[Any, Any]
        """
        problem_module = importlib.import_module(f"problems")
        problem_class = getattr(problem_module, self.config.problem_name, None)
        assert problem_class is not None, f"Problem {self.config.problem_name} not found."
        problem = problem_class(config=self.config,
                                device=self.device,
                                tmp_dir=self.tmp_dir)

        return problem

    def log(self):
        """
        Logs the current training status.
        """
        loggingDict = self.get_metrics()

        # Update the run_terminator
        self.run_terminator.update({**loggingDict['train'], **loggingDict['problem_metrics']})

        # Log and push to Wandb
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"{metric_type}"] = val

        wandb.log(loggingDict)

    def train(self):
        """
        Main training loop in the parametric setting, i.e. where we sample from predefined intervals and update.
        """
        self.problem_metrics = self.problem.get_metrics(list_of_distances=self.config.metrics['eval_distances'])
        self.log(); self.reset_averaged_metrics()

        
        for step in tqdm(range(1, self.config.training['n_steps'] + 1, 1)):
            if step % self.config.metrics['log_metrics_every_k_steps'] == 0:
                self.reset_averaged_metrics()

            self.optimizer.zero_grad()  # Zero the gradient buffers

            # Get new batch
            batch = self.problem.sample_points(n_samples=self.config.training['batch_size'])   

            # Move batch to CUDA
            for key, val in batch.items():
                batch[key] = val.to(device=self.device, non_blocking=True)

            # Compute model outputs (some problems have multiple tensors in a batch which we store in a dict)
            model_outputs = self.problem.get_model_outputs(batch=batch)

            # Compute problem specific loss
            loss_per_sample = self.problem.compute_loss(model_outputs=model_outputs, batch=batch)

            loss = loss_per_sample.mean()
            loss.backward()
            self.optimizer.step(); self.scheduler.step()
            
            # Update the metrics
            self.metrics['train']['loss'](value=loss, weight=self.effective_batch_size)

            # Check if run should be terminated
            if self.run_terminator.check_termination_and_update(): break
            
            is_last_step = step == self.config.training['n_steps']
            should_log_by_type = {log_type: is_last_step or (step % self.config.metrics[f'log_{log_type}_every_k_steps'] == 0) 
                          for log_type in ['imgs', 'metrics', 'model']}

            if should_log_by_type['imgs'] and self.problem.dim == 2: self.problem.log_plots(save_path=self.tmp_dir)
            if should_log_by_type['metrics']:
                self.problem_metrics = self.problem.get_metrics(list_of_distances=self.config.metrics['eval_distances'])
                self.log()
            if should_log_by_type['model']:
                GeneralUtility.save_model(model=self.problem.model, model_identifier=f'step_{step}', tmp_dir=self.tmp_dir, sync=True)


    def run(self):
        """Controls the execution of the script."""
        # We start training from fs
        self.seed = int((os.getpid() + 1) * time.time()) % 2 ** 32
        GeneralUtility.set_seed(seed=self.seed)  # Set the seed

        # Define problem
        self.problem = self.define_problem()

        # Set the model of the problem
        self.problem.set_model(reinit=True)

        # Define optimizer and scheduler
        self.optimizer, self.scheduler = self.define_optimizer_scheduler(optimizer_config=self.config.optimizer)

        # Train    
        self.train()

        
        # Save the trained model and upload it to wandb
        GeneralUtility.save_model(model=self.problem.model, model_identifier='trained', tmp_dir=self.tmp_dir, sync=True)

