import argparse
import getpass
import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager

import torch
import wandb
import yaml

from runner import Runner
from utilities import GeneralUtility

defaults = dict(
    # System
    run_id=1,

    # Problem definition
    problem_name='HadwigerNelson', # 'HadwigerNelson' or 'PolychromaticNumber'
    dim=2,
    n_colours=6,

    # Optimizer definition
    optimizer=dict(
        name='AdamW',
        learning_rate=0.001,    # Linear Schedule from learning_rate to 0 after 5% warmup iterations
        weight_decay=0.0
    ),

    # Training definition
    training=dict(
        # General
        n_steps=10000,  # total number of parameter updates
        batch_size=2048,  # Batch size for training
        n_circle_points=8, # number of proximity points to sample for each colour
        tile_grid=False,  # Whether to tile the grid or not (periodic boundary conditions)
        grid_input_scale=1,  # Scale of the grid as how it is input to the network
        loss_fn="prob", # Which loss function to use
        grid_sizes=(6,6),  # Must be a tuple with the grid lengths for each dimension (var dim)
        p_norm=2,  # The norm that induces the distance w.r.t which we sample "unit distance" points
        colour_distances=(1., 1., 1., 1., 1., (0.4, 0.6)), # only relevant for Polychromatic Number. Pass Intervals to include in NN domain
        sample_all_colours=True,  # If True, samples all colours in each batch, else samples only one colour, not relevant for HadwigerNelson
        temperature=5.0, # temperature for weighting the circle points, -1 means infty, i.e. hard selecting the max
        good_coloring=False,  # for lagrangian term for last colour
        good_coloring_weight=0.01, # the \lambda penalizing the last colour
    ),

    # Model definition
    model=dict(
        name='ResMLP',
        n_hidden_layers=4,
        n_hidden_units=32, 
        activation='sin',
        initialization="siren",  # If None, uses default value.
        disable_residual_connections=True,  # If True, disables residual connections between hidden layers,
    ),

    metrics=dict(
        plot_grid_size=256, # grid size for the plots
        val_grid_size=400, # grid size for the metrics
        n_circle_points=128, # the same for plots and metrics
        log_metrics_every_k_steps=2000,  # how often to log metrics
        log_imgs_every_k_steps=10000,  # how often to log images
        log_model_every_k_steps=100000,  # how often to log the model
        eval_distances = [[1.0, 1.0, 1.0, 1.0, 1.0, 0.5]] # what distances to evaluate (only for PolyChromaticNumber)
    ),

    kill_criterion=dict(
        metric=None,  # which metric to monitor
        orientation="minimize",  # whether metric is "good" when it is low or high
        threshold=0.1,  # at which threshold to kill
        patience=10000,  # after how many steps to kill
    )
)

def load_config_from_file_and_merge_with_defaults(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args, unknown = parser.parse_known_args()

    with open(args.config, "r") as f:
        file_config = yaml.safe_load(f)

    # Deep merge: update defaults with loaded file_config
    updated = GeneralUtility.update_config_with_default(file_config, defaults)
    return updated


# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

config = load_config_from_file_and_merge_with_defaults(defaults)

# Configure wandb logging
wandb.init(
    config=config,
    project='test',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = GeneralUtility.update_config_with_default(config, defaults)

# Check if config contains any parameters that are not in defaults, then this should raise an exception
has_unknown_params, params = GeneralUtility.config_has_unknown_params(config, defaults)
assert not has_unknown_params, f"Unknown parameters {params} in config."

ngpus = torch.cuda.device_count()
if ngpus > 0:
    config.update(dict(device='cuda:0'))
else:
    config.update(dict(device='cpu'))


@contextmanager
def tempdir():
    username = getpass.getuser()
    tmp_root = '/fs/local/' + username
    tmp_path = os.path.join(tmp_root, 'tmp')
    if os.path.isdir('/fs/local/') and not os.path.isdir(tmp_root):
        os.makedirs(tmp_root, exist_ok=True)
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path): os.makedirs(tmp_path, exist_ok=True)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        assert 'cluster-' not in os.uname().nodename, "Not allowed to write to /tmp on cluster- machines."
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    # Check if we are running on the GCP cluster, if so, mark as potentially preempted
    is_cluster = 'cluster-' in os.uname().nodename
    is_gcp = 'gpu' in os.uname().nodename and not is_cluster
    if is_gcp:
        print('Running on GCP, marking as preemptable.')
        wandb.mark_preempting()  

    runner = Runner(config=config, tmp_dir=tmp_dir)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
