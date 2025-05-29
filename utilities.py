from bisect import bisect_right
import os
import sys
from typing import Optional, Tuple
from math import floor, ceil

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm
from torch.distributions.gamma import Gamma


class GeneralUtility:
    """Utility functions for general purposes."""

    @staticmethod
    def set_seed(seed: int):
        """
        Sets the seed for the current run.
        :param seed: seed to be used
        """
        wandb.config.update({'seed': seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(seed)  # This works if CUDA not available

    @staticmethod
    def save_model(model: torch.nn.Module, model_identifier: str, tmp_dir: str, sync: bool = False) -> str:
        """
        Saves the model's state_dict to a file.
        :param model_identifier: Name of the file type.
        :type model_identifier: str
        :param sync: Whether to sync the file to wandb.
        :type sync: bool
        :return: Path to the saved model.
        :rtype: str
        """
        fName = f"{model_identifier}_model.pt"
        fPath = os.path.join(tmp_dir, fName)

        # Remove all wrappers from the model
        while hasattr(model, 'base_model'):
            model = model.base_model

        # Only save models in their non-module version, to avoid problems when loading
        if hasattr(model, 'module'):
            model = model.module
        model_state_dict = model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict

        if sync:
            wandb.save(fPath)
        return fPath

    @staticmethod
    def update_config_with_default(configDict, defaultDict):
        """Update config with default values recursively."""
        for key, default_value in defaultDict.items():
            if key not in configDict:
                configDict[key] = default_value
            elif isinstance(default_value, dict):
                configDict[key] = GeneralUtility.update_config_with_default(configDict.get(key, {}), default_value)
        return configDict

    @staticmethod
    @torch.no_grad()
    def convert_to_tiling(points: torch.Tensor, grid_bounds: tuple[float]) -> torch.Tensor:
        """Converts datapoints to tiling, as if box was repeated in every dimension. Every point beyond the box is
         modulo'ed back into the box.
        :param points: Points to convert
        :param grid_bounds: Bounds of the grid in each dimension
        """
        points = points.clone().detach()

        # Get the overhead in each dimension
        overheads = []
        for i, bound in enumerate(grid_bounds):
            axis = points[..., i]
            overhead = axis.sign() * torch.nn.functional.relu(axis.abs() - bound)
            overheads.append(overhead)

        # Add the overhead to the points such as the box was repeated in every dimension
        for i, overhead in enumerate(overheads):
            overhead_pos = overhead[overhead > 0]
            overhead_neg = overhead[overhead < 0]
            bound = grid_bounds[i]
            points[..., i][overhead > 0] = -bound + overhead_pos
            points[..., i][overhead < 0] = bound + overhead_neg

        return points


    @staticmethod
    def prepend_centering_scaling_to_module(model: torch.nn.Module, scaling: Optional[float],
                                            centering: Optional[float]) -> torch.nn.Module:
        """Prepends a scaling layer to the model, if scaling is not None."""

        class CenteringRescalingWrapper(torch.nn.Module):
            def __init__(self, base_model: torch.nn.Module, scale_value: float | None, centering_value: float | None):
                super().__init__()
                self.scale_factor = scale_value or 1.0
                self.centering = centering_value or 0.0
                self.base_model = base_model

            def forward(self, x):
                x = x - self.centering
                x = x * self.scale_factor
                x = self.base_model(x)
                return x

        CenteringRescalingWrapper.__name__ = model._get_name()
        return CenteringRescalingWrapper(base_model=model, scale_value=scaling, centering_value=centering)

    @staticmethod
    def verify_config(config: wandb.Config):
        """Verifies that the config is valid."""

        assert isinstance(config.training['grid_sizes'], list) or isinstance(config.training['grid_sizes'], tuple), f"Grid sizes must be an iterable, found type {type(config.training['grid_sizes'])}, concretely {config.training['grid_sizes']}"
        assert len(config.training['grid_sizes']) == config.dim, "Grid sizes must have the same length as the dimension."
        assert min(config.training['grid_sizes']) >= 1, "Grid sizes must be at least 1."

        assert isinstance(config.training['batch_size'], (int, list)), "Batch size must be an integer or a list."  

        if not config.problem_name == 'HadwigerNelson': # in HadwigerNelson we only pass one distance
            assert isinstance(config.training['colour_distances'], list) or isinstance(config.training['colour_distances'], tuple), "Colour distances must be an iterable."
            assert len(config.training['colour_distances']) == config.n_colours, "Colour distances must have the same length as the number of colours."
        
        assert config.kill_criterion["patience"] is None or \
               config.metrics["log_metrics_every_k_steps"] <= config.kill_criterion[
                   "patience"], "We need to log at least once before we can check if a run should be terminated."
        

    @staticmethod
    def config_has_unknown_params(config: wandb.Config, defaults: dict) -> Tuple[bool, list]:
        """Recursively check if there are unknown config parameters, i.e. parameters that are not in the defaults."""
        does_not_exist = False
        params = []
        for config_key, config_val in config.items():
            if config_key not in defaults:
                does_not_exist = True
                params.append(config_key)
            elif isinstance(config_val, dict):
                if not does_not_exist:
                    does_not_exist, params = GeneralUtility.config_has_unknown_params(config_val, defaults[config_key])
        return does_not_exist, params

    @staticmethod
    def gnormal(*size, p=0.5, device=None, dtype=None, generator=None):
        alpha = torch.ones(*size, dtype=dtype, device=device)
        alpha *= torch.tensor(1 / p, dtype=dtype, device=device)
        beta = torch.tensor(1, dtype=dtype, device=device)
        gamma_distribution = Gamma(alpha, beta)
        r = 2 * torch.bernoulli(0.5 * torch.ones(*size, dtype=dtype, device=device), generator=generator) - 1
        return r * gamma_distribution.sample() ** (1 / p)

    @staticmethod
    def sphere(*size, d=2, p=2, device=None, dtype=None, generator=None):
        g = GeneralUtility.gnormal(*size, d, p=p, device=device, dtype=dtype, generator=generator)
        denom = (((torch.abs(g) ** p).sum(dim=-1)) ** (1 / p)).unsqueeze(-1)
        return g / denom

    @staticmethod
    @torch.no_grad()
    def get_parameter_count(model: torch.nn.Module) -> Tuple[int, int]:
        n_total = 0
        n_nonzero = 0
        param_list = ['weight', 'bias']
        for name, module in model.named_modules():
            for param_type in param_list:
                if hasattr(module, param_type) and not isinstance(getattr(module, param_type), type(None)):
                    p = getattr(module, param_type)
                    n_total += int(p.numel())
                    n_nonzero += int(torch.sum(p != 0))
        return n_total, n_nonzero

    @staticmethod
    def fill_dict_with_none(d):
        for key in d:
            if isinstance(d[key], dict):
                GeneralUtility.fill_dict_with_none(d[key])  # Recursive call for nested dictionaries
            else:
                d[key] = None
        return d
    
    @staticmethod
    def evaluate_on_grid(grid_bounds:tuple[float], 
                         model:torch.nn.Module, 
                         device:torch.device,
                         n_circle_points:int,
                         gridsize:int,
                         dim:int,
                         p_norm:int,
                         concat_colours:bool,
                         colour_distances:list[float],
                         verbose:bool = True,
                         good_coloring:bool = True,
                         tile_grid:bool = False
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Evaluates the model on an equidisant grid of points in the domain 
        for a given list of distances.

        :param grid_bounds: The bounds of the grid in each dimension.
        :param model: The model to evaluate.
        :param device: The device to evaluate the model on.
        :param n_circle_points: The number of circle points to sample per point.
        :param gridsize: The number of gridpoints in each dimension.
        :param dim: The dimension of the space.
        :param p_norm: The norm that induces the distance w.r.t which we sample "unit distance" points.
        :param colour_distances: The distances to avoid for each colour.
        :param concat_colours: If the colours should be concatenated with the input. For vanilla Hadwiger Nelson does not need to happen.

        :return: The colours of the grid points and the number of conflicts per point.
        """

        # Create a meshgrid of the domain
        x_array = torch.linspace(-grid_bounds[0], grid_bounds[0], gridsize)
        y_array = torch.linspace(-grid_bounds[1], grid_bounds[1], gridsize)

        if dim == 2:
            x, y = torch.meshgrid(x_array, y_array, indexing = "xy")
            # flatten it for evaluation of the model
            flattened_input = torch.stack((x.flatten(), y.flatten()), dim=1)

            # total number of samples and how many circle points we sample per point
            n_samples = gridsize**2
        
        elif dim == 3:
            z_array = torch.linspace(-grid_bounds[2], grid_bounds[2], gridsize)
            x, y, z = torch.meshgrid(x_array, y_array, z_array, indexing = "xy")
            flattened_input = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
            n_samples = gridsize**3
        else:
            raise NotImplementedError("Only 2D and 3D are supported.")
        

        # we now evaluate the model on the grid
        # to do this, we need to concatenate the grid with the distances (at least for the general PolyChromaticNumber Case)

        repeated_distances = torch.tensor(colour_distances).expand(n_samples, -1)
        if concat_colours:
            input_with_distances = torch.cat((flattened_input, repeated_distances), dim = 1)
            model_outs = model(input_with_distances.to(device))
            grid_colours = model_outs.argmax(dim = -1)

            if torch.allclose(model_outs.sum(dim=-1), torch.ones(n_samples, device=device)):
                probs = model_outs
            else:
                probs = model_outs.softmax(dim = -1)
            grid_confidences = probs.max(dim = -1).values
        else:
            model_outs = model(flattened_input.to(device))
            grid_colours = model_outs.argmax(dim = -1)

            if torch.allclose(model_outs.sum(dim=-1), torch.ones(n_samples, device=device)):
                probs = model_outs
            else:
                probs = model_outs.softmax(dim = -1)
            grid_confidences = probs.max(dim = -1).values
        
        # now, for each point in the grid, we need to sample the circle points
        # we'll take the same circle points for each point
        unit_circle_points = GeneralUtility.sphere(n_circle_points, d=dim, p=p_norm)
        repeated_circle_points = unit_circle_points[None, :, :].expand(n_samples, -1, -1)

        # the distance for each point is determined by the colour it has
        distances = repeated_distances[torch.arange(n_samples), grid_colours.to("cpu")]

        # now we simply need to multiply the circle points with the distances
        distance_circle_points = repeated_circle_points * distances[:, None, None]

        # to avoid OOM we process the proximity points in batches
        proximity_colours = torch.empty((n_samples, n_circle_points))

        conflicts_per_point = torch.zeros(gridsize**dim, device=device)

        if verbose:
            for i in tqdm(range(n_circle_points)):
                proximity_points = flattened_input + distance_circle_points[:, i, :]
                if tile_grid:
                    proximity_points = GeneralUtility.convert_to_tiling(proximity_points, grid_bounds)
                if concat_colours:
                    proximity_points = torch.cat((proximity_points, repeated_distances), dim = 1)
                    if tile_grid:
                        proximity_points = GeneralUtility.convert_to_tiling(proximity_points, grid_bounds)
                proximity_colours = model(proximity_points.to(device)).argmax(dim = -1)
                conflicts_per_point += (grid_colours == proximity_colours)
        else:
            for i in range(n_circle_points):
                proximity_points = flattened_input + distance_circle_points[:, i, :]
                if tile_grid:
                    proximity_points = GeneralUtility.convert_to_tiling(proximity_points, grid_bounds)
                if concat_colours:
                    proximity_points = torch.cat((proximity_points, repeated_distances), dim = 1)
                    if tile_grid:
                        proximity_points = GeneralUtility.convert_to_tiling(proximity_points, grid_bounds)
                proximity_colours = model(proximity_points.to(device)).argmax(dim = -1)
                conflicts_per_point += (grid_colours == proximity_colours)
       

        # this is returned to calculate different metrics later
        if dim == 2:
            if good_coloring:
                grid_colours = grid_colours.reshape(gridsize, gridsize) 
                last_colour_mask = grid_colours == (model_outs.shape[1] - 1)
                conflicts_per_point = conflicts_per_point.reshape(gridsize, gridsize)
                conflicts_per_point[last_colour_mask] = 0
                return grid_colours, conflicts_per_point, grid_confidences.reshape(gridsize, gridsize)
            else:
                return grid_colours.reshape(gridsize, gridsize), conflicts_per_point.reshape(gridsize, gridsize), grid_confidences.reshape(gridsize, gridsize)
        elif dim == 3:
            return grid_colours.reshape(gridsize, gridsize, gridsize), conflicts_per_point.reshape(gridsize, gridsize, gridsize), grid_confidences.reshape(gridsize, gridsize, gridsize)
        
    @staticmethod
    def evaluate_3D_grid(grid_bounds:tuple[float], 
                         model:torch.nn.Module, 
                         device:torch.device,
                         n_circle_points:int,
                         gridsize:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Essentially a copy of evaluate on grid, but for 3D. 
        Copy is necessary as 3D has some subtleties which would 
        make the already hard to read evaluate_on_grid function even worse.
        Note: Only supports OG Hadwiger Nelson Problem, not PolyChromaticNumber.
        """

        x_array = torch.linspace(-grid_bounds[0], grid_bounds[0], gridsize)
        y_array = torch.linspace(-grid_bounds[1], grid_bounds[1], gridsize)
        z_array = torch.linspace(-grid_bounds[2], grid_bounds[2], gridsize)

        x, y, z = torch.meshgrid(x_array, y_array, z_array, indexing = "xy")

        full_coords = torch.stack((x, y, z), dim=-1)

        grid_colours = torch.zeros(gridsize, gridsize, gridsize, device=device)
        for i in tqdm(range(gridsize)):
            grid_colours[i] = model(full_coords[i].to(device)).argmax(dim = -1)

        circle_points = GeneralUtility.sphere(n_circle_points, d=3, p=2)

        conflicts_per_point = torch.zeros(gridsize, gridsize, gridsize, device=device)

        for i in tqdm(range(n_circle_points)):

            proximity_points = full_coords + circle_points[i]

            proximity_colours = torch.zeros(gridsize, gridsize, gridsize, device=device)
            for j in range(gridsize):
                proximity_colours[j] = model(proximity_points[j].to(device)).argmax(dim = -1)
            conflicts_per_point += (grid_colours == proximity_colours)


        return grid_colours, conflicts_per_point
        

    
    @staticmethod
    def create_conflict_plot(grid_colours:torch.Tensor, 
                             conflicts_per_point:torch.Tensor,
                             grid_confidences:torch.Tensor,
                             grid_bounds:tuple[float],
                             gridsize:int,
                             n_colours:int,
                             save_path:str = None):
                             
        
        conflicts_mask = conflicts_per_point > 0

        x_array = torch.linspace(-grid_bounds[0], grid_bounds[0], gridsize)
        y_array = torch.linspace(-grid_bounds[1], grid_bounds[1], gridsize)

        fig = plt.figure(figsize=(15, 5))

        plt.subplot(131)

        plt.title("Coloring")
        plt.pcolor(x_array, y_array, grid_colours.cpu(), cmap=plt.cm.get_cmap("Pastel1", n_colours), vmin=0, vmax=n_colours - 1)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)

        # set x and y limits
        ax.set_xlim(-grid_bounds[0], grid_bounds[0])
        ax.set_ylim(-grid_bounds[1], grid_bounds[1])

        plt.subplot(132)

        num_conflicting_points = conflicts_mask.cpu().sum()
        plt.title(f"Points with conflicts (Ratio = {num_conflicting_points / gridsize ** 2:.06f})")
        plt.pcolor(x_array, y_array, conflicts_mask.cpu(), cmap=plt.cm.get_cmap("binary", 2), vmin=0, vmax=1)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)

        plt.subplot(133)

        plt.title("Confidence")
        plt.pcolor(x_array, y_array, grid_confidences.detach().cpu(), cmap=plt.cm.rainbow, vmin=0, vmax=1)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)
        plt.colorbar(shrink=0.78)

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def create_good_coloring_plot(grid_colours:torch.Tensor, 
                             conflicts_per_point:torch.Tensor,
                             grid_confidences:torch.Tensor,
                             grid_bounds:tuple[float],
                             gridsize:int,
                             n_colours:int,
                             save_path:str = None,
                             center_coords:torch.Tensor = None,
                             center_logits:torch.Tensor = None):
        
        conflicts_mask = conflicts_per_point > 0
        last_color_mask = grid_colours == (n_colours - 1)

        x_array = torch.linspace(-grid_bounds[0], grid_bounds[0], gridsize)
        y_array = torch.linspace(-grid_bounds[1], grid_bounds[1], gridsize)

        fig = plt.figure(figsize=(15, 5))

        plt.subplot(131)

        plt.title("Coloring")
        plt.pcolor(x_array, y_array, grid_colours.cpu(), cmap=plt.cm.get_cmap("Pastel1", n_colours), vmin=0, vmax=n_colours - 1)
        # plt.colorbar(drawedges=True, location=cbar_location, shrink=shrink)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)

        # set x and y limits
        ax.set_xlim(-grid_bounds[0], grid_bounds[0])
        ax.set_ylim(-grid_bounds[1], grid_bounds[1])


        if center_coords is not None:
            plt.scatter(center_coords[:, 0], center_coords[:, 1], c='black', s=10, alpha=0.5)

        plt.subplot(132)


        plot_mask = torch.zeros_like(conflicts_mask).float()
        plot_mask[conflicts_mask] = 1.
        plot_mask[last_color_mask] = 2.
        num_conflicting_points = conflicts_mask.cpu().sum()
        plt.title(f"Conflicts ratio = {num_conflicting_points / gridsize ** 2:.06f}, Last colour ratio: {last_color_mask.cpu().sum() / gridsize ** 2:.06f}")
        plt.pcolor(x_array, y_array, plot_mask.cpu(), cmap=plt.cm.get_cmap("rainbow", 3), vmin=0, vmax=2)
        #plt.colorbar(drawedges=True, location=cbar_location, shrink=shrink)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)

        plt.subplot(133)

        plt.title("Confidence")
        plt.pcolor(x_array, y_array, grid_confidences.detach().cpu(), cmap=plt.cm.rainbow, vmin=0, vmax=1)
        ax = plt.gca()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis='x', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='x', which='minor', length=3, width=1)  # Small ticks
        ax.tick_params(axis='y', which='major', length=6, width=1.5)  # Big ticks
        ax.tick_params(axis='y', which='minor', length=3, width=1)
        plt.colorbar(shrink=0.78)

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def oom_aware_model_output(model, input_tensor, device):
        """Processes the input tensor in chunks of adaptive size to avoid
        cuda out of memory errors.
        """
        chunk_size = input_tensor.shape[0]
        chunk_reduction_factor = 2
        min_chunk_size = 8

        while chunk_size >= min_chunk_size:
            try:
                chunks = torch.chunk(input_tensor, input_tensor.shape[0] // chunk_size, dim=0)
                results = [model(chunk.to(device)) for chunk in chunks]
                return torch.cat(results, dim=0)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    chunk_size = chunk_size // chunk_reduction_factor
                else:
                    raise e

        raise RuntimeError("Input chunks are too small to process without running out of memory.")
    
    @staticmethod
    def get_chunked_output(model, input_tensor, device, reduction_factor=64):
        """Processes the input tensor in chunks of fixed size to avoid
        cuda out of memory errors.
        """
        chunks = torch.chunk(input_tensor, reduction_factor, dim=0)
        results = [model(chunk.to(device)) for chunk in chunks]
        return torch.cat(results, dim=0)
    

class NotebookUtility:
    """Utility functions for jupyter notebooks."""

    @staticmethod
    def eval_model(model, input, dist, device):
        input = torch.cat((input, dist * torch.ones(input.size(0), 1).to(device)), dim=1)
        with torch.no_grad():
            return model(input).cpu()

    @staticmethod    
    def modulo_repeat_pad(tensor, old_x, old_y, new_x, new_y):
        H = tensor.shape[0]
        W = tensor.shape[1]
        
        resolution = int(H / old_y)
        assert abs(resolution - int(W / old_x)) <= 1, (resolution, int(W / old_x))
        
        x_resolution = int(resolution * new_x)
        y_resolution = int(resolution * new_y)
        
        pad_left = floor((x_resolution-W)/2)
        pad_right = ceil((x_resolution-W)/2)
        pad_top = floor((y_resolution-H)/2)
        pad_bottom = ceil((y_resolution-H)/2)
        
        rows = torch.arange(-pad_top, H + pad_bottom) % H
        cols = torch.arange(-pad_left, W + pad_right) % W
        
        padded_tensor = tensor[rows][:, cols]

        return padded_tensor

    @staticmethod
    def points_at_distance(tensor, dist, base_circle_points, nr_of_points, resolution):
        H, W, C = tensor.shape
        
        output = torch.zeros(H, W, nr_of_points, C)
        
        circle = dist*base_circle_points
        
        for i, (x, y) in list(enumerate(circle)):
            x_shift = int(x * resolution)
            y_shift = int(y * resolution)
            
            rows = torch.arange(y_shift, H + y_shift) % H
            cols = torch.arange(x_shift, W + x_shift) % W
            
            output[:,:,i,:] = tensor[rows][:, cols]
        
        return output

class SequentialSchedulers(torch.optim.lr_scheduler.SequentialLR):
    """
    Repairs SequentialLR to properly use the last learning rate of the previous scheduler when reaching milestones
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(SequentialSchedulers, self).__init__(**kwargs)

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()


# Activation functions
class Sin(torch.nn.Module):
    """Sin activation function."""

    def forward(self, forward_input: torch.Tensor) -> torch.Tensor:
        return torch.sin(forward_input)

class RunTerminator:
    """
    Class to terminate a run if a given metric
    hasn't sufficiently increased / decreased after
    a given amount of steps.
    """

    def __init__(self, **config):

        self.metric = config['metric']  # Metric to monitor
        self.orientation = config['orientation']  # minimize or maximize
        self.threshold = config['threshold']  # Threshold to check against
        self.patience = config['patience']  # Number of steps to wait before checking for termination

        # Internal variables
        self.step = 1
        self.best_metric = None

        self.is_null = config['metric'] is None or \
                       config['orientation'] is None or \
                       config['threshold'] is None or \
                       config['patience'] is None

    def update(self, metrics: dict[str, float]):
        """Update the current metrics."""
        if not self.is_null:
            if self.best_metric is None:
                self.best_metric = metrics[self.metric]
            else:
                if self.orientation == 'minimize':
                    if metrics[self.metric] < self.best_metric:
                        self.best_metric = metrics[self.metric]
                elif self.orientation == 'maximize':
                    if metrics[self.metric] > self.best_metric:
                        self.best_metric = metrics[self.metric]

    def check_termination(self) -> bool:
        """Check if the run should be terminated."""
        if self.is_null:
            return False

        if self.step >= self.patience:
            if self.orientation == "minimize":
                return self.best_metric > self.threshold
            elif self.orientation == "maximize":
                return self.best_metric < self.threshold
            
    def check_termination_and_update(self) -> bool:
        """Check if the run should be terminated and update the step count."""
        termination_status = self.check_termination()
        if termination_status:
            sys.stdout.write(f"Metric {self.metric} reached {self.threshold} at step {self.step}. Terminating run.")
        self.step += 1
        return termination_status



