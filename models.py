import math
import sys
from typing import Optional

import torch.nn
import torch.nn as nn

from utilities import Sin


class ResMLP(nn.Module):
    """A simple MLP with residual connections."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_hidden_layers: int,
                 n_hidden_units: int,
                 activation: str = "relu",
                 output_scale: float = 1.0,
                 initialization: str = "default",
                 disable_residual_connections: bool = False,
                 **kwargs):
        super().__init__()

        # Set attributes
        self.input_dim = input_dim
        self.disable_residual_connections = disable_residual_connections  # If True, residual connections are disabled
        self.input_layer = nn.Linear(input_dim, n_hidden_units)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden_units, n_hidden_units) for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(n_hidden_units, output_dim)
        self.output_scale = output_scale
        self.n_colors = output_dim
        
        # Check validity
        assert initialization == 'default' or 'siren' in initialization, 'Initialization must be default or siren.'

        # Set activation function
        if isinstance(activation, list):
            sys.stdout.write(f"Specified activation list: {activation}.\n")
            activation_list = activation
            assert len(
                activation_list) == n_hidden_layers + 1, "Length of activation list must be equal to number of hidden layers + 1."
        else:
            activation_list = [activation] * (n_hidden_layers + 1)

        activation_mapping = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "sin": Sin,
        }
        self.activations = []
        for activation in activation_list:
            assert activation in activation_mapping.keys(), f"Activation {activation} not implemented."
            self.activations.append(activation_mapping[activation]())

        # Set initialization
        if initialization != "default":
            init_split = initialization.split('_')
            if len(init_split) == 1:
                first_layer_scale = 30.0  # Default value from the SIREN paper
            else:
                first_layer_scale = float(init_split[1])
            layers = [self.input_layer] + [hidden_layer for hidden_layer in self.hidden_layers] + [self.output_layer]
            for idx, module in enumerate(layers):
                fan_in = module.in_features
                scale = first_layer_scale if idx == 0 else None
                self.siren_initilization(module, fan_in=fan_in, first_layer_scale=scale)

    def siren_initilization(self, layer: torch.nn.Module, fan_in: int, first_layer_scale: Optional[float] = None):
        """Apply SIREN initialization to a layer.
         Code adapted from https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py"""
        with torch.no_grad():
            if first_layer_scale is not None:
                # We initialize the first layer differently than the rest
                std = 1. / fan_in  # We do not sample directly using std first_layer_scale because the bias is different
            else:
                std = math.sqrt(6. / fan_in)

            layer.weight.uniform_(-std, std)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.uniform_(-std, std)

            if first_layer_scale is not None:
                layer.weight.mul_(first_layer_scale)

    def forward(self, x):
        out = self.activations[0](self.input_layer(x))

        for idx, layer in enumerate(self.hidden_layers):
            activation = self.activations[idx + 1]
            pre_activation = layer(out)
            if not self.disable_residual_connections:
                out = out + activation(pre_activation)
            else:
                out = activation(pre_activation)

        out = self.output_layer(out)
        return self.output_scale * out
