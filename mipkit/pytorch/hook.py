import torch.nn as nn
from collections import OrderedDict
from .utils import convert_pymodule_to_flatten_dict
from ..logging import turn_red
import warnings


class PytorchHook(nn.Module):
    def __init__(self, model, output_layers, *args, **kwargs):
        """Forward Hook is to extract outputs from specific layers.

        Args:
            model (torch.Module): 
            output_layers ([type]): [description]
        """
        super().__init__(*args, **kwargs)
        self.output_layers = output_layers
        self.layer_outputs: dict = OrderedDict()
        self.model = model
        self.model_dict = convert_pymodule_to_flatten_dict(model)
        self.hook_handles = []
        self.register_hook()

    def get_layers_output(self):
        return self.layer_outputs

    def get_output_layers(self):
        return self.output_layers

    def register_hook(self, model, layer_name):
        raise NotImplementedError

    def clear_hook(self):
        for handler in self.hook_handles:
            handler.remove()
        self.hook_handles.clear()

    def clear_hook_outputs(self):
        self.layer_outputs.clear()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _hook_func(self, layer_name):
        def hook(module, input, output):
            self.layer_outputs[layer_name] = output
        return hook


class ForwardHook(PytorchHook):
    def register_hook(self):
        for layer_name in self.output_layers:
            layer_module = self.model_dict.get(layer_name, None)
            if layer_module is None:
                message = turn_red(f'Not found layer name `{layer_name}`')
                warnings.warn(message)
                continue
            self.hook_handles.append(
                layer_module.register_forward_hook(
                    self._hook_func(layer_name)))


class BackwardHook(PytorchHook):
    def register_hook(self):
        for layer_name in self.output_layers:
            layer_module = self.model_dict.get(layer_name, None)
            if layer_module is None:
                message = turn_red(f'Not found layer name `{layer_name}`')
                warnings.warn(message)
                continue
            self.hook_handles.append(
                layer_module.register_backward_hook(
                    self._hook_func(layer_name)))