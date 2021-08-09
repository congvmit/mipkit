from collections import OrderedDict
# import pytorch_lightning as pl
import torch
import numpy as np
import random


def convert_pymodule_to_flatten_dict(model):
    all_layers = OrderedDict()

    def collect_layers(module, all_layers, previous_layer_name=''):
        layers = getattr(module, 'named_children', dict())
        if not isinstance(layers, dict):
            layers = dict(list(layers()))

            if len(layers) > 0:

                if previous_layer_name != '':
                    all_layers[previous_layer_name] = module

                layers_ = layers.copy()
                for layer_name, layer_module in layers_.items():
                    if previous_layer_name != '':
                        lname = previous_layer_name + '.' + layer_name
                    else:
                        lname = layer_name
                    collect_layers(layer_module, all_layers, lname)
            else:
                all_layers[previous_layer_name] = module

    collect_layers(model, all_layers)
    return all_layers


def convert_pymodule_to_tree(model):
    layers = getattr(model, 'named_children', OrderedDict())
    if not isinstance(layers, OrderedDict):
        layers = OrderedDict(list(layers()))
        if len(layers) > 0:
            for layer_name, layer_module in layers.items():
                layers[layer_name] = convert_pymodule_to_tree(layer_module)
        else:
            return model
    return layers


def seed_everything(seed):
    """Seed everything in training process

    Args:
        seed (int): random seed number.
    """
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def tensor_to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()


def get_imagenet_transform():
    """Initialize pytorch transform composer for ImageNet.
       This function is for experiments

    Returns:
        transform: torchvision.transform.Compose
    """
    from torchvision import transforms as T
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    return transform