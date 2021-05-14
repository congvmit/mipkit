"""
 The MIT License (MIT)
 Copyright (c) 2021 Cong Vo
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 Provided license texts might have their own copyrights and restrictions
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
"""

import sys
import time
import yaml
import warnings
import logging
import argparse
from datetime import datetime
import yaml
import os
import pytorch_lightning as pl
import torch
import numpy as np
import random
from glob import glob
import argparse
from tqdm import tqdm as _tqdm

import os
from glob import glob


def glob_all_files(folder_dir, ext=None, recursive=False):
    """Glob all files

    Args:
        folder_dir (str): folder directory
        ext (str | list | None), optional): file extension. Defaults to None.

    Returns:
        list: all file paths
    """
    if ext is None:
        return glob(os.path.join(folder_dir, '*.*'))
    elif isinstance(ext, list):
        paths = []
        for e in ext:
            paths.extend(glob(os.path.join(folder_dir, '**.' + str(e))))
        return paths
    elif isinstance(ext, str):
        return glob(os.path.join(folder_dir, '**.' + str(ext)))


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def tqdm(*args, **kwargs):
    # tqdm without multiline
    if hasattr(_tqdm, '__instances'):
        _tqdm._instances.clear()
    return _tqdm(*args, **kwargs)


def generate_datetime():
    """Generate datetime string

    Returns:
        str: datetime string
    """
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    return date_time


# ===============================================================================
# Argparse
# ===============================================================================
class ArgSpace(dict):
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)

    def keys(self):
        return list(self.__dict__.keys())

    def values(self):
        return list(self.__dict__.values())

    def todict(self):
        return self.__dict__


def to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = to_namespace(v)
    return argparse.Namespace(**d)


def load_parser(name=''):
    parser = argparse.ArgumentParser(name=name)
    return parser


def load_yaml_config(file_path, to_dict=False, verbose=True, to_args=True):
    if verbose:
        print('Load yaml config file from', file_path)
    with open(file_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data = ArgSpace(**data)
    if to_dict:
        return data.todict()
    elif to_args:
        return to_namespace(data.todict())
    else:
        return data


def timeit(verbose=True):
    def timeit_decorator(func):
        def timeit_func(*args, **kwargs):
            start_time = time.time()
            results = func(*args, **kwargs)
            if verbose:
                print(
                    f'The function takes {time.time() - start_time:4f} (s) to process')
            return results
        return timeit_func
    return timeit_decorator


def deprecated(message=''):
    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator


def save_config_as_yaml(args: argparse.Namespace,
                        folder_to_save: str,
                        prefix: str = 'config',
                        verbose=True):
    now = datetime.now()
    dt_string = now.strftime(f"{prefix}_%d-%m-%Y_%H:%M:%S.yaml")
    os.makedirs(folder_to_save, exist_ok=True)
    path_to_save = os.path.join(folder_to_save, dt_string)
    if verbose:
        print('Save yaml config file in :', path_to_save)
    with open(path_to_save, 'w') as f:
        yaml.dump(vars(args), f)


def split_seq(seq, k=10):
    ''' Split a given sequence into `k` parts. 
    '''
    length = len(seq)
    n_items = length//k + 1
    return [seq[i*n_items: i*n_items + n_items] for i in range(k)]


def to_str_with_pad(number, n_char=0, pad_value=0):
    """Convert number to string representation with zero padding.

    Args:
        i (int): number.  
        n_char (int, optional): zero padding. Defaults to 0.

    Returns:
        [type]: [description]
    """
    return f'{number:{pad_value}{n_char}d}'


def convert_tensor_to_np(tensor, is_detach=True, to_device='cpu'):
    """Convert tensor to numpy"""
    assert to_device.split(':')[0] in ['cpu', 'cuda']
    if is_detach:
        tensor = tensor.detach()
    return tensor.to(to_device).numpy()


def seed_everything(seed):
    """Seed everything in training process

    Args:
        seed (int): random seed number.
    """
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def glob_all_files(folder_dir, ext=None, recursive=False):
    """Glob all files

    Args:
        folder_dir (str): folder directory
        ext (str | list | None), optional): file extension. Defaults to None.

    Returns:
        list: all file paths
    """
    if ext is None:
        return glob(os.path.join(folder_dir, '*.*'))
    elif isinstance(ext, list):
        paths = []
        for e in ext:
            paths.extend(glob(os.path.join(folder_dir, '**.' + str(e))))
        return paths
    elif isinstance(ext, str):
        return glob(os.path.join(folder_dir, '**.' + str(ext)))
