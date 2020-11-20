import time
import yaml
import warnings
import logging
import argparse
from datetime import datetime
import yaml
import os


class Struct(dict):
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


def load_yaml_config(file_path, todict=False, verbose=True):
    if verbose:
        print('Load yaml config file from', file_path)
    with open(file_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data = Struct(**data)
    if todict:
        return data.todict()
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


def convert_int_to_str(i, n_char=5):
    return f'{i:0{n_char}d}'


if __name__ == '__main__':
    # Testing
    config = load_yaml_config('/home/congvm/Workspace/mipkit/test/config.yaml')
    config = load_yaml_config(
        '/home/congvm/Workspace/mipkit/test/config.yaml', todict=True)
    print(config)
