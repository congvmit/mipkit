import yaml
import warnings


class Struct(dict):
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return str(self.__dict__)

    def keys(self):
        return list(self.__dict__.keys())

    def values(self):
        return list(self.__dict__.values())


def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        data = yaml.load(f)
    return Struct(**data)


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
