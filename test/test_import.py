import pkgutil
import sys
import os


# eggs_loader = pkgutil.find_loader('eggs')
# found = eggs_loader is not None
# print(os.__file__)
import os

path_to_check = os.path.abspath(__file__)
try:
    import x
    import y
except ModuleNotFoundError as e:
    print(e)


# from pkgutil import iter_modules
# import importlib


# def module_exists(module_name):
#     return module_name in (name for loader, name, ispkg in iter_modules())


# print(importlib.find_loader("mipkit"))
# print(importlib.exec_module('mipkit'))


from modulefinder import ModuleFinder

finder = ModuleFinder()
finder.run_script(path_to_check)


print("\n".join(finder.badmodules.keys()))

# print('Loaded modules:')
# for name, mod in finder.badmodules.items():
#     print('%s: ' % name, end='')
#     print(','.join(list(mod.keys())[:3]))


# print(module_exists('x'))
