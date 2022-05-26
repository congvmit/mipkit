import os
import pkgutil
import sys
import pip
import importlib

# eggs_loader = pkgutil.find_loader('eggs')
# found = eggs_loader is not None
# print(os.__file__)

# path_to_check = os.path.abspath(__file__)
# try:
#     import x
#     import y
# except ModuleNotFoundError as e:
#     print(e)


# from pkgutil import iter_modules
# import importlib


# def module_exists(module_name):
#     return module_name in (name for loader, name, ispkg in iter_modules())


# print(importlib.find_loader("mipkit"))
# print(importlib.exec_module('mipkit'))


# from modulefinder import ModuleFinder
# 
# finder = ModuleFinder()
# finder.run_script(path_to_check)


# print("\n".join(finder.badmodules.keys()))

# print('Loaded modules:')
# for name, mod in finder.badmodules.items():
#     print('%s: ' % name, end='')
#     print(','.join(list(mod.keys())[:3]))


# print(module_exists('x'))


class InstallError(Exception):
    pass

import pkgutil
import pip
import importlib

def _search_and_install(package_name, install_if_not_found=True, required_version=None):
    if pkgutil.find_loader(package_name) is None:
        if install_if_not_found:
            install_package_name = (
                f"{package_name}=={required_version}"
                if required_version
                else f"{package_name}"
            )
            pip.main(["install", install_package_name])
    
    else:
        return importlib.import_module(package_name)
            


np = _search_and_install('numpy')
print(np.ones([1, 2]))

_search_and_install('numpy_1xxx')