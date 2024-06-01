import importlib
import pkgutil

import pip


def install_and_import(package_name, install_if_not_found=True, required_version=None):
    if pkgutil.find_loader(package_name) is None:
        if install_if_not_found:
            install_package_name = (
                f"{package_name}=={required_version}" if required_version else f"{package_name}"
            )
            pip.main(["install", install_package_name])

    else:
        return importlib.import_module(package_name)
