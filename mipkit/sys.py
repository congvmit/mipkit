# Copyright (c) 2021 congvm
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

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

if __name__ == '__main__':
    folder_dir = '/Users/congvo/Workspace/mipkit'
    paths = glob_all_files(folder_dir)
    print(paths)
    print(len(paths))


    paths = glob_all_files(folder_dir, ext='py')
    print(paths)
    print(len(paths))


    paths = glob_all_files(folder_dir, ext=['md', 'py'])
    print(paths)
    print(len(paths))


    paths = glob_all_files(folder_dir, ext=['md', 'py'])
    print(paths)
    print(len(paths))