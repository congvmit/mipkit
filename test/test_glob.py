from mipkit import glob_all_files

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
