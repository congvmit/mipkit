import numpy as np


def show_stats(img_arr, return_stats=False, verbose=True):
    stats = {}
    stats['mean'] = img_arr.mean()
    stats['median'] = img_arr.median()
    stats['max'] = img_arr.max()
    stats['min'] = img_arr.min()
    stats['shape'] = img_arr.shape
    if verbose:
        msg = ''
        for key, val in stats.items():
            msg += f'{key}:{val}'
        msg = msg.strip()
        print(msg)
    if return_stats:
        return stats
