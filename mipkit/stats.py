import numpy as np
import torch

def show_stats(np_arr, return_stats=False, verbose=True):
    stats = {}
    is_bool = np_arr.dtype == np.bool or np_arr.dtype == torch.bool
    if not is_bool:
        stats['mean'] = np_arr.mean()
        stats['median'] = np_arr.median()
        stats['max'] = np_arr.max()
        stats['min'] = np_arr.min()
    stats['shape'] = np_arr.shape
    stats['dtype'] = np_arr.dtype
    stats['is_tensor'] = torch.is_tensor(np_arr)

    if verbose:
        msg = ''
        for key, val in stats.items():
            msg += f'* {key}: {val}\n'
        msg = msg.strip()
        print(msg)
    if return_stats:
        return stats
