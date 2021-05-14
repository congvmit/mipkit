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

import numpy as np
import torch


def show_stats(np_arr, return_stats=False, verbose=True):
    stats = {}
    is_bool = (np_arr.dtype == np.bool) or (np_arr.dtype == torch.bool)
    if not is_bool:
        stats['mean'] = np_arr.mean()
        stats['max'] = np_arr.max()
        stats['min'] = np_arr.min()

        if isinstance(np_arr, np.ndarray):
            stats['median'] = np.median(np_arr)
        else:
            stats['median'] = np_arr.median()

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
