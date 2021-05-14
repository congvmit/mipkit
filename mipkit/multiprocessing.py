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

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


def pool_worker(target, inputs, use_thread=False, num_worker=None, verbose=True):
    """Run target function in multi-process

    Parameters
    ----------
    target : func
        function to excute multi process
    inputs: list
        list of argument of target function
    num_worker: int
    use_thread: bool
        default use pool
    number of worker
    verbose: bool
        True: progress bar
        False: silent

    Returns
    -------
    list of output of func
    """
    if use_thread:
        pool_use = ThreadPool
    else:
        pool_use = Pool

    if num_worker is None:
        num_worker = cpu_count()
    if num_worker != 1:
        if verbose:
            with pool_use(num_worker) as p:
                res = list(tqdm(p.imap(target, inputs), total=len(inputs)))
        else:
            with pool_use(num_worker) as p:
                res = p.map(target, inputs)
    else:
        if verbose:
            res = [target(_input) for _input in tqdm(inputs)]
        else:
            res = [target(_input) for _input in inputs]
    return res
