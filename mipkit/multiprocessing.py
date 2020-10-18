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