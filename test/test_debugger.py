import numpy as np
import torch
from mipkit import debug_pdb, debug_ipython


def main():
    x = torch.ones([1, 2, 3, 4])
    x_arr = np.ones([1, 2, 3])
    debug_ipython()

if __name__ == '__main__':
    main()