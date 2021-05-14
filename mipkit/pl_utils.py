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

# Pytorch Lightning Utils
import pytorch_lightning as pl
from argparse import ArgumentParser
from argparse import Namespace
from .utils import load_yaml_config
from .logging import print_info

def parse_args(add_pl_args=True, is_notebook=False, print_args=True):
    parser = ArgumentParser()

    # ===============================================================================
    # Training Arguments
    # ===============================================================================
    parser.add_argument(
        "--mode", choices=["train", "test"], type=str, default="train")
    parser.add_argument("-c", "--config_file", type=str,
                        help="configuration file")
    parser.add_argument(
        "-j", "--num_workers", type=int, default=8, help="number of workers"
    )

    # ===============================================================================
    # Learing Rate and Scheduler
    # ===============================================================================
    parser.add_argument("--debug", action="store_true", help="debug")

    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="models",
        help="directory to download pretrained models",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="lightning_logs",
        help="pytorch lightning log directory",
    )

    parser.add_argument("--seed", type=int, default=2021, help="random seed")

    # ===============================================================================
    # Testing
    # ===============================================================================
    parser.add_argument(
        "--resume_ckpt", type=str, default=None, help="resuming checkpoint"
    )

    if add_pl_args:
        parser = pl.Trainer.add_argparse_args(parser)
    else:
        parser.add_argument("--gpus", type=int, help="Number of GPUS")

    # Generate args
    if is_notebook:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    config_args = load_yaml_config(args.config_file, to_args=True)
    args = Namespace(**vars(args), **vars(config_args))

    if print_args:
        print('=================================================')
        print_info('Arguments')
        print(args)
        print('=================================================')
    return args
