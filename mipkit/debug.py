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

# ===============================================================================
# DEBUGGER
#
# References
# [1] https://stackoverflow.com/questions/16867347/step-by-step-debugging-with-ipython
# ===============================================================================
import IPython
import ipdb
import pdb
import os
import sys
from inspect import getframeinfo, stack
from .logging import print_warning


class Debugger():
    def __init__(self, method='ipython'):
        """Debugger with available methods: ipdb, pdb or ipython

        Args:
            method (str, optional): debugger method. Defaults to 'ipdb'.
        """
        self.tracer = self.init_tracer(method)

    def init_tracer(self, method):
        assert method in ['ipdb', 'pdb', 'ipython']
        if method == 'ipdb':
            return ipdb.set_trace
        elif method == 'pdb':
            return pdb.set_trace
        elif method == 'ipython':
            return IPython.embed

    @classmethod 
    def set_trace(cls, method=None):
        caller = getframeinfo(stack()[-1][0])
        print_warning(
            f'Set trace is call from `{caller.filename}` at line: {caller.lineno}')
        if method is not None:
            cls.tracer = cls.init_tracer(method)
        return cls.tracer


set_trace = Debugger.set_trace()
