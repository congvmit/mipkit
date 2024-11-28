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
# [2] https://stackoverflow.com/questions/32456881/getting-values-from-functions-that-run-as-asyncio-tasks
# ===============================================================================
import asyncio
import warnings
from typing import Any, Optional

__all__ = ["set_trace"]


class Debugger:
    @staticmethod
    def set_trace(method=None):
        tracer = None
        if method is not None:
            assert method in ["ipdb", "pdb", "ipython", "bpython"]
            if method == "ipdb":
                import ipdb

                tracer = ipdb.set_trace
            elif method == "pdb":
                import pdb

                tracer = pdb.set_trace
            elif method == "ipython":
                import IPython

                tracer = IPython.embed
            elif method == "bpython":
                import bpython

                tracer = bpython.embed
        return tracer

    @staticmethod
    def run_async_func(func) -> Optional[Any]:
        if not asyncio.get_event_loop().is_running():
            warnings.warn("Event loop is not running.")
            return
        loop = asyncio.get_event_loop()
        task = loop.create_task(func)
        loop.run_until_complete(asyncio.wait([task]))
        return task.result()


try:
    if asyncio.get_event_loop().is_running():
        from . import nest_asyncio

        nest_asyncio.apply()
except RuntimeError:
    pass
set_trace = Debugger.set_trace(method="ipython")
run_async_func = Debugger.run_async_func
