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

import inspect
import functools


# https://stackoverflow.com/questions/3652851/what-is-the-best-way-to-do-automatic-attribute-assignment-in-python-and-is-it-a
# Thank to an implementation from Stackoverflow
def autoargs(*include, **kwargs):
    def _autoargs(func):
        attrs, varargs, varkw, defaults = inspect.getargspec(func)

        def sieve(attr):
            if kwargs and attr in kwargs['exclude']:
                return False
            if not include or attr in include:
                return True
            else:
                return False

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # handle default values
            if defaults:
                for attr, val in zip(reversed(attrs), reversed(defaults)):
                    if sieve(attr):
                        setattr(self, attr, val)
            # handle positional arguments
            positional_attrs = attrs[1:]
            for attr, val in zip(positional_attrs, args):
                if sieve(attr):
                    setattr(self, attr, val)
            # handle varargs
            if varargs:
                remaining_args = args[len(positional_attrs):]
                if sieve(varargs):
                    setattr(self, varargs, remaining_args)
            # handle varkw
            if kwargs:
                for attr, val in kwargs.items():
                    if sieve(attr):
                        setattr(self, attr, val)
            return func(self, *args, **kwargs)

        return wrapper

    return _autoargs