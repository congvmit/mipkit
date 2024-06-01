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

from os import path

from setuptools import find_packages, setup

__author__ = "Cong M. Vo"
__author_email__ = "congvm.it@gmail.com"
__version__ = "1.7.0"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mipkit",
    version=__version__,
    description="mipkit",
    packages=find_packages(),
    # package_data={'mipkit': ['faces/*']},
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author=__author__,
    author_email=__author_email__,
    license="MIT",
    zip_safe=True,
    install_requires=requirements,
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ),
    entry_points={
        "console_scripts": [
            "expandPDF=mipkit.pdf:main",
            "pyfmt=mipkit.fmt:main",
        ]
    },
)
