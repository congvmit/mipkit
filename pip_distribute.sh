__='
   This is the default license template.
   
   File: pip_distribute.sh
   Author: congvm
   Copyright (c) 2020-2021 congvm
   
   To edit this license information: Press Ctrl+Shift+P and press 'Create new License Template...'.
'

rm -rf build dist
python setup.py bdist_wheel
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*  --verbose