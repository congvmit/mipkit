rm -rf build dist
python setup.py bdist_wheel
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*  --verbose