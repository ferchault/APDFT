rm -rf dist
python3 setup.py sdist bdist_wheel
#eval $(conda shell.bash hook)
conda create --yes --name pip-debug python=3.6
conda activate pip-debug
pip install dist/apdft-*-py3-none-any.whl
