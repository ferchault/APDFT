.PHONY: test dist distclean
test:
	PYTHONPATH="src:${PYTHONPATH}" python3 -m pytest test/
format:
	black src
dist:
	echo "rm -rf dist; python3 setup.py sdist bdist_wheel; conda create --yes --name pip-debug python=3.6; source ~/.bashrc; conda activate pip-debug; pip install dist/apdft-*-py3-none-any.whl; python -c 'import apdft'" | bash -i
distclean:
	echo "conda remove --yes --name pip-debug --all" | bash -i
