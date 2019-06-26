.PHONY: test
test:
	PYTHONPATH="src:${PYTHONPATH}" python3 -m pytest test/
