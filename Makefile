.PHONY: test
test:
	PYTHONPATH="src:${PYTHONPATH}" python3 -m pytest test/
format:
	black src
