.PHONY: install run test

install:
	pip install -r requirements.txt

run:
	python scripts/run_analysis.py

test:
	pytest -q
