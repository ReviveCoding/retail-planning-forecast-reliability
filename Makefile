.PHONY: run run-core run-driver run-specialty run-narrative run-all test test-unit test-integration clean sample

run: run-all

run-core:
	python -m src.main --project project1

run-driver:
	python -m src.main --project project2

run-specialty:
	python -m src.main --project project3

run-narrative:
	python -m src.main --project project4

run-all:
	python -m src.main --project all

sample:
	python -m src.main --project sample

test: test-unit test-integration

test-unit:
	python -m pytest -q tests/unit

test-integration:
	python -m pytest -q tests/integration

clean:
	python scripts/clean_outputs.py
