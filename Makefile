.PHONY: check test lint format typecheck build example

PYTHON ?= python

check: lint typecheck test

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests examples typings
	$(PYTHON) -m ruff format --check src tests examples typings

format:
	$(PYTHON) -m ruff format src tests examples typings

typecheck:
	$(PYTHON) -m mypy
	$(PYTHON) -m pyright

build:
	$(PYTHON) -m build

example:
	$(PYTHON) examples/attention.py
