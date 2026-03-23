.PHONY: install install-dev test lint typecheck format serve eval clean build publish publish-test clean-build

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=openbias --cov-report=term-missing

lint:
	ruff check openbias/ tests/

typecheck:
	mypy openbias/

format:
	ruff check --fix openbias/ tests/
	ruff format openbias/ tests/

serve:
	openbias serve

eval:
	openbias eval

validate:
	openbias validate $(file)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-build:
	rm -rf dist/ build/ *.egg-info

build: clean-build
	python -m build
	twine check dist/*

publish-test: build
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*
