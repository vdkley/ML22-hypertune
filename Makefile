.PHONY: help install build test lint format

.DEFAULT: help
help:
	@echo "make lint"
	@echo "       run flake8 and mypy"
	@echo "make format"
	@echo "       run isort and black"
	@echo "make help"
	@echo "       print this help message"

lint:
	poetry run flake8 src
	poetry run flake8 hypertune.py
	poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports src
	poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports hypertune.py

format:
	poetry run isort -v hypertune.py
	poetry run isort -v src
	poetry run black src
	poetry run black hypertune.py
