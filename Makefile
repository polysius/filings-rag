.PHONY: install test lint corpus

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

# Requires SEC_USER_AGENT, e.g. "Your Name your.email@example.com"
corpus:
	filings-rag build-corpus --config configs/companies.yaml --out data
