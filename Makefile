.PHONY: install test lint corpus parse extract eval-retrieval

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

# Requires SEC_USER_AGENT, e.g. "Your Name your.email@example.com"
corpus:
	filings-rag build-corpus --config configs/companies.yaml --out data

parse:
	filings-rag parse-corpus --data data

# Requires Claude API credentials, e.g. ANTHROPIC_API_KEY
extract:
	filings-rag extract --data data

# Needs data/questions.jsonl, see the README
eval-retrieval:
	filings-rag eval-retrieval --data data
