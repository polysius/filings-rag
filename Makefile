.PHONY: install test lint corpus parse extract eval-retrieval bridge

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

# Run `make bridge` in another terminal first.
extract:
	filings-rag extract --data data

# Needs data/questions.jsonl, see the README
eval-retrieval:
	filings-rag eval-retrieval --data data

# Serves the signed in Claude Code CLI on http://127.0.0.1:8787/v1
bridge:
	python scripts/claude_cli_bridge.py
