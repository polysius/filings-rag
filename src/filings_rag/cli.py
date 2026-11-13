import argparse
from pathlib import Path

from filings_rag.corpus import build_corpus, load_config
from filings_rag.edgar import EdgarClient, user_agent_from_env
from filings_rag.extract_corpus import extract_corpus
from filings_rag.extractor import completer_from_env
from filings_rag.parse_corpus import parse_corpus


def main() -> None:
    parser = argparse.ArgumentParser(prog="filings-rag")
    commands = parser.add_subparsers(dest="command", required=True)

    corpus = commands.add_parser("build-corpus", help="Download filings and XBRL facts")
    corpus.add_argument("--config", type=Path, default=Path("configs/companies.yaml"))
    corpus.add_argument("--out", type=Path, default=Path("data"))

    parse = commands.add_parser("parse-corpus", help="Parse downloaded filings into blocks")
    parse.add_argument("--data", type=Path, default=Path("data"))

    extract = commands.add_parser("extract", help="Extract fields with Claude and score vs XBRL")
    extract.add_argument("--data", type=Path, default=Path("data"))

    args = parser.parse_args()
    if args.command == "build-corpus":
        client = EdgarClient(user_agent_from_env())
        entries = build_corpus(client, load_config(args.config), args.out)
        print(f"Wrote manifest with {len(entries)} filings to {args.out / 'manifest.json'}")
    elif args.command == "parse-corpus":
        results = parse_corpus(args.data)
        failed = sum(r.status == "failed" for r in results)
        print(
            f"Parsed {len(results) - failed} of {len(results)}; see {args.data / 'PARSING_REPORT.md'}"
        )
    elif args.command == "extract":
        run = extract_corpus(completer_from_env(), args.data)
        print(
            f"Scored {len(run.scores)} fields, {len(run.failures)} filings failed; "
            f"see {args.data / 'RESULTS.md'}"
        )
        if run.failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
