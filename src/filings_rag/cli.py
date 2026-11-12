import argparse
from pathlib import Path

from filings_rag.corpus import build_corpus, load_config
from filings_rag.edgar import EdgarClient, user_agent_from_env


def main() -> None:
    parser = argparse.ArgumentParser(prog="filings-rag")
    commands = parser.add_subparsers(dest="command", required=True)

    corpus = commands.add_parser("build-corpus", help="Download filings and XBRL facts")
    corpus.add_argument("--config", type=Path, default=Path("configs/companies.yaml"))
    corpus.add_argument("--out", type=Path, default=Path("data"))

    args = parser.parse_args()
    if args.command == "build-corpus":
        client = EdgarClient(user_agent_from_env())
        entries = build_corpus(client, load_config(args.config), args.out)
        print(f"Wrote manifest with {len(entries)} filings to {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
