import argparse
import os
from pathlib import Path

from filings_rag.corpus import build_corpus, load_config
from filings_rag.dense import CachedEmbedder
from filings_rag.edgar import EdgarClient, user_agent_from_env
from filings_rag.extract_corpus import extract_corpus
from filings_rag.extractor import completer_from_env
from filings_rag.onnx_models import EMBEDDING_MODEL, load_embedder, load_reranker
from filings_rag.parse_corpus import parse_corpus
from filings_rag.pgvector_index import PgVectorIndex, connect
from filings_rag.retrieval_eval import QuestionFileError, evaluate_corpus

PG_URL_VARIABLE = "FILINGS_RAG_PG_URL"
MISSING_EXTRA_MESSAGE = 'error: dense search needs `pip install -e ".[retrieval]"`'


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

    retrieval = commands.add_parser(
        "eval-retrieval", help="Compare chunking strategies and retrieval methods"
    )
    retrieval.add_argument("--data", type=Path, default=Path("data"))
    retrieval.add_argument(
        "--questions", type=Path, help="question file, default is questions.jsonl in --data"
    )
    retrieval.add_argument(
        "--dense", action="store_true", help="also score dense and hybrid search (local models)"
    )
    retrieval.add_argument(
        "--rerank", action="store_true", help="also score reranked hybrid search (needs --dense)"
    )
    retrieval.add_argument(
        "--store",
        choices=("memory", "pgvector"),
        default="memory",
        help=f"where dense vectors live; pgvector reads the database URL from {PG_URL_VARIABLE}",
    )
    retrieval.add_argument("--models-dir", type=Path, default=Path("data/models"))

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
    elif args.command == "eval-retrieval":
        options = retrieval_options(args, parser)
        try:
            evaluate_corpus(args.data, args.questions, **options)
        except (OSError, QuestionFileError) as error:
            raise SystemExit(f"error: {error}") from error
        print(f"Wrote {args.data / 'RETRIEVAL_RESULTS.md'}")


def retrieval_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """Turn the dense, rerank and store flags into arguments for `evaluate_corpus`."""
    if (args.rerank or args.store == "pgvector") and not args.dense:
        parser.error("--rerank and --store pgvector need --dense")
    if not args.dense:
        return {}

    # Check the cheap things first, so a mistake does not cost a model download.
    if not (args.data / "manifest.json").exists():
        raise SystemExit(f"error: no manifest.json in {args.data}; run `make corpus` first")
    options: dict = {}
    if args.store == "pgvector":
        options["make_index"] = _pgvector_index_factory(parser)
    try:
        embedder = load_embedder(args.models_dir)
        options["embedder"] = CachedEmbedder(
            embedder, args.data / "embeddings.sqlite", EMBEDDING_MODEL.revision
        )
        if args.rerank:
            options["reranker"] = load_reranker(args.models_dir)
    except ImportError as error:
        raise SystemExit(MISSING_EXTRA_MESSAGE) from error
    return options


def _pgvector_index_factory(parser: argparse.ArgumentParser):
    url = os.environ.get(PG_URL_VARIABLE)
    if not url:
        parser.error(f"--store pgvector needs the database URL in {PG_URL_VARIABLE}")
    try:
        connection = connect(url)
    except ImportError as error:
        raise SystemExit(MISSING_EXTRA_MESSAGE) from error
    except Exception as error:
        message = f"error: cannot connect to the database in {PG_URL_VARIABLE}: {error}"
        raise SystemExit(message) from error

    def make_index(table, chunks, vectors):
        return PgVectorIndex.build(connection, table, chunks, vectors)

    return make_index


if __name__ == "__main__":
    main()
