import os
from pathlib import Path

import httpx
import pytest

from filings_rag.onnx_models import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    ModelSpec,
    download_model,
)

SPEC = ModelSpec("tiny", "owner/tiny", "abc123", ("onnx/model.onnx", "tokenizer.json"))


def serving(files: dict[str, bytes], requests: list[httpx.Request] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        name = request.url.path.split("/resolve/abc123/")[-1]
        if name in files:
            return httpx.Response(200, content=files[name])
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_the_models_are_pinned_to_a_commit():
    for spec in (EMBEDDING_MODEL, RERANKER_MODEL):
        assert len(spec.revision) == 40 and int(spec.revision, 16) >= 0


def test_download_writes_each_file_from_the_pinned_revision(tmp_path):
    requests: list[httpx.Request] = []
    client = serving({"onnx/model.onnx": b"weights", "tokenizer.json": b"{}"}, requests)

    folder = download_model(SPEC, tmp_path, client)

    assert folder == tmp_path / "tiny"
    assert (folder / "onnx" / "model.onnx").read_bytes() == b"weights"
    assert (folder / "tokenizer.json").read_bytes() == b"{}"
    assert (
        str(requests[0].url) == "https://huggingface.co/owner/tiny/resolve/abc123/onnx/model.onnx"
    )


def test_files_already_on_disk_are_not_downloaded_again(tmp_path):
    client = serving({"onnx/model.onnx": b"weights", "tokenizer.json": b"{}"})
    download_model(SPEC, tmp_path, client)
    requests: list[httpx.Request] = []

    download_model(SPEC, tmp_path, serving({}, requests))

    assert requests == []


def test_a_failed_download_leaves_no_file_behind(tmp_path):
    client = serving({"onnx/model.onnx": b"weights"})  # tokenizer.json answers 404

    with pytest.raises(httpx.HTTPStatusError):
        download_model(SPEC, tmp_path, client)

    assert (tmp_path / "tiny" / "onnx" / "model.onnx").exists()
    assert not (tmp_path / "tiny" / "tokenizer.json").exists()
    assert not list((tmp_path / "tiny").rglob("*.part"))


MODELS_DIR = Path(os.environ.get("FILINGS_RAG_MODELS", "data/models"))
MODELS_PRESENT = all(
    (MODELS_DIR / spec.name / file).exists()
    for spec in (EMBEDDING_MODEL, RERANKER_MODEL)
    for file in spec.files
)


@pytest.mark.skipif(not MODELS_PRESENT, reason="models are not downloaded; see the README")
class TestRealModels:
    """Run against the real ONNX models. Set FILINGS_RAG_MODELS to the folder that holds them."""

    def test_a_paraphrase_is_closer_than_an_unrelated_sentence(self):
        from filings_rag.onnx_models import OnnxEmbedder

        embedder = OnnxEmbedder(MODELS_DIR / EMBEDDING_MODEL.name)
        passage, paraphrase, unrelated = embedder.embed_documents(
            [
                "We depend on a single vendor for the chips used in our products.",
                "Our products rely on one supplier for semiconductors.",
                "The company leases office space in three cities.",
            ]
        )

        def similarity(a, b):
            return sum(x * y for x, y in zip(a, b))

        assert similarity(passage, paraphrase) > similarity(passage, unrelated)
        assert similarity(passage, passage) == pytest.approx(1.0, abs=1e-3)  # unit length

    def test_a_query_finds_the_passage_that_answers_it(self):
        from filings_rag.onnx_models import OnnxEmbedder

        embedder = OnnxEmbedder(MODELS_DIR / EMBEDDING_MODEL.name)
        query = embedder.embed_query("Which supplier does the company depend on?")
        answer, other = embedder.embed_documents(
            [
                "We depend on a single vendor for the chips used in our products.",
                "Quarterly dividends are declared by the board of directors.",
            ]
        )

        assert sum(x * y for x, y in zip(query, answer)) > sum(x * y for x, y in zip(query, other))

    def test_the_reranker_puts_the_relevant_passage_first(self):
        from filings_rag.onnx_models import OnnxReranker

        reranker = OnnxReranker(MODELS_DIR / RERANKER_MODEL.name)
        scores = reranker.score(
            "What are the company's main risk factors?",
            [
                "The cafeteria menu changes every Monday.",
                "Our main risk factors include supply chain disruption and cyber attacks.",
                "The weather was mild throughout the quarter.",
            ],
        )

        assert scores.index(max(scores)) == 1

    def test_long_text_is_cut_to_the_token_limit_instead_of_failing(self):
        from filings_rag.onnx_models import OnnxEmbedder

        embedder = OnnxEmbedder(MODELS_DIR / EMBEDDING_MODEL.name)

        vector = embedder.embed_documents(["supplier " * 5000])[0]

        assert len(vector) == 384
