import json
import sys

import pytest

from filings_rag.cli import PG_URL_VARIABLE, main


def run_cli(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["filings-rag", "eval-retrieval", *args])
    main()


def test_rerank_and_pgvector_need_dense(monkeypatch, tmp_path, capsys):
    for flag in (["--rerank"], ["--store", "pgvector"]):
        with pytest.raises(SystemExit) as raised:
            run_cli(monkeypatch, "--data", str(tmp_path), *flag)
        assert raised.value.code == 2
    assert "need --dense" in capsys.readouterr().err


def test_dense_without_a_corpus_fails_before_anything_is_created(monkeypatch, tmp_path):
    data = tmp_path / "nothing-here"

    with pytest.raises(SystemExit, match="make corpus"):
        run_cli(monkeypatch, "--data", str(data), "--dense", "--models-dir", str(tmp_path / "m"))

    assert not data.exists()
    assert not (tmp_path / "m").exists()  # no model was downloaded


def write_manifest(data) -> None:
    data.mkdir(exist_ok=True)
    (data / "manifest.json").write_text(json.dumps([]))


def test_pgvector_store_needs_the_database_url(monkeypatch, tmp_path, capsys):
    write_manifest(tmp_path)
    monkeypatch.delenv(PG_URL_VARIABLE, raising=False)

    with pytest.raises(SystemExit) as raised:
        run_cli(monkeypatch, "--data", str(tmp_path), "--dense", "--store", "pgvector")

    assert raised.value.code == 2
    assert PG_URL_VARIABLE in capsys.readouterr().err
    assert not (tmp_path / "embeddings.sqlite").exists()


def test_an_unreachable_database_gives_a_clear_error_before_models_load(monkeypatch, tmp_path):
    pytest.importorskip("pg8000")
    write_manifest(tmp_path)
    monkeypatch.setenv(PG_URL_VARIABLE, "postgresql://postgres@127.0.0.1:1/postgres")

    with pytest.raises(SystemExit, match="cannot connect to the database"):
        run_cli(
            monkeypatch,
            "--data", str(tmp_path), "--dense", "--store", "pgvector",
            "--models-dir", str(tmp_path / "m"),
        )  # fmt: skip

    assert not (tmp_path / "m").exists()
    assert not (tmp_path / "embeddings.sqlite").exists()
