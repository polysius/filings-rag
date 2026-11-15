"""Local embedding and reranking models that run through ONNX Runtime.

The models are downloaded once from Hugging Face into `data/models/`, pinned to a fixed revision.
`onnxruntime` and `tokenizers` come with the `retrieval` extra and are imported only when a model
is created, so the rest of the package works without them.
"""

from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

MODEL_URL = "https://huggingface.co/{repo}/resolve/{revision}/{file}"
DOWNLOAD_TIMEOUT_SECONDS = 600
MAX_TOKENS = 512  # the longest input both models were trained for
BATCH_SIZE = 16

# BGE models search better when a short query starts with this sentence. Documents do not get it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelSpec:
    name: str  # folder under the models directory
    repo: str
    revision: str  # a commit hash, so the files never change underneath us
    files: tuple[str, ...]


# bge-small-en-v1.5 pools the first token ([CLS]) and normalizes the vectors to length 1.
EMBEDDING_MODEL = ModelSpec(
    name="bge-small-en-v1.5",
    repo="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    files=("onnx/model.onnx", "tokenizer.json"),
)
# A cross-encoder reads the query and a passage together and returns one relevance logit.
RERANKER_MODEL = ModelSpec(
    name="ms-marco-MiniLM-L-6-v2",
    repo="cross-encoder/ms-marco-MiniLM-L-6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    files=("onnx/model.onnx", "tokenizer.json"),
)


def model_folder(spec: ModelSpec, models_dir: Path) -> Path:
    """The folder name includes the revision, so files of another revision are never reused."""
    return models_dir / f"{spec.name}-{spec.revision[:8]}"


def download_model(spec: ModelSpec, models_dir: Path, client: httpx.Client | None = None) -> Path:
    """Download the model files that are not on disk yet and return the model folder."""
    folder = model_folder(spec, models_dir)
    client = client or httpx.Client(timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True)
    for file in spec.files:
        target = folder / file
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = MODEL_URL.format(repo=spec.repo, revision=spec.revision, file=file)
        partial = target.with_suffix(target.suffix + ".part")
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as out:
                for piece in response.iter_bytes():
                    out.write(piece)
        partial.replace(target)  # a model file is either complete or absent
    return folder


class _OnnxModel:
    """A tokenizer and an ONNX session. The two model classes below differ in what they feed in."""

    def __init__(self, folder: Path):
        import onnxruntime
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding()
        model_path = str(folder / "onnx" / "model.onnx")
        self._session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_names = {model_input.name for model_input in self._session.get_inputs()}

    def _run(self, encodings) -> np.ndarray:
        """Run one batch of encodings and return the first output of the model."""
        wanted = {
            "input_ids": [e.ids for e in encodings],
            "attention_mask": [e.attention_mask for e in encodings],
            "token_type_ids": [e.type_ids for e in encodings],
        }
        # Models differ in which of these inputs they take, so pass only the ones they list.
        inputs = {
            name: np.array(values, dtype=np.int64)
            for name, values in wanted.items()
            if name in self._input_names
        }
        return self._session.run(None, inputs)[0]


class OnnxEmbedder(_OnnxModel):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors += self._embed(texts[start : start + BATCH_SIZE])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        hidden = self._run(self._tokenizer.encode_batch(texts))  # batch x tokens x dimensions
        first_token = hidden[:, 0, :]
        lengths = np.linalg.norm(first_token, axis=1, keepdims=True)
        return (first_token / lengths).tolist()


class OnnxReranker(_OnnxModel):
    def score(self, query: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), BATCH_SIZE):
            pairs = [(query, text) for text in texts[start : start + BATCH_SIZE]]
            logits = self._run(self._tokenizer.encode_batch(pairs))  # batch x 1
            scores += [float(row[0]) for row in logits]
        return scores


def load_embedder(models_dir: Path) -> OnnxEmbedder:
    return OnnxEmbedder(download_model(EMBEDDING_MODEL, models_dir))


def load_reranker(models_dir: Path) -> OnnxReranker:
    return OnnxReranker(download_model(RERANKER_MODEL, models_dir))
