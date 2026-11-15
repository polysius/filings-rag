"""Small stand-ins for the embedding and reranking models, so tests run offline."""

import re

import numpy as np

VOCABULARY = ("supplier", "vendor", "revenue", "sales", "lease", "office", "litigation", "cyber")
# Words that mean the same thing share a dimension, like a real embedding model would put them
# close together.
SYNONYMS = {"vendor": "supplier", "sales": "revenue"}


class KeywordEmbedder:
    """One dimension per topic word. A text's vector counts the topics it mentions."""

    def __init__(self):
        self.embedded_texts: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts += texts
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        topics = [SYNONYMS.get(word, word) for word in re.findall(r"\w+", text.lower())]
        counts = np.array([topics.count(word) for word in VOCABULARY], dtype=np.float32)
        length = np.linalg.norm(counts)
        return (counts / length if length else counts).tolist()


class LengthReranker:
    """Prefers shorter texts, which is enough to show that the order was changed."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return [-float(len(text)) for text in texts]
