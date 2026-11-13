"""BM25 keyword search over chunks."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from filings_rag.chunking import Chunk

K1 = 1.5  # how quickly repeated terms stop adding score
B = 0.75  # how strongly long chunks are penalised

TOKEN_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Bm25Index:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._term_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._average_length = sum(self._lengths) / len(chunks) if chunks else 0.0
        document_frequency = Counter(term for counts in self._term_counts for term in counts)
        total = len(chunks)
        self._idf = {
            term: math.log(1 + (total - n + 0.5) / (n + 0.5))
            for term, n in document_frequency.items()
        }

    def search(self, query: str, k: int, document: str | None = None) -> list[Hit]:
        """Return the `k` best chunks. With `document`, only chunks of that filing compete."""
        terms = set(tokenize(query))
        hits = []
        for index, chunk in enumerate(self._chunks):
            if document is not None and chunk.document != document:
                continue
            score = self._score(terms, index)
            if score > 0:
                hits.append(Hit(chunk, score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]

    def _score(self, terms: set[str], index: int) -> float:
        counts = self._term_counts[index]
        length_norm = 1 - B + B * self._lengths[index] / self._average_length
        score = 0.0
        for term in terms:
            frequency = counts.get(term, 0)
            if frequency:
                score += self._idf[term] * frequency * (K1 + 1) / (frequency + K1 * length_norm)
        return score
