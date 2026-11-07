"""BM25 lexical retrieval engine over document chunks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from rag_system.ingestion.schema import Chunk
from rag_system.retrieval.schema import Evidence


def tokenize(text: str) -> list[str]:
    """Tokenize alphanumeric words."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """Okapi BM25 retriever with Robertson-Sparck Jones IDF weighting."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0.0
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.is_indexed: bool = False

    def index(self, chunks: list[Chunk]) -> BM25Retriever:
        """Build inverted index from document chunks."""
        self.chunks = list(chunks)
        self.inverted_index.clear()
        self.idf.clear()
        self.doc_lens.clear()

        n_docs = len(chunks)
        if n_docs == 0:
            raise ValueError("Cannot index an empty collection of chunks.")

        doc_frequencies: Counter[str] = Counter()

        for idx, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            self.doc_lens.append(len(tokens))
            counts = Counter(tokens)
            for term, count in counts.items():
                self.inverted_index[term].append((idx, count))
                doc_frequencies[term] += 1

        self.avg_doc_len = sum(self.doc_lens) / n_docs

        for term, df in doc_frequencies.items():
            val = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self.idf[term] = max(0.0, val)

        self.is_indexed = True
        return self

    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        """Retrieve top evidence chunks matching lexical query."""
        if top_k <= 0:
            raise ValueError("Result count must be positive.")
        if not self.is_indexed:
            raise RuntimeError("BM25Retriever must be indexed before calling retrieve.")

        tokens = tokenize(query)
        if not tokens:
            return []

        scores: dict[int, float] = defaultdict(float)

        for term in tokens:
            if term not in self.inverted_index:
                continue
            idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, tf in postings:
                d_len = self.doc_lens[doc_idx]
                norm_len = 1.0 - self.b + self.b * (d_len / self.avg_doc_len)
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * norm_len
                scores[doc_idx] += idf * (num / denom)

        ranked = sorted(scores.items(), key=lambda x: (-x[1], self.chunks[x[0]].chunk_id))[:top_k]

        return [
            Evidence(
                chunk_id=self.chunks[doc_idx].chunk_id,
                document_id=self.chunks[doc_idx].document_id,
                document_title=self.chunks[doc_idx].document_title,
                page_start=self.chunks[doc_idx].page_start,
                page_end=self.chunks[doc_idx].page_end,
                text=self.chunks[doc_idx].text,
                score=float(score),
                rank=r + 1,
                retriever_name="bm25",
            )
            for r, (doc_idx, score) in enumerate(ranked)
        ]
