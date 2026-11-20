"""Lexical coverage reranker (offline baseline, not a cross-encoder)."""

from __future__ import annotations

import re
import time

from rag_system.retrieval.schema import Evidence


class LexicalOverlapReranker:
    """Reranks evidence by query term coverage; deterministic and offline."""

    _STOPWORDS = {
        "a", "an", "the", "of", "for", "to", "in", "on", "by", "and", "or",
        "with", "what", "which", "who", "when", "where", "how", "is", "are",
        "was", "were", "do", "does", "did", "should", "can", "could", "would",
        "according", "mentioned", "detailed", "explain", "text", "document",
        "documents", "nist", "ai", "rmf",
    }

    def score_pair(self, query: str, text: str) -> float:
        """Compute lexical coverage relevance between query and evidence."""
        q_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if token not in self._STOPWORDS and not token.isdigit()
        }
        t_tokens = re.findall(r"[a-z0-9]+", text.lower())

        if not q_tokens or not t_tokens:
            return 0.0

        matched_terms = sum(1 for tok in q_tokens if tok in t_tokens)
        coverage_score = matched_terms / len(q_tokens)

        q_words = query.lower().split()
        t_words = text.lower().split()
        q_bigrams = set(zip(q_words[:-1], q_words[1:], strict=False))
        t_bigrams = set(zip(t_words[:-1], t_words[1:], strict=False))
        bigram_bonus = (
            len(q_bigrams.intersection(t_bigrams)) / len(q_bigrams)
            if q_bigrams
            else 0.0
        )

        raw_score = 0.7 * coverage_score + 0.3 * bigram_bonus
        return float(min(1.0, max(0.0, raw_score)))

    def rerank(
        self,
        query: str,
        candidates: list[Evidence],
        top_k: int = 5,
    ) -> tuple[list[Evidence], float]:
        """Rerank candidates and return top_k evidence items with latency ms."""
        start_time = time.perf_counter()

        scored_candidates: list[tuple[float, Evidence]] = []
        for cand in candidates:
            score = self.score_pair(query, cand.text)
            scored_candidates.append((score, cand))

        scored_candidates.sort(key=lambda item: (-item[0], item[1].rank))

        reranked_evidence: list[Evidence] = []
        for r, (score, cand) in enumerate(scored_candidates[:top_k]):
            reranked_evidence.append(
                Evidence(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    document_title=cand.document_title,
                    page_start=cand.page_start,
                    page_end=cand.page_end,
                    text=cand.text,
                    score=score,
                    rank=r + 1,
                    retriever_name="lexical_rerank",
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return reranked_evidence, elapsed_ms
