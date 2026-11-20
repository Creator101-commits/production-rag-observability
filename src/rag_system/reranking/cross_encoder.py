"""Cross-encoder reranking over retrieved evidence pools."""

from __future__ import annotations

import time

from rag_system.reranking.scorer import PairScorer
from rag_system.retrieval.schema import Evidence


class CrossEncoderReranker:
    """Reranks the top candidate pool with a pairwise relevance scorer.

    Only the first ``max_candidates`` candidates are scored; the rest keep
    their retrieval order after the reranked head.
    """

    def __init__(self, scorer: PairScorer, max_candidates: int = 20) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        self.scorer = scorer
        self.max_candidates = max_candidates

    @property
    def model_id(self) -> str:
        return self.scorer.model_id

    @property
    def revision(self) -> str:
        return self.scorer.revision

    def rerank(
        self,
        query: str,
        candidates: list[Evidence],
        top_k: int = 5,
    ) -> tuple[list[Evidence], float]:
        """Score candidates and return top_k evidence items plus elapsed ms."""
        start_time = time.perf_counter()

        head = candidates[: self.max_candidates]
        tail = candidates[self.max_candidates :]
        scores = self.scorer.score_pairs(query, [c.text for c in head])

        scored = sorted(
            zip(head, scores, strict=True),
            key=lambda item: (-item[1], item[0].rank),
        )

        reranked: list[Evidence] = []
        for r, (cand, score) in enumerate(scored[:top_k]):
            reranked.append(
                Evidence(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    document_title=cand.document_title,
                    page_start=cand.page_start,
                    page_end=cand.page_end,
                    text=cand.text,
                    score=score,
                    rank=r + 1,
                    retriever_name="cross_encoder",
                )
            )

        # Fill remaining slots from the unscored tail, preserving order.
        for cand in tail:
            if len(reranked) >= top_k:
                break
            reranked.append(cand)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return reranked, elapsed_ms
