# NIST RAG Benchmark Report

- git SHA: `94ac5e4c80f5d8845b6f00c7496532db691f19e4`
- GPU: Tesla T4
- Frozen system: hybrid-rrf-k10-w1-1 + rerank=cross_encoder@20, chunk=512 tokens, provider=qwen2.5-3b-fp16, k_ctx=5, abstain_threshold=0.00034306816121190784
- Threshold score scale: cross-encoder relevance score (post-rerank top score); effectively inactive: True
- Generator selection objective: required_point_recall_s.t._all_dev_unanswerables_abstain

## End-to-end test metrics (frozen config, run once)

| Metric | Value |
| :--- | :--- |
| Required-point recall | 86.9% [76.1, 95.0] |
| All points covered | 81.1% [67.6, 91.9] |
| Abstain precision | 0.900 |
| Abstain recall | 1.000 |
| Abstain F1 | 0.947 |
| False-answer rate | 0.000 |
| False-abstain rate | 0.027 |
| Answerable answer rate | 0.973 |
| Citation label validity | 100.0% [100.0, 100.0] |
| Citation page precision | 79.9% [67.1, 91.7] |
| Citation hit rate | 86.1% [72.2, 97.2] |
| NLI claim support, windowed (estimate) | 66.1% (37/56) |
| NLI claim support, whole-chunk (secondary) | 12.5% |
| Abstentions by source | {'citation_validation': 1, 'model_abstain': 9} |
| Parse failures | 0 |
| Schema bound violations | 0 |
| Truncation rate | 0.0% (0/46 hit cap) |
| Prompt tokens total | 132041 |
| Completion tokens total | 5888 |
| GPU-seconds | 6700.41 |

## Retrieval ablations (test, retrieval-only)

| System | Hit@5 | MRR@10 | nDCG@10 |
| :--- | ---: | ---: | ---: |
| bm25 | 0.946 | 0.727 | 0.602 |
| dense | 0.811 | 0.689 | 0.517 |
| hybrid-rrf-k10-w1-1 | 0.919 | 0.753 | 0.607 |
| hybrid-rrf-k10-w1-1+rerank-cross_encoder | 0.973 | 0.883 | 0.617 |
| hybrid-rrf-k10-w1-1+rerank-none | 0.919 | 0.753 | 0.607 |
| hybrid-rrf-k10-w1-2 | 0.865 | 0.751 | 0.591 |
| hybrid-rrf-k10-w2-1 | 0.919 | 0.759 | 0.625 |
| hybrid-rrf-k30-w1-1 | 0.892 | 0.736 | 0.600 |
| hybrid-rrf-k30-w1-2 | 0.892 | 0.730 | 0.588 |
| hybrid-rrf-k30-w2-1 | 0.919 | 0.741 | 0.609 |
| hybrid-rrf-k60-w1-1 | 0.892 | 0.736 | 0.601 |
| hybrid-rrf-k60-w1-2 | 0.892 | 0.732 | 0.589 |
| hybrid-rrf-k60-w2-1 | 0.919 | 0.744 | 0.610 |

## Generator x context ablation (test)

| Candidate | Req-point recall | Abstain F1 | False-abstain | Parse failures | Truncation | Schema violations |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen2.5-3b-fp16__k3 | 0.842 | 0.947 | 0.027 | 1 | 0.000 | 0 |
| qwen2.5-3b-fp16__k5 (frozen) | 0.869 | 0.947 | 0.027 | 0 | 0.000 | 0 |
| qwen2.5-7b-nf4__k3 | 0.910 | 0.941 | 0.000 | 0 | 0.000 | 0 |
| qwen2.5-7b-nf4__k5 | 0.405 | 0.474 | 0.541 | 0 | 0.000 | 0 |

## Per-stage latency (from exported spans)

| Stage | Count | P50 ms | P95 ms | Mean ms |
| :--- | ---: | ---: | ---: | ---: |
| fusion | 70 | 0.11 | 0.14 | 0.11 |
| generation | 280 | 18918.08 | 46551.25 | 23465.50 |
| nli_scoring | 127 | 496.74 | 1481.68 | 659.63 |
| request | 70 | 678.63 | 691.64 | 675.39 |
| rerank | 70 | 664.87 | 676.98 | 661.39 |
| retrieval | 70 | 13.10 | 15.00 | 13.38 |

## Dev selection

### Chunk size

| Tokens | Hit@5 | nDCG@10 | Mean ctx tokens | Chunks |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.842 | 0.641 | 254.8 | 250 |
| 512 | 0.895 | 0.653 | 506.5 | 126 |
| 768 | 0.842 | 0.620 | 758.4 | 84 |

### Retrieval system

| System | nDCG@10 | Hit@5 | MRR@10 |
| :--- | ---: | ---: | ---: |
| bm25 | 0.589 | 0.789 | 0.795 |
| dense | 0.602 | 0.895 | 0.790 |
| hybrid-rrf-k10-w1-1 | 0.654 | 0.895 | 0.839 |
| hybrid-rrf-k10-w1-2 | 0.651 | 0.895 | 0.826 |
| hybrid-rrf-k10-w2-1 | 0.637 | 0.895 | 0.836 |
| hybrid-rrf-k30-w1-1 | 0.653 | 0.895 | 0.839 |
| hybrid-rrf-k30-w1-2 | 0.646 | 0.895 | 0.807 |
| hybrid-rrf-k30-w2-1 | 0.630 | 0.895 | 0.836 |
| hybrid-rrf-k60-w1-1 | 0.653 | 0.895 | 0.839 |
| hybrid-rrf-k60-w1-2 | 0.645 | 0.895 | 0.807 |
| hybrid-rrf-k60-w2-1 | 0.628 | 0.895 | 0.836 |

### Rerank

| Option | nDCG@10 | Hit@5 |
| :--- | ---: | ---: |
| none | 0.654 | 0.895 |
| cross_encoder | 0.668 | 1.000 |

### Generator x context

| Provider | k_ctx | Dev req recall | Dev abstain F1 | All dev unans abstained |
| :--- | ---: | ---: | ---: | :--- |
| qwen2.5-3b-fp16 | 3 | 0.947 | 1.000 | True |
| qwen2.5-3b-fp16 | 5 | 1.000 | 1.000 | True |
| qwen2.5-7b-nf4 | 3 | 0.895 | 0.909 | True |
| qwen2.5-7b-nf4 | 5 | 0.526 | 0.526 | True |

## Limitations

- 70-question set authored by a single person; treat as a smoke
  benchmark, not a broad capability measure.
- Required-point scoring is lexical after normalization; paraphrases
  outside the acceptable-alternative lists count as misses.
- NLI claim support is a model-based estimate, not verified entailment.
