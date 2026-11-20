# Evaluation set: nist_qa_v1

`nist_qa_v1.jsonl` is the frozen evaluation set for the NIST AI RMF RAG benchmark.

- 70 questions total: 24 dev, 46 test.
- Dev: 19 answerable, 5 unanswerable. Test: 37 answerable, 9 unanswerable.
- Authored by the maintainer from the PDF text; each required point was verified
  to occur on a cited evidence page.
- The dev split is used for all tuning (chunk size, fusion weights, rerank
  option, abstention threshold). The test split is evaluated once with the
  frozen configuration.

## Schema

One JSON object per line:

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `question_id` | string | Stable identifier (`q001`..`q070`). |
| `split` | `"dev"` or `"test"` | Tuning vs held-out partition. |
| `category` | string | Question type (e.g. `direct_lookup`, `synthesis`, `absent_detail`). |
| `difficulty` | string | `easy`, `medium`, or `hard`. |
| `answerable` | boolean | Whether the corpus contains enough evidence to answer. |
| `question` | string | The user question. |
| `evidence` | list | For answerable questions: `{document_id, pages}` entries. `pages` are 1-based physical PDF page indices. Empty for unanswerable questions. |
| `required_points` | list | For answerable questions: a list of points, each a list of acceptable alternative phrases. A point counts as covered when any alternative occurs in the answer after normalization. Empty for unanswerable questions. |

`document_id` values match the ingestion manifest: `nist_ai_100_1` (NIST AI
100-1, 48 pages) and `nist_ai_600_1` (NIST AI 600-1, 64 pages).

This file is frozen ground truth. Do not edit labels or questions; report
suspected errors instead.
