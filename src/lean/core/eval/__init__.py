"""Retrieval evaluation harness: hit_rate@k, MRR@k, NDCG@k, Recall@k.

Pure Python implementations of standard IR metrics (LlamaIndex-
compatible definitions). No external deps.
"""

from lean.core.eval.runner import (
    EvalResult,
    EvalSample,
    build_eval_dataset,
    evaluate,
    load_curated_dataset,
)

__all__ = [
    "EvalSample",
    "EvalResult",
    "load_curated_dataset",
    "build_eval_dataset",
    "evaluate",
]
