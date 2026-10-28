"""Retrieval metrics: hit rate @k and mean reciprocal rank (MRR)."""
import json

from .retrievers import top_k


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(retriever, questions, k_values=(1, 3)):
    hits = {k: 0 for k in k_values}
    rr, misses = 0.0, []
    for q in questions:
        ranked = [doc_id for doc_id, _ in top_k(retriever, q["question"], k=len(retriever.docs))]
        rank = ranked.index(q["relevant_id"]) + 1
        rr += 1 / rank
        for k in k_values:
            hits[k] += rank <= k
        if rank > 1:
            misses.append((q["question"], q["relevant_id"], ranked[0], rank))
    n = len(questions)
    metrics = {f"hit@{k}": hits[k] / n for k in k_values}
    metrics["mrr"] = rr / n
    return metrics, misses
