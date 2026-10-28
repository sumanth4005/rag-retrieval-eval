"""Compare TF-IDF, BM25 and hybrid retrieval on paraphrased questions: python run.py"""
from src.evaluate import evaluate, load_jsonl
from src.retrievers import BM25Retriever, HybridRetriever, TfidfRetriever


def main():
    docs = load_jsonl("data/faq_corpus.jsonl")
    questions = load_jsonl("data/eval_questions.jsonl")
    print(f"{len(docs)} documents, {len(questions)} test questions\n")
    print(f"{'Retriever':<10} {'hit@1':>7} {'hit@3':>7} {'MRR':>7}")
    results = {}
    for name, r in [("TF-IDF", TfidfRetriever(docs)), ("BM25", BM25Retriever(docs)),
                    ("Hybrid", HybridRetriever(docs, alpha=0.5))]:
        m, misses = evaluate(r, questions)
        results[name] = (m, misses)
        print(f"{name:<10} {m['hit@1']:>7.1%} {m['hit@3']:>7.1%} {m['mrr']:>7.3f}")
    best = max(results, key=lambda n: results[n][0]["mrr"])
    print(f"\nMisses at rank 1 for {best} (question -> expected / got, rank of expected):")
    for q, exp, got, rank in results[best][1]:
        print(f"  - {q!r}: {exp} / {got} (rank {rank})")


if __name__ == "__main__":
    main()
