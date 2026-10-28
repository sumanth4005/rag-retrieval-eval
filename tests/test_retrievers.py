import unittest

from src.evaluate import evaluate, load_jsonl
from src.retrievers import BM25Retriever, HybridRetriever, TfidfRetriever, tokenize, top_k

DOCS = load_jsonl("data/faq_corpus.jsonl")


class TestRetrievers(unittest.TestCase):
    def test_tokenize_drops_stopwords_and_stems(self):
        self.assertEqual(tokenize("What are the wires fees?"), ["wire", "fee"])

    def test_exact_title_query_ranks_first(self):
        for cls in (TfidfRetriever, BM25Retriever, HybridRetriever):
            r = cls(DOCS)
            self.assertEqual(top_k(r, "Mobile check deposit", 1)[0][0], "mobile-deposit")

    def test_metrics_are_bounded(self):
        qs = load_jsonl("data/eval_questions.jsonl")
        m, _ = evaluate(BM25Retriever(DOCS), qs)
        self.assertTrue(0 <= m["hit@1"] <= m["hit@3"] <= 1)
        self.assertTrue(0 < m["mrr"] <= 1)


if __name__ == "__main__":
    unittest.main()
