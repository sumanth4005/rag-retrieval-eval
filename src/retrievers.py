"""Three retrievers for a small RAG knowledge base, written from scratch with NumPy.

- TfidfRetriever: cosine similarity over TF-IDF vectors
- BM25Retriever:  Okapi BM25 (k1, b) term scoring
- HybridRetriever: min-max normalised blend of the two
"""
import math
import re
from collections import Counter

import numpy as np

STOPWORDS = set("""a an the and or of to in on for with by at from is are was were be been being it its this that
i me my we our you your he she they them their do does did can could should would will what how when where why which
who whom there here if not no yes any some about into out up down over under than then so as just too very much more
most get got have has had am im""".split())


def tokenize(text):
    """Lowercase, keep letters/numbers, drop stopwords, light suffix stripping."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) > 5 and w.endswith("ing"):
            w = w[:-3]
        elif len(w) > 4 and w.endswith("ed"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class TfidfRetriever:
    def __init__(self, docs):
        self.docs = docs
        toks = [tokenize(d["title"] + " " + d["text"]) for d in docs]
        self.vocab = {t: i for i, t in enumerate(sorted({t for d in toks for t in d}))}
        n = len(docs)
        df = Counter(t for d in toks for t in set(d))
        self.idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[t])) + 1
        self.matrix = np.vstack([self._vec(d) for d in toks])

    def _vec(self, tokens):
        v = np.zeros(len(self.vocab))
        for t, c in Counter(tokens).items():
            if t in self.vocab:
                v[self.vocab[t]] = (1 + math.log(c)) * self.idf[self.vocab[t]]
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def scores(self, query):
        return self.matrix @ self._vec(tokenize(query))


class BM25Retriever:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.toks = [tokenize(d["title"] + " " + d["text"]) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.len = np.array([len(t) for t in self.toks], dtype=float)
        self.avg = self.len.mean()
        n = len(docs)
        df = Counter(t for d in self.toks for t in set(d))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        q = tokenize(query)
        s = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for t in q:
                if t in tf:
                    s[i] += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
        return s


def _minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class HybridRetriever:
    def __init__(self, docs, alpha=0.5):
        self.docs, self.alpha = docs, alpha
        self.tfidf, self.bm25 = TfidfRetriever(docs), BM25Retriever(docs)

    def scores(self, query):
        return self.alpha * _minmax(self.tfidf.scores(query)) + (1 - self.alpha) * _minmax(self.bm25.scores(query))


def top_k(retriever, query, k=3):
    s = retriever.scores(query)
    idx = np.argsort(-s)[:k]
    return [(retriever.docs[i]["id"], float(s[i])) for i in idx]
