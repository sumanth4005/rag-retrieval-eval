# RAG Retrieval Evaluation: TF-IDF vs BM25 vs Hybrid

The "R" in RAG decides most answer quality: if the right document isn't retrieved, the LLM can't ground its answer. This repo builds three retrievers **from scratch in NumPy** and measures them on paraphrased customer questions, the way real users actually ask.

## Results

30 banking FAQ documents, 60 test questions (2 paraphrases per document, worded differently from the source text):

| Retriever | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF (cosine) | 76.7% | 91.7% | 0.845 |
| **BM25** (k1=1.5, b=0.75) | **76.7%** | **93.3%** | **0.849** |
| Hybrid (0.5 TF-IDF + 0.5 BM25) | 76.7% | 93.3% | 0.846 |

**What the misses show** (`python run.py` prints them all):

- **Near-duplicate topics** cause most rank-2 misses: domestic vs international wires, overdraft fee vs monthly fee, mobile deposit vs check holds. A re-ranker or metadata filter would help here.
- **Vocabulary gaps** cause the two worst misses: *"charges when I buy something in **euros**"* never mentions "foreign transaction", and *"going to **Mexico**, should I notify the bank"* never mentions "travel notice" (ranked 26th and 27th). Keyword retrieval can't bridge these; this is exactly where embedding search earns its cost.

**Takeaway:** for a small, well-titled knowledge base, BM25 alone gets 93% hit@3. The next step isn't a bigger LLM; it's adding dense embeddings for synonym-style questions, then evaluating again on a fresh question set.

## How it works

- `tokenize` – lowercase, stopword removal, light suffix stripping
- `TfidfRetriever` – log-scaled TF × smoothed IDF, L2-normalised, cosine similarity
- `BM25Retriever` – Okapi BM25 with document-length normalisation
- `HybridRetriever` – min-max normalises both scores and blends them (`alpha`)
- `evaluate` – hit@1, hit@3 and mean reciprocal rank, plus a list of every miss for error analysis

## Run it

```bash
pip install -r requirements.txt
python run.py              # prints the results table and every miss
python -m unittest -v      # tokenizer, ranking sanity check, metric bounds
```

## Data

`data/faq_corpus.jsonl` is a hand-written sample knowledge base for a fictional bank (fees, cards, wires, security). `data/eval_questions.jsonl` holds the test questions with the id of the document that answers each one. Swap in your own documents and questions with the same format to evaluate a real knowledge base.

Built with Python and NumPy.
