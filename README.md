# IRTM-Assignment-3

**One‑file IR system (indexing + TF‑IDF retrieval + query expansion + evaluation)**

This repository contains an implementation of a small Information Retrieval pipeline used for Assignment 3 of the IRTM course. It provides utilities to preprocess text, build an inverted index, compute TF‑IDF representations, run retrieval with cosine similarity, perform automatic query expansion (TF‑IDF based), and interactively evaluate retrieved results.

---

## Contents

```
IRTM-Assignment-3/
├─ Articles.csv                # Raw dataset (Article, Date, Heading, NewsType)
├─ main.py                     # CLI entry point to run retrieval + interactive demo
├─ preprocess.py               # Tokenization, stopword removal, lemmatization (NLTK)
├─ indexer.py                  # Build / load inverted index and metadata
├─ retriever.py                # Build / load TF‑IDF vectorizer and retrieval functions
├─ expansion.py                # Query expansion using TF‑IDF term centroids
├─ evaluator.py                # Evaluation measures & interactive evaluation helper
├─ utils.py                    # small helpers (progress bars)
├─ requirements.txt            # pinned Python package versions
└─ ir_cache/                   # (optional) cached index & TF‑IDF artifacts
   ├─ inverted_index.pkl
   ├─ index_meta.pkl
   └─ tfidf_data.pkl
```

---

## Quick overview

- **Preprocessing**: `preprocess.py` uses NLTK for tokenization, POS tagging and lemmatization. It removes stopwords and non-alphanumeric tokens and returns cleaned, lemmatized text strings used by the rest of the pipeline.

- **Indexing**: `indexer.py` builds an inverted index mapping terms to lists of `(doc_id, frequency)` pairs. It also extracts document ids and stores processed texts. Caching is supported via `ir_cache` to avoid rebuilding on every run.

- **TF‑IDF & Retrieval**: `retriever.py` builds a `TfidfVectorizer` and TF‑IDF matrix from preprocessed texts; retrieval uses cosine similarity to find the top‑k documents for a query.

- **Query expansion**: `expansion.py` implements a TF‑IDF centroid approach: given the query vector, it finds terms whose TF‑IDF term vectors are similar to the query centroid and appends the top terms (default `top_k_terms=3`) to the query.

- **Evaluation**: `evaluator.py` provides `precision@k`, `recall@k`, `MAP`, `nDCG@k` and an `interactive_evaluation` function that prompts the user for relevant doc ids to compute evaluation metrics on the retrieved results.

---

## Requirements

See `requirements.txt` for pinned versions. At minimum the project uses:

- Python 3.10+ (recommended)
- `numpy`, `pandas`, `scikit-learn`, `nltk`, `tqdm`

Install with:

```bash
python -m pip install -r requirements.txt
```

**NLTK data**: The preprocessing module will attempt to download required NLTK resources (`punkt`, `wordnet`, `averaged_perceptron_tagger`) if they are not already installed. If running in an environment without internet access, download these resources beforehand:

```py
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

---

## Usage

Run the demo CLI which builds/loads the index & TF‑IDF models and enters an interactive retrieval loop:

```bash
python main.py --dataset /path/to/Articles.csv
```

Example interaction flow (what `main.py` does):
1. Build or load the inverted index and TF‑IDF artifacts (cache in `./ir_cache`).
2. Prompt for a query from the user.
3. Preprocess the query with the same preprocessing pipeline.
4. Optionally perform query expansion using TF‑IDF (implemented in `expansion.py`).
5. Retrieve top‑k documents with cosine similarity (default k controlled in code).
6. Print brief document metadata (Heading, Date, NewsType, snippet) for inspection.
7. Optionally run `interactive_evaluation` to enter relevant doc ids and compute metrics.

### Example (local)
```
$ python main.py --dataset Articles.csv
Enter query: pakistan economy
Use query expansion? (y/n): y
Top results:
1) doc_id_123  score: 0.4123
   Heading: ...
   Date: ...
   Article: ...

Would you like to evaluate the retrieved results? (y/n): y
Relevant doc ids (comma-separated): doc_id_123, doc_id_456
# shows Precision@5, Recall@5, F1, MAP, nDCG@5
```

---

## Cache files

When the index or TF‑IDF is built it is saved under `ir_cache/` with the following files (used by `build_or_load_index` and `build_or_load_tfidf`):

- `inverted_index.pkl` — pickled inverted index (term -> [(doc_id, frequency), ...])
- `index_meta.pkl` — metadata including `doc_ids`, `processed_texts` and a small portion of the original dataframe
- `tfidf_data.pkl` — pickled `{'vectorizer': vectorizer, 'tfidf_matrix': tfidf_matrix}`

If you want to force a rebuild, remove the appropriate `.pkl` file(s) from `ir_cache/`.

---

## Notes & suggestions for improvement

- **Large datasets / memory**: Some functions (e.g., converting a TF‑IDF matrix transpose to dense arrays for term vectors) may use large amounts of memory for big collections. For large corpora consider using sparse operations, incremental methods or sampling.

- **Tokenizer & normalization**: The current pipeline lemmatizes and removes non‑alphanumeric tokens; you may want to add more normalization (e.g., numbers handling, named entity preservation) depending on the task.

- **Query expansion**: The simple TF‑IDF centroid approach works well for small experiments. Consider using word embeddings (Word2Vec, FastText, or transformer embeddings) for semantically richer expansion.

- **Evaluation harness**: `interactive_evaluation` is manual. Add a script that accepts a labeled qrels file for batch evaluation (multiple queries) if you want automated experiments.
