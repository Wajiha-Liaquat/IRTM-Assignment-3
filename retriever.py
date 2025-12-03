import pickle
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def identity(x):
    return x


def space_tokenizer(text):
    return text.split()


def build_or_load_tfidf(processed_texts: List[str], cache_dir: Path):
    tfidf_path = cache_dir / 'tfidf_data.pkl'

    if tfidf_path.exists():
        print("Loading TF-IDF data from cache...")
        with open(tfidf_path, "rb") as f:
            saved = pickle.load(f)
        return saved["vectorizer"], saved["tfidf_matrix"]

    print("Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        lowercase=False,
        tokenizer=space_tokenizer,
        preprocessor=identity
    )

    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    with open(tfidf_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix}, f)

    return vectorizer, tfidf_matrix


def retrieve_top_k(query: str, vectorizer: TfidfVectorizer, tfidf_matrix, doc_ids: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(doc_ids[i], float(sims[i])) for i in top_idx]
