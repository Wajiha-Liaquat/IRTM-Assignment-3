import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm

from preprocess import preprocess_texts


def find_text_column(df: pd.DataFrame) -> str:
    """Automatically detect which column contains article text."""

    candidates = ['Article', 'Date', 'Heading', 'NewsType']
    for col in candidates:
        if col in df.columns:
            return col

    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        raise ValueError("No valid text column found in dataset.")

    lengths = {c: df[c].astype(str).map(len).mean() for c in text_cols}
    return max(lengths, key=lengths.get)


def build_inverted_index(processed_texts: List[str], doc_ids: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """Build inverted index: term -> [(doc_id, frequency)]"""

    inverted_index = {}

    for i in tqdm(range(len(processed_texts)), desc='Building inverted index', unit='doc'):
        tokens = processed_texts[i].split()
        term_freq = {}

        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        for term, freq in term_freq.items():
            inverted_index.setdefault(term, []).append((doc_ids[i], freq))

    return inverted_index


def build_or_load_index(dataset_path: str, cache_dir: Path):
    """Load from cache if available, otherwise build inverted index from scratch."""

    inv_index_path = cache_dir / 'inverted_index.pkl'
    meta_path = cache_dir / 'index_meta.pkl'

    #LOAD FROM CACHE
    if inv_index_path.exists() and meta_path.exists():
        print("Loading cached inverted index metadata...")

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        # Reload dataframe
        df = pd.read_csv(meta["Articles.csv"])

        with open(inv_index_path, 'rb') as f:
            inverted_index = pickle.load(f)

        return inverted_index, meta['doc_ids'], meta['processed_texts'], df

    #BUILD FROM SCRATCH
    print("Reading dataset...")
    df = pd.read_csv(dataset_path)

    # Detect article column
    text_col = find_text_column(df)

    # Extract texts
    texts = df[text_col].fillna('').astype(str).tolist()

    # Assign doc_ids
    if 'id' in df.columns:
        doc_ids = df['id'].astype(str).tolist()
    else:
        doc_ids = [str(i) for i in range(len(df))]

    # Preprocess
    print("Preprocessing documents...")
    processed_texts = preprocess_texts(texts)

    # Build inverted index
    print("Building inverted index...")
    inverted_index = build_inverted_index(processed_texts, doc_ids)

    # Save index
    print("Saving index...")
    with open(inv_index_path, 'wb') as f:
        pickle.dump(inverted_index, f)

    # Save metadata
    print("Saving metadata...")
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'doc_ids': doc_ids,
            'processed_texts': processed_texts,
            'dataset_path': dataset_path
        }, f)

    return inverted_index, doc_ids, processed_texts, df
