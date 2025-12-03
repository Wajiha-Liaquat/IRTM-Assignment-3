import argparse
import os
from pathlib import Path

from preprocess import preprocess_texts, preprocess_query
from indexer import build_or_load_index
from retriever import build_or_load_tfidf, retrieve_top_k
from expansion import expand_query_tfidf
from evaluator import interactive_evaluation

CACHE_DIR = Path('./ir_cache')
CACHE_DIR.mkdir(exist_ok=True)

def main(dataset_path: str = '/mnt/data/Articles.csv'):
    # Build or load inverted index and TF-IDF + vectorizer
    inverted_index, doc_ids, processed_texts, df = build_or_load_index("Articles.csv", CACHE_DIR)
    vectorizer, tfidf_matrix = build_or_load_tfidf(processed_texts, CACHE_DIR)

    print('Ready. Enter your query (or type "exit" to quit):')
    query = input('> ').strip()
    if query.lower() in ('exit', 'quit'):
        print('Bye!')
        return

    processed_query = preprocess_query(query)
    expanded = expand_query_tfidf(processed_query, vectorizer, tfidf_matrix, top_k_terms=3)
    if expanded != processed_query:
        print(f'Expanded query: {expanded}')
    else:
        print('No expansion terms found or query terms not in vocabulary.')

    results = retrieve_top_k(expanded, vectorizer, tfidf_matrix, doc_ids, top_k=5)

    print('\nTop 5 retrieved documents:\n')
    for rank, (doc_id, score) in enumerate(results, start=1):
        row = df.iloc[int(doc_id)]

        heading = row.get('Heading', 'N/A')
        article = row.get('Article', 'N/A')
        news_type = row.get('NewsType', 'N/A')
        date = row.get('Date', 'N/A')

        print(f"RANK {rank}")
        print(f"Doc ID   : {doc_id}")
        print(f"Score    : {score:.4f}")
        print(f"Heading  : {heading}")
        print(f"Type     : {news_type}")
        print(f"Date     : {date}")
        print(f"Article  : {article[:500]}...")  # show 500 chars max
        print("-" * 80)

# Evaluation
    interactive_evaluation(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/mnt/data/Articles.csv')
    args = parser.parse_args()
    main(args.dataset)
