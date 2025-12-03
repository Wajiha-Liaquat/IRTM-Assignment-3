import numpy as np
from typing import List, Tuple, Set


def precision_at_k(retrieved: List[str], relevant_set: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for d in retrieved_k if d in relevant_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant_set: set, k: int) -> float:
    if len(relevant_set) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for d in retrieved_k if d in relevant_set)
    return hits / len(relevant_set)


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def average_precision(retrieved: List[str], relevant_set: set) -> float:
    hits = 0
    score = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            hits += 1
            score += hits / i
    if hits == 0:
        return 0.0
    return score / len(relevant_set)


def dcg_at_k(retrieved: List[str], relevant_set: set, k: int) -> float:
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        rel = 1 if doc in relevant_set else 0
        dcg += (2**rel - 1) / np.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: List[str], relevant_set: set, k: int) -> float:
    dcg = dcg_at_k(retrieved, relevant_set, k)
    ideal_rels = [1] * min(len(relevant_set), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1) / np.log2(i + 1)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def interactive_evaluation(results: List[Tuple[str, float]]):
    print('Would you like to evaluate the retrieved results?')
    yn = input('(y/n): ').strip().lower()
    if yn != 'y':
        print('Done.')
        return
    retrieved_ids = [doc for doc, _ in results]
    print('Please provide the relevant document ids (comma-separated).')
    rel_input = input('Relevant doc ids (comma-separated): ').strip()
    rel_ids = [x.strip() for x in rel_input.split(',') if x.strip()]
    relevant_set = set(rel_ids)

    precision = precision_at_k(retrieved_ids, relevant_set, k=5)
    recall = recall_at_k(retrieved_ids, relevant_set, k=5)
    f1 = f1_at_k(precision, recall)
    ap = average_precision(retrieved_ids, relevant_set)
    ndcg = ndcg_at_k(retrieved_ids, relevant_set, k=5)

    print('Evaluation results (top-5):')
    print(f'Precision@5: {precision:.4f}')
    print(f'Recall@5:    {recall:.4f}')
    print(f'F1@5:        {f1:.4f}')
    print(f'MAP:         {ap:.4f}')
    print(f'nDCG@5:      {ndcg:.4f}')
