import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def expand_query_tfidf(query: str, vectorizer: TfidfVectorizer, tfidf_matrix, top_k_terms: int = 3) -> str:
    feature_names = np.array(vectorizer.get_feature_names_out())
    q_vec = vectorizer.transform([query])
    q_dense = q_vec.toarray().ravel()
    nonzero_idx = np.where(q_dense > 0)[0]
    if len(nonzero_idx) == 0:
        return query
    # build term vectors in doc-space
    try:
        term_vecs = tfidf_matrix.T.toarray()
    except MemoryError:
        return query
    centroid = term_vecs[nonzero_idx].mean(axis=0)
    from numpy.linalg import norm
    term_norms = np.linalg.norm(term_vecs, axis=1) + 1e-12
    centroid_norm = norm(centroid) + 1e-12
    sims = (term_vecs @ centroid) / (term_norms * centroid_norm)
    top_idx = sims.argsort()[::-1]
    new_terms = []
    query_terms = set(feature_names[nonzero_idx])
    for idx in top_idx:
        term = feature_names[idx]
        if term in query_terms:
            continue
        if term.isalnum():
            new_terms.append(term)
        if len(new_terms) >= top_k_terms:
            break
    expanded = query + ' ' + ' '.join(new_terms)
    return expanded
