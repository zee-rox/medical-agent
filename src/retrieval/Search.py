from typing import List, Tuple
import numpy as np
from src.retrieval.DataLoader import (
    faiss_index, 
    chunks, 
    bm25, 
    colbert_reranker
)

from src.models.LoadEmbeddingModel import (
    primary_embeddings_model,
    alternative_embeddings_model,
    nlp
)

# from utils.TextProcessing import (
# clean_query,
# compute_query_embedding
# )


def clean_query(text):
    """Clean the query using the medical NLP pipeline."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def compute_query_embedding(query_text):
    """Compute query embeddings using multiple models and average them."""
    try:
        emb1 = np.asarray(primary_embeddings_model.embed_query(query_text), dtype="float32")
        emb2 = np.asarray(alternative_embeddings_model.embed_query(query_text), dtype="float32")
        if emb1.shape != emb2.shape:
            print(f"⚠ Inconsistent embedding shapes detected: {emb1.shape} vs {emb2.shape}. Using primary model.")
            return emb1.reshape(1, -1)
        return ((emb1 + emb2) / 2.0).reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to compute embeddings: {e}")

def search(query: str, embedding=None) -> Tuple[str, List[str]]:
    """
    Retrieves clinical context for a query.
    Uses FAISS and BM25 to extract candidate text chunks,
    and reranks them using ColBERT reranker.
    
    Returns a tuple:
      - context: concatenated string of top re-ranked chunks (from ColBERT),
      - rag_chunks: a list of individual text chunks (from ColBERT).
    """
    cleaned = clean_query(query)
    emb = embedding if embedding is not None else compute_query_embedding(cleaned)
    
    k = 60
    distances, indices = faiss_index.search(emb, k)
    bm25_scores = bm25.get_scores(cleaned.split())
    scores = []
    for idx, faiss_dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(chunks):
            score = -faiss_dist + bm25_scores[idx]
            scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    
    top_indices = [idx for idx, _ in scores[:30]]
    candidate_texts = [chunks[idx].page_content for idx in top_indices]
    
    k_final = 30
    reranked_results = colbert_reranker.rerank(query, candidate_texts, k=k_final)
    colbert_candidates = [candidate_texts[res["result_index"]] for res in reranked_results]
    
    # print("\n📖 Retrieval Results:")
    # for i, text in enumerate(colbert_candidates):
    #     print(f"Rank {i+1}: {text[:300]}...")
    
    retrieved_context = "\n".join(colbert_candidates)
    return retrieved_context, colbert_candidates

def final_answer(answer: str) -> dict:
    """
    Return the final diagnosis report.
    Only returns the "answer" field.
    """
    return {"answer": answer}