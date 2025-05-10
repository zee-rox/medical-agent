import numpy as np

def clean_query(self, text):
    """Clean the query using the medical NLP pipeline."""
    doc = self.nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def compute_query_embedding(self, query_text):
    """Compute query embeddings using multiple models and average them."""
    try:
        emb1 = np.asarray(self.primary_embeddings.embed_query(query_text), dtype="float32")
        emb2 = np.asarray(self.alternative_embeddings.embed_query(query_text), dtype="float32")
        if emb1.shape != emb2.shape:
            print(f"⚠ Inconsistent embedding shapes detected: {emb1.shape} vs {emb2.shape}. Using primary model.")
            return emb1.reshape(1, -1)
        return ((emb1 + emb2) / 2.0).reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to compute embeddings: {e}")