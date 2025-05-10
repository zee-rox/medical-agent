import os
import pickle
from ragatouille import RAGPretrainedModel
from rank_bm25 import BM25Okapi
import pandas as pd
import faiss
import warnings

warnings.filterwarnings("ignore")

faiss_index_path = "./data/faiss_index_D.idx"
if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"❌ FAISS index '{faiss_index_path}' not found.")
faiss_index = faiss.read_index(faiss_index_path)
print(f"✅ FAISS index loaded from '{faiss_index_path}'.")

chunks_path = "./data/chunks.pkl"
if not os.path.exists(chunks_path):
    raise FileNotFoundError(f"❌ Pickle file '{chunks_path}' not found.")
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)
print(f"✅ Loaded {len(chunks)} document chunks.")

corpus = [chunk.page_content for chunk in chunks]
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

colbert_reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
print("✅ ColBERT reranker initialized.")

history_index_path = "./data/faiss_index_merged_df_diagnosis.idx"
if not os.path.exists(history_index_path):
    raise FileNotFoundError(f"❌ Patient history FAISS index '{history_index_path}' not found.")
# history_faiss_index = faiss.read_index(history_index_path)
history_faiss_index = []
print(f"✅ Patient history FAISS index loaded from '{history_index_path}'.")

df_history = pd.read_pickle("./data/merged_df_diagnosis.pkl")
print(f"✅ Loaded patient history data from 'merged_df_diagnosis.pkl' (Total records: {len(df_history)})")