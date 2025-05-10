import torch
from langchain_huggingface import HuggingFaceEmbeddings
import spacy
import en_core_sci_scibert
from sentence_transformers import CrossEncoder

# Load NLP model
nlp = en_core_sci_scibert.load()

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create embedding models
primary_embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

alternative_embeddings_model = HuggingFaceEmbeddings(
    model_name="Zybg/synthetic-clinical-embedding-model",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# Initialize cross encoder
cross_encoder = CrossEncoder("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")