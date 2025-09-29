import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small and fast
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"  # lighter than bart-large
QA_MODEL = "deepset/tinyroberta-squad2"  # smaller QA model

# FAISS configuration
FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner product for cosine similarity
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Processing parameters
CHUNK_SIZE = 350
CHUNK_OVERLAP = 30
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 30

# Evaluation metrics
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
RETRIEVAL_METRICS = ["recall@1", "recall@5", "recall@10", "f1"]

# Streamlit configuration
PAGE_CONFIG = {
    "page_title": "Contextual Meeting Assistant",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

