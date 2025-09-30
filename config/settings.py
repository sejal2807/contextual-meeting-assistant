import os
from pathlib import Path
import os

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

"""Centralized configuration for models and runtime.

These defaults are tuned for Streamlit Cloud constraints (CPU-only, limited RAM).
You can override via environment variables without changing code.
"""

# Model configurations (small/fast by default for cloud)
EMBEDDING_MODEL = os.getenv("CMA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUMMARIZATION_MODEL = os.getenv("CMA_SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
QA_MODEL = os.getenv("CMA_QA_MODEL", "deepset/tinyroberta-squad2")

# FAISS configuration
FAISS_INDEX_TYPE = os.getenv("CMA_FAISS_INDEX_TYPE", "IndexFlatIP")  # Inner product for cosine similarity
EMBEDDING_DIM = int(os.getenv("CMA_EMBEDDING_DIM", "384"))  # all-MiniLM-L6-v2

# Processing parameters
CHUNK_SIZE = int(os.getenv("CMA_CHUNK_SIZE", "350"))
CHUNK_OVERLAP = int(os.getenv("CMA_CHUNK_OVERLAP", "30"))
MAX_SUMMARY_LENGTH = int(os.getenv("CMA_MAX_SUMMARY_LENGTH", "150"))
MIN_SUMMARY_LENGTH = int(os.getenv("CMA_MIN_SUMMARY_LENGTH", "30"))

# Runtime/memory tuning
HF_CACHE_DIR = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache")))
TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM", "false")
TORCH_NUM_THREADS = int(os.getenv("CMA_TORCH_NUM_THREADS", "1"))
EMBED_BATCH_SIZE = int(os.getenv("CMA_EMBED_BATCH_SIZE", "16"))
MAX_SEQ_LEN_QA = int(os.getenv("CMA_MAX_SEQ_LEN_QA", "384"))

# Reranking / accuracy controls
ENABLE_CROSS_ENCODER_RERANK = os.getenv("CMA_ENABLE_CROSS_ENCODER_RERANK", "false").lower() == "true"
CROSS_ENCODER_MODEL = os.getenv("CMA_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_MMR = os.getenv("CMA_USE_MMR", "true").lower() == "true"
MMR_LAMBDA = float(os.getenv("CMA_MMR_LAMBDA", "0.5"))
TOP_K_DEFAULT = int(os.getenv("CMA_TOP_K_DEFAULT", "5"))
CONFIDENCE_THRESHOLD_DEFAULT = float(os.getenv("CMA_CONFIDENCE_THRESHOLD", "0.3"))

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

# Apply process-wide defaults for stability on Streamlit Cloud
def apply_runtime_safety():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", TOKENIZERS_PARALLELISM)
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
    try:
        import torch
        torch.set_num_threads(TORCH_NUM_THREADS)
    except Exception:
        pass

