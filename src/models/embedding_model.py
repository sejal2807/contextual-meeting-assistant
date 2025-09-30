import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import os

class EmbeddingModel:
    """Handle text embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 16):
        self.model_name = model_name
        # Honor cache/env to avoid repeated downloads on Streamlit Cloud
        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
        self.model = SentenceTransformer(model_name, device=None, cache_folder=cache_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.batch_size = batch_size
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = None) -> np.ndarray:
        """Generate embeddings for input texts"""
        if isinstance(texts, str):
            texts = [texts]
        if batch_size is None:
            batch_size = self.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()

