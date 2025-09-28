import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class EmbeddingModel:
    """Handle text embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for input texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()

