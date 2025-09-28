import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import json

class FAISSIndex:
    """Manage FAISS index for semantic search"""
    
    def __init__(self, embedding_dim: int, index_type: str = "IndexFlatIP"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]):
        """Add embeddings and metadata to index"""
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Normalize embeddings for cosine similarity
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[dict]]:
        """Search for similar embeddings"""
        if self.index.ntotal == 0:
            return np.array([]), []
        
        # Normalize query embedding
        if self.index_type == "IndexFlatIP":
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Filter out invalid indices
        valid_indices = indices[0][indices[0] >= 0]
        valid_scores = scores[0][indices[0] >= 0]
        valid_metadata = [self.metadata[i] for i in valid_indices]
        
        return valid_scores, valid_metadata
    
    def save(self, filepath: Path):
        """Save index and metadata"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(filepath.with_suffix('.faiss')))
        
        # Save metadata
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load(self, filepath: Path):
        """Load index and metadata"""
        if not filepath.with_suffix('.faiss').exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(filepath.with_suffix('.faiss')))
        
        # Load metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            self.metadata = json.load(f)
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type
        }

