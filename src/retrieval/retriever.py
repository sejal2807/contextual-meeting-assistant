import numpy as np
from typing import List, Dict, Tuple
from retrieval.faiss_index import FAISSIndex
from models.embedding_model import EmbeddingModel

class Retriever:
    """Semantic retriever using FAISS and sentence transformers"""
    
    def __init__(self, embedding_model: EmbeddingModel, faiss_index: FAISSIndex):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
    
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Retrieve relevant documents for a query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        scores, metadata = self.faiss_index.search(query_embedding[0], k=k)
        
        # Extract text chunks
        chunks = [meta['text'] for meta in metadata]
        
        return chunks, scores.tolist(), metadata
    
    def retrieve_with_threshold(self, query: str, k: int = 5, threshold: float = 0.5) -> Tuple[List[str], List[float], List[Dict]]:
        """Retrieve documents above a similarity threshold"""
        chunks, scores, metadata = self.retrieve(query, k)
        
        # Filter by threshold
        filtered_chunks = []
        filtered_scores = []
        filtered_metadata = []
        
        for chunk, score, meta in zip(chunks, scores, metadata):
            if score >= threshold:
                filtered_chunks.append(chunk)
                filtered_scores.append(score)
                filtered_metadata.append(meta)
        
        return filtered_chunks, filtered_scores, filtered_metadata
    
    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics"""
        return self.faiss_index.get_stats()

