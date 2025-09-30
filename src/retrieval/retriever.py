import numpy as np
from typing import List, Dict, Tuple
from retrieval.faiss_index import FAISSIndex
from models.embedding_model import EmbeddingModel
from math import inf
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

class Retriever:
    """Semantic retriever using FAISS and sentence transformers"""
    
    def __init__(self, embedding_model: EmbeddingModel, faiss_index: FAISSIndex, cross_encoder_model_name: str = None):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.cross_encoder = None
        if cross_encoder_model_name and CrossEncoder is not None:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model_name)
            except Exception:
                self.cross_encoder = None
    
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Retrieve relevant documents for a query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        scores, metadata = self.faiss_index.search(query_embedding[0], k=k)
        
        # Extract text chunks
        chunks = [meta['text'] for meta in metadata]
        
        return chunks, scores.tolist(), metadata

    def mmr(self, query: str, candidates: List[str], candidate_scores: List[float], lambda_mult: float = 0.5, top_k: int = 5) -> List[int]:
        """Maximal Marginal Relevance selection returning indices of chosen items."""
        if not candidates:
            return []
        # Precompute embeddings for candidates for diversity term
        candidate_embeddings = self.embedding_model.encode(candidates)
        query_embedding = self.embedding_model.encode([query])[0]

        selected: List[int] = []
        candidate_indices = list(range(len(candidates)))

        while len(selected) < min(top_k, len(candidates)):
            best_idx = None
            best_score = -inf
            for idx in candidate_indices:
                relevance = candidate_scores[idx]
                diversity = 0.0
                if selected:
                    # cosine similarity to already selected; take max
                    sim_to_selected = candidate_embeddings[selected] @ candidate_embeddings[idx]
                    # normalize by norms
                    # Avoid adding heavy np.linalg.norm per loop by pre-normalizing
                # Compute normalized vectors
                # Pre-normalize embeddings
            # Recompute with normalized embeddings to keep code simple and stable
            import numpy as np
            normed = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
            qn = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            # Relevance as cosine to query if not provided
            if candidate_scores is None or len(candidate_scores) == 0:
                candidate_scores = (normed @ qn).tolist()
            selected = []
            available = list(range(len(candidates)))
            while len(selected) < min(top_k, len(candidates)):
                best_idx = None
                best_score = -1e9
                for idx in available:
                    relevance = candidate_scores[idx]
                    if not selected:
                        score = relevance
                    else:
                        max_sim = max((normed[idx] @ normed[j] for j in selected))
                        score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                selected.append(best_idx)
                available.remove(best_idx)
            return selected
    
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

    def rerank(self, query: str, chunks: List[str], scores: List[float], top_k: int = 5, use_mmr: bool = True, mmr_lambda: float = 0.5) -> Tuple[List[str], List[float]]:
        """Optionally rerank with CrossEncoder or MMR for accuracy improvements."""
        if self.cross_encoder is not None:
            try:
                pairs = [[query, c] for c in chunks]
                ce_scores = self.cross_encoder.predict(pairs).tolist()
                ranked = sorted(zip(chunks, ce_scores), key=lambda x: x[1], reverse=True)[:top_k]
                new_chunks, new_scores = zip(*ranked)
                return list(new_chunks), list(new_scores)
            except Exception:
                pass
        if use_mmr:
            idxs = self.mmr(query, chunks, scores, lambda_mult=mmr_lambda, top_k=top_k)
            reranked_chunks = [chunks[i] for i in idxs]
            reranked_scores = [scores[i] for i in idxs]
            return reranked_chunks, reranked_scores
        return chunks[:top_k], scores[:top_k]
    
    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics"""
        return self.faiss_index.get_stats()

