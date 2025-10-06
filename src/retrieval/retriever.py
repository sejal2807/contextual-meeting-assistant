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
        # BM25 components (lazy)
        self._bm25 = None
        self._bm25_corpus_tokens: List[List[str]] = []
        self._bm25_texts: List[str] = []
        self._hybrid_enabled: bool = False
    
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Retrieve relevant documents for a query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        scores, metadata = self.faiss_index.search(query_embedding[0], k=k)
        
        # Extract text chunks
        chunks = [meta['text'] for meta in metadata]
        dense_scores = scores.tolist()

        # If hybrid enabled and BM25 available, fuse scores
        if self._hybrid_enabled and self._bm25 is not None and self._bm25_texts:
            try:
                from rank_bm25 import BM25Okapi  # ensure package exists at runtime
                # Get BM25 scores for the whole corpus, then select top-k indices
                bm25_scores_full = self._bm25.get_scores(self._tokenize(query))
                # Map current metadata back to original indices via text match
                # Prefer stable bm25_idx when available to avoid building large maps
                text_to_index = None
                bm25_for_return = []
                for meta in metadata:
                    idx = meta.get('bm25_idx', -1)
                    if idx == -1 and text_to_index is None:
                        text_to_index = {t: i for i, t in enumerate(self._bm25_texts)}
                        idx = text_to_index.get(meta.get('text', ''), -1)
                    bm25_for_return.append(bm25_scores_full[idx] if idx >= 0 else 0.0)
                # Reciprocal Rank Fusion (RRF) with small constant
                epsilon = 60.0
                fused = []
                # Convert dense scores to ranks within returned set
                dense_rank = {i: r for r, i in enumerate(sorted(range(len(dense_scores)), key=lambda x: dense_scores[x], reverse=True), start=1)}
                bm25_rank = {i: r for r, i in enumerate(sorted(range(len(bm25_for_return)), key=lambda x: bm25_for_return[x], reverse=True), start=1)}
                for i in range(len(chunks)):
                    rrf = 1.0 / (epsilon + dense_rank[i]) + 1.0 / (epsilon + bm25_rank[i])
                    fused.append(rrf)
                ranked = sorted(range(len(chunks)), key=lambda i: fused[i], reverse=True)[:k]
                chunks = [chunks[i] for i in ranked]
                metadata = [metadata[i] for i in ranked]
                dense_scores = [dense_scores[i] for i in ranked]
            except Exception:
                # Fall back to dense-only if anything goes wrong
                pass
        
        return chunks, dense_scores, metadata

    def mmr(self, query: str, candidates: List[str], candidate_scores: List[float], lambda_mult: float = 0.5, top_k: int = 5) -> List[int]:
        """Maximal Marginal Relevance (MMR) returning indices of chosen items.

        Greedy selection trades off query relevance and diversity among selected items.
        """
        if not candidates:
            return []

        import numpy as np

        # Compute embeddings (for diversity) and normalize once
        candidate_embeddings = self.embedding_model.encode(candidates)
        query_embedding = self.embedding_model.encode([query])[0]
        normed = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        qn = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # If caller didn't pass relevance scores, use cosine to query
        if not candidate_scores:
            candidate_scores = (normed @ qn).tolist()

        top_k = min(top_k, len(candidates))
        selected: List[int] = []
        available = list(range(len(candidates)))

        while len(selected) < top_k and available:
            best_idx = None
            best_score = -inf
            for idx in available:
                relevance = candidate_scores[idx]
                if not selected:
                    score = relevance
                else:
                    # Max similarity to any selected item (diversity term)
                    max_sim = max((float(normed[idx] @ normed[j]) for j in selected))
                    score = lambda_mult * relevance - (1.0 - lambda_mult) * max_sim
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

    # ----------------------- Hybrid helpers -----------------------
    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    def build_bm25(self, texts: List[str]):
        """Build BM25 over provided texts. Call when (re)indexing transcripts."""
        try:
            from rank_bm25 import BM25Okapi
        except Exception:
            self._bm25 = None
            self._bm25_corpus_tokens = []
            self._bm25_texts = []
            return
        self._bm25_texts = list(texts)
        self._bm25_corpus_tokens = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._bm25_corpus_tokens)

    def set_hybrid_enabled(self, enabled: bool):
        self._hybrid_enabled = bool(enabled)

