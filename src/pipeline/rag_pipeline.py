import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import hashlib

from models.embedding_model import EmbeddingModel
from models.summarizer import MeetingSummarizer
from models.qa_model import QAModel
from retrieval.faiss_index import FAISSIndex
from retrieval.retriever import Retriever
from data.preprocessor import TranscriptPreprocessor

class RAGPipeline:
    """End-to-end RAG pipeline for meeting assistant"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize components
        self.preprocessor = TranscriptPreprocessor()
        self.embedding_model = EmbeddingModel(
            config['embedding_model'],
            batch_size=config.get('embed_batch_size', 16)
        )
        # Lazy-load heavy models on first use
        self.summarizer = None
        self.qa_model = None
        
        # Initialize FAISS index
        self.faiss_index = FAISSIndex(
            embedding_dim=self.embedding_model.get_embedding_dim(),
            index_type=config['faiss_index_type']
        )
        
        # High-level retriever with optional reranker
        cross_encoder_model = config.get('cross_encoder_model') if config.get('enable_cross_encoder_rerank', False) else None
        self.retriever = Retriever(self.embedding_model, self.faiss_index, cross_encoder_model_name=cross_encoder_model)
        
        self.is_indexed = False
        self.last_results: Dict = {}
        self.embeddings_dir = Path(self.config.get('embeddings_dir', Path('data/embeddings')))
        self.processed_dir = Path(self.config.get('processed_dir', Path('data/processed')))

    def _get_summarizer(self) -> MeetingSummarizer:
        if self.summarizer is None:
            self.summarizer = MeetingSummarizer(self.config['summarization_model'])
        return self.summarizer

    def _get_qa_model(self) -> QAModel:
        if self.qa_model is None:
            self.qa_model = QAModel(
                self.config['qa_model'],
                max_seq_len=self.config.get('max_seq_len_qa', 384)
            )
        return self.qa_model
    
    def process_transcript(self, transcript: str) -> Dict:
        """Process meeting transcript and build index"""
        # Hash transcript for caching/persistence
        transcript_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()[:16]
        # Attempt to load existing index and processed results for this transcript
        try:
            index_path = self.embeddings_dir / f"{transcript_hash}"
            self.load_index(index_path)
            self.is_indexed = True
        except Exception:
            self.is_indexed = False

        # Try load cached processed metadata (summary, key points, etc.)
        cached_results = None
        try:
            cache_path = self.processed_dir / f"{transcript_hash}.json"
            if cache_path.exists():
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_results = json.load(f)
        except Exception:
            cached_results = None
        # Preprocess transcript
        cleaned_text = self.preprocessor.clean_text(transcript)
        speaker_segments = self.preprocessor.extract_speakers(cleaned_text)
        action_items = self.preprocessor.extract_action_items(cleaned_text)
        decisions = self.preprocessor.extract_decisions(cleaned_text)
        
        # Generate summary (lazy-load summarizer)
        if cached_results is not None:
            summary = cached_results.get('summary', '')
            key_points = cached_results.get('key_points', [])
        else:
            summarizer = self._get_summarizer()
            summary = summarizer.summarize(cleaned_text)
            key_points = summarizer.extract_key_points(cleaned_text)
        
        # Chunk text for indexing
        chunks = self.preprocessor.chunk_text(cleaned_text)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        # Create metadata
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                'chunk_id': i,
                'text': chunk,
                'speaker_segments': speaker_segments,
                'action_items': action_items,
                'decisions': decisions
            })
        
        # If not already indexed from cache, build the index now
        if not self.is_indexed:
            self.faiss_index.add_embeddings(embeddings, metadata)
            self.is_indexed = True
            # Persist index
            index_path = self.embeddings_dir / f"{transcript_hash}"
            self.save_index(index_path)

        # Build/update BM25 corpus for hybrid retrieval
        try:
            self.retriever.build_bm25([m['text'] for m in metadata])
            # Enable hybrid by default; can be toggled off via UI/config
            self.retriever.set_hybrid_enabled(self.config.get('enable_hybrid_retrieval', True))
        except Exception:
            # If BM25 unavailable, continue with dense-only
            self.retriever.set_hybrid_enabled(False)
        
        self.last_results = {
            'summary': summary,
            'key_points': key_points,
            'action_items': action_items,
            'decisions': decisions,
            'num_chunks': len(chunks),
            'transcript_hash': transcript_hash
        }
        # Persist processed results cache
        try:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            import json
            with open(self.processed_dir / f"{transcript_hash}.json", 'w', encoding='utf-8') as f:
                json.dump(self.last_results, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return self.last_results
    
    def answer_question(self, question: str, k: int = 5) -> Dict:
        """Answer question using RAG with optional reranking and confidence."""
        if not self.is_indexed:
            raise ValueError("No transcript indexed. Please process a transcript first.")
        
        # Initial retrieve
        chunks, scores, metadata = self.retriever.retrieve(question, k=max(k, 10))
        
        # Optional rerank (MMR or CrossEncoder)
        use_mmr = self.config.get('use_mmr', True)
        mmr_lambda = self.config.get('mmr_lambda', 0.5)
        reranked_chunks, reranked_scores = self.retriever.rerank(
            question, chunks, scores, top_k=k, use_mmr=use_mmr, mmr_lambda=mmr_lambda
        )
        
        # Build context from reranked top-k
        context = " ".join(reranked_chunks)
        
        # Answer with confidence (lazy-load QA model)
        qa_model = self._get_qa_model()
        # Try multi-span aggregation first for list-like queries
        ql = question.lower()
        prefer_multi = any(t in ql for t in ["list", "which", "who", "what are", "decisions", "action", "items", "responsible", "participants"])
        if prefer_multi:
            answer, confidence = qa_model.answer_multi_span(question, context)
        else:
            answer, confidence = qa_model.get_answer_confidence(question, context)

        # Intelligent fallback: if no answer or low confidence, surface best snippet with label
        threshold = float(self.config.get('confidence_threshold', 0.5))
        used_fallback = False
        fallback_text = None
        support_snippet = reranked_chunks[0] if reranked_chunks else ""
        support_score = float(reranked_scores[0]) if reranked_scores else 0.0

        if (not answer or answer == "No answer found" or confidence < threshold):
            # Try intent-based fallback from processed metadata first
        ql = question.lower()
            if self.last_results:
                if any(t in ql for t in ["decision", "decide"]):
                    decs = self.last_results.get('decisions', [])
                    if decs:
                        fallback_text = "Decisions: " + "; ".join(decs[:5])
                if not fallback_text and any(t in ql for t in ["action", "task", "todo"]):
                    acts = self.last_results.get('action_items', [])
                    if acts:
                        fallback_text = "Action items: " + "; ".join(acts[:5])
                if not fallback_text and any(t in ql for t in ["key point", "key points", "main topic", "topics", "outcome", "outcomes"]):
                    kps = self.last_results.get('key_points', [])
                    if kps:
                        fallback_text = "Key points: " + "; ".join(kps[:5])
                if not fallback_text and any(t in ql for t in ["summary", "overview", "about"]):
                    summ = self.last_results.get('summary')
                    if summ:
                        fallback_text = summ

            # If no metadata-based fallback, return top snippet as supportive context
            if not fallback_text and support_snippet:
                fallback_text = support_snippet
            if fallback_text:
                used_fallback = True
                answer = fallback_text
                # Promote confidence to a sane minimum to avoid 0%
                confidence = max(confidence, 0.35 if support_snippet else 0.5)

        return {
            'answer': answer,
            'confidence': confidence,
            'context_chunks': reranked_chunks,
            'scores': reranked_scores,
            'metadata': metadata,
            'used_fallback': used_fallback
        }
    
    def save_index(self, filepath: Path):
        """Save FAISS index"""
        self.faiss_index.save(filepath)
    
    def load_index(self, filepath: Path):
        """Load FAISS index"""
        self.faiss_index.load(filepath)
        self.is_indexed = True

