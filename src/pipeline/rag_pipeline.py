import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

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
        self.summarizer = MeetingSummarizer(config['summarization_model'])
        self.qa_model = QAModel(
            config['qa_model'],
            max_seq_len=config.get('max_seq_len_qa', 384)
        )
        
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
    
    def process_transcript(self, transcript: str) -> Dict:
        """Process meeting transcript and build index"""
        # Preprocess transcript
        cleaned_text = self.preprocessor.clean_text(transcript)
        speaker_segments = self.preprocessor.extract_speakers(cleaned_text)
        action_items = self.preprocessor.extract_action_items(cleaned_text)
        decisions = self.preprocessor.extract_decisions(cleaned_text)
        
        # Generate summary
        summary = self.summarizer.summarize(cleaned_text)
        key_points = self.summarizer.extract_key_points(cleaned_text)
        
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
        
        # Add to FAISS index
        self.faiss_index.add_embeddings(embeddings, metadata)
        self.is_indexed = True
        
        self.last_results = {
            'summary': summary,
            'key_points': key_points,
            'action_items': action_items,
            'decisions': decisions,
            'num_chunks': len(chunks)
        }
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
        
        # Answer with confidence
        answer, confidence = self.qa_model.get_answer_confidence(question, context)

        # Lightweight intent-based fallback when extractive QA is uncertain
        used_fallback = False
        fallback_text = None
        ql = question.lower()
        if (not answer or answer == "No answer found" or confidence < 0.45) and self.last_results:
            if any(t in ql for t in ["decision", "decide"]):
                decs = self.last_results.get('decisions', [])
                if decs:
                    fallback_text = "Decisions: " + "; ".join(decs[:5])
            elif any(t in ql for t in ["action", "task", "todo"]):
                acts = self.last_results.get('action_items', [])
                if acts:
                    fallback_text = "Action items: " + "; ".join(acts[:5])
            elif any(t in ql for t in ["key point", "key points", "main topic", "topics", "key discussion", "outcome", "outcomes"]):
                kps = self.last_results.get('key_points', [])
                if kps:
                    fallback_text = "Key points: " + "; ".join(kps[:5])
            elif any(t in ql for t in ["summary", "overview", "about"]):
                summ = self.last_results.get('summary')
                if summ:
                    fallback_text = summ
            if fallback_text:
                used_fallback = True
                answer = fallback_text
                confidence = max(confidence, 0.6)

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

