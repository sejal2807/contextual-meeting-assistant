import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

from ..models.embedding_model import EmbeddingModel
from ..models.summarizer import MeetingSummarizer
from ..models.qa_model import QAModel
from ..retrieval.faiss_index import FAISSIndex
from ..data.preprocessor import TranscriptPreprocessor

class RAGPipeline:
    """End-to-end RAG pipeline for meeting assistant"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize components
        self.preprocessor = TranscriptPreprocessor()
        self.embedding_model = EmbeddingModel(config['embedding_model'])
        self.summarizer = MeetingSummarizer(config['summarization_model'])
        self.qa_model = QAModel(config['qa_model'])
        
        # Initialize FAISS index
        self.faiss_index = FAISSIndex(
            embedding_dim=self.embedding_model.get_embedding_dim(),
            index_type=config['faiss_index_type']
        )
        
        self.is_indexed = False
    
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
        
        return {
            'summary': summary,
            'key_points': key_points,
            'action_items': action_items,
            'decisions': decisions,
            'num_chunks': len(chunks)
        }
    
    def answer_question(self, question: str, k: int = 5) -> Dict:
        """Answer question using RAG"""
        if not self.is_indexed:
            raise ValueError("No transcript indexed. Please process a transcript first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([question])
        
        # Retrieve relevant chunks
        scores, metadata = self.faiss_index.search(query_embedding[0], k=k)
        
        # Prepare context for QA model
        context_chunks = [meta['text'] for meta in metadata]
        context = " ".join(context_chunks)
        
        # Generate answer
        answer = self.qa_model.answer_question(question, context)
        
        return {
            'answer': answer,
            'context_chunks': context_chunks,
            'scores': scores.tolist(),
            'metadata': metadata
        }
    
    def save_index(self, filepath: Path):
        """Save FAISS index"""
        self.faiss_index.save(filepath)
    
    def load_index(self, filepath: Path):
        """Load FAISS index"""
        self.faiss_index.load(filepath)
        self.is_indexed = True

