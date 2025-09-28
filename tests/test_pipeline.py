import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline.rag_pipeline import RAGPipeline

class TestRAGPipeline:
    """Test cases for RAGPipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'summarization_model': 'facebook/bart-large-cnn',
            'qa_model': 'deepset/roberta-base-squad2',
            'faiss_index_type': 'IndexFlatIP'
        }
        self.pipeline = RAGPipeline(self.config)
        self.sample_transcript = """
        John: Let's start the project meeting.
        Sarah: We need to discuss the timeline.
        Mike: The deadline is next Friday.
        John: We should assign tasks.
        Sarah: I'll handle the frontend.
        Mike: I'll work on the backend.
        John: Great, let's meet next week.
        """
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline is not None
        assert not self.pipeline.is_indexed
    
    def test_process_transcript(self):
        """Test transcript processing"""
        results = self.pipeline.process_transcript(self.sample_transcript)
        
        assert 'summary' in results
        assert 'key_points' in results
        assert 'action_items' in results
        assert 'decisions' in results
        assert 'num_chunks' in results
        
        assert isinstance(results['summary'], str)
        assert isinstance(results['key_points'], list)
        assert isinstance(results['action_items'], list)
        assert isinstance(results['decisions'], list)
        assert isinstance(results['num_chunks'], int)
        
        assert self.pipeline.is_indexed
    
    def test_answer_question_before_indexing(self):
        """Test answering question before indexing"""
        with pytest.raises(ValueError):
            self.pipeline.answer_question("What was discussed?")
    
    def test_answer_question_after_indexing(self):
        """Test answering question after indexing"""
        # First process transcript
        self.pipeline.process_transcript(self.sample_transcript)
        
        # Then ask question
        response = self.pipeline.answer_question("What was discussed?")
        
        assert 'answer' in response
        assert 'context_chunks' in response
        assert 'scores' in response
        assert 'metadata' in response
        
        assert isinstance(response['answer'], str)
        assert isinstance(response['context_chunks'], list)
        assert isinstance(response['scores'], list)
        assert isinstance(response['metadata'], list)

