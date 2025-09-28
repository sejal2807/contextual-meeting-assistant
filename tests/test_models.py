import pytest
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.embedding_model import EmbeddingModel
from models.summarizer import MeetingSummarizer
from models.qa_model import QAModel

class TestEmbeddingModel:
    """Test cases for EmbeddingModel"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.embedding_model = EmbeddingModel()
    
    def test_encode_single_text(self):
        """Test encoding single text"""
        text = "This is a test sentence."
        embedding = self.embedding_model.encode(text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1
    
    def test_encode_multiple_texts(self):
        """Test encoding multiple texts"""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = self.embedding_model.encode(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
    
    def test_get_embedding_dim(self):
        """Test getting embedding dimension"""
        dim = self.embedding_model.get_embedding_dim()
        assert isinstance(dim, int)
        assert dim > 0

class TestMeetingSummarizer:
    """Test cases for MeetingSummarizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.summarizer = MeetingSummarizer()
    
    def test_summarize(self):
        """Test summarization functionality"""
        text = """
        This is a long text about a meeting. The team discussed various topics.
        They talked about project deadlines, resource allocation, and team coordination.
        Several decisions were made regarding the upcoming sprint.
        Action items were assigned to different team members.
        The meeting concluded with a plan for the next week.
        """
        summary = self.summarizer.summarize(text)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) < len(text)
    
    def test_extract_key_points(self):
        """Test key points extraction"""
        text = """
        First important point about the project.
        Second key discussion about resources.
        Third major decision regarding timeline.
        Fourth critical action item.
        Fifth important conclusion.
        """
        key_points = self.summarizer.extract_key_points(text)
        assert isinstance(key_points, list)
        assert len(key_points) > 0

class TestQAModel:
    """Test cases for QAModel"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.qa_model = QAModel()
    
    def test_answer_question(self):
        """Test question answering"""
        question = "What was discussed in the meeting?"
        context = """
        The team discussed project deadlines and resource allocation.
        They talked about hiring new developers and expanding the team.
        Several action items were assigned to team members.
        """
        answer = self.qa_model.answer_question(question, context)
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_get_answer_confidence(self):
        """Test answer with confidence"""
        question = "What were the main topics?"
        context = """
        The main topics included project planning, resource allocation, and team coordination.
        The team discussed deadlines and deliverables.
        """
        answer, confidence = self.qa_model.get_answer_confidence(question, context)
        assert isinstance(answer, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

