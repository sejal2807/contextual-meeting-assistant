import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocessor import TranscriptPreprocessor

class TestTranscriptPreprocessor:
    """Test cases for TranscriptPreprocessor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = TranscriptPreprocessor()
        self.sample_transcript = """
        John: Hello everyone, let's start the meeting.
        Sarah: I agree, we need to discuss the project timeline.
        Mike: The deadline is next Friday.
        John: We should assign tasks to team members.
        Sarah: I will handle the frontend development.
        Mike: I'll take care of the backend.
        John: Great, let's meet again next week.
        """
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        dirty_text = "  Hello   world!  \n\n  This   is   a   test.  "
        cleaned = self.preprocessor.clean_text(dirty_text)
        assert cleaned == "Hello world! This is a test."
    
    def test_extract_speakers(self):
        """Test speaker extraction"""
        speakers = self.preprocessor.extract_speakers(self.sample_transcript)
        assert len(speakers) > 0
        assert all(isinstance(speaker, tuple) and len(speaker) == 2 for speaker in speakers)
    
    def test_chunk_text(self):
        """Test text chunking"""
        text = "This is a test sentence. " * 100  # Create long text
        chunks = self.preprocessor.chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 50 for chunk in chunks)
    
    def test_extract_action_items(self):
        """Test action item extraction"""
        text_with_actions = """
        We need to complete the report by Friday.
        John will handle the data analysis.
        Sarah should prepare the presentation.
        The team must review the code.
        """
        actions = self.preprocessor.extract_action_items(text_with_actions)
        assert len(actions) > 0
        assert all(len(action) > 10 for action in actions)
    
    def test_extract_decisions(self):
        """Test decision extraction"""
        text_with_decisions = """
        We decided to use Python for the project.
        The team agreed on the new architecture.
        We concluded that the deadline is feasible.
        The consensus was to hire more developers.
        """
        decisions = self.preprocessor.extract_decisions(text_with_decisions)
        assert len(decisions) > 0
        assert all(len(decision) > 10 for decision in decisions)

