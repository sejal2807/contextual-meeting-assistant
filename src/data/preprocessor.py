import re
import nltk
import spacy
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path

class TranscriptPreprocessor:
    """Preprocess meeting transcripts for RAG pipeline"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def extract_speakers(self, text: str) -> List[Tuple[str, str]]:
        """Extract speaker segments from transcript"""
        # Pattern for speaker identification
        speaker_pattern = r'^([A-Z][a-z]+):\s*(.+)$'
        segments = []
        
        for line in text.split('\n'):
            match = re.match(speaker_pattern, line.strip())
            if match:
                speaker, content = match.groups()
                segments.append((speaker, content.strip()))
        
        return segments
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def extract_action_items(self, text: str) -> List[str]:
        """Extract potential action items using NLP patterns"""
        doc = self.nlp(text)
        action_items = []
        
        # Look for action verbs and commitments
        action_patterns = [
            r'(?:will|should|need to|must|have to)\s+[^.]*',
            r'(?:action|task|todo|follow up|next steps?)[^.]*',
            r'(?:assign|delegate|responsible for)[^.]*'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            action_items.extend(matches)
        
        return [item.strip() for item in action_items if len(item.strip()) > 10]
    
    def extract_decisions(self, text: str) -> List[str]:
        """Extract decisions made during the meeting"""
        doc = self.nlp(text)
        decisions = []
        
        decision_patterns = [
            r'(?:decided|agreed|concluded|resolved|determined)[^.]*',
            r'(?:consensus|unanimous|majority)[^.]*',
            r'(?:final decision|outcome|resolution)[^.]*'
        ]
        
        for pattern in decision_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            decisions.extend(matches)
        
        return [decision.strip() for decision in decisions if len(decision.strip()) > 10]

