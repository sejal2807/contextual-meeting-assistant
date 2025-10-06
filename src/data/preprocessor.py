import re
import nltk
try:
    import spacy
except Exception:
    spacy = None
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path

class TranscriptPreprocessor:
    """Preprocess meeting transcripts for RAG pipeline"""
    
    def __init__(self, use_spacy: bool = True):
        # Optional spaCy load to reduce memory in constrained environments
        self.nlp = None
        if use_spacy and spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None
        # Avoid large nltk downloads in cloud runtime
        
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
        """Split text into overlapping chunks with sentence preservation and token-aware budgeting.

        Uses spaCy sentence boundaries and approximates token counts via spaCy tokens to keep
        semantic units intact while respecting chunk size and overlap budgets.
        """
        if not text or not text.strip():
            return []

        if self.nlp is not None:
            doc = self.nlp(text)
            sentences = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
        else:
            sentences = [s.strip() for s in re.split(r'[\.!?]\s+', text) if s and s.strip()]
        chunks: List[str] = []

        current_tokens = 0
        current_sentences: List[str] = []

        def sentence_token_len(s: str) -> int:
            if self.nlp is not None:
                return len(self.nlp.make_doc(s))
            return max(1, len(s.split()))

        for sent in sentences:
            sent_tokens = sentence_token_len(sent)
            if current_tokens + sent_tokens <= chunk_size or not current_sentences:
                current_sentences.append(sent)
                current_tokens += sent_tokens
            else:
                # finalize current chunk
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # prepare next chunk with overlap from the end of current chunk
                # compute overlap by tokens, approximate by words from the tail
                if overlap > 0 and chunk_text:
                    words = chunk_text.split()
                    tail = words[-overlap:]
                    current_sentences = [" ".join(tail)] if tail else []
                    current_tokens = len(tail)
                else:
                    current_sentences = []
                    current_tokens = 0

                # add current sentence to new chunk
                current_sentences.append(sent)
                current_tokens += sent_tokens

        # flush remainder
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks
    
    def extract_action_items(self, text: str) -> List[str]:
        """Extract potential action items using NLP patterns"""
        if self.nlp is None:
            # Lightweight regex-only when spaCy not loaded
            action_items = []
            action_patterns = [
                r'(?:will|should|need to|must|have to)\s+[^.]*',
                r'(?:action|task|todo|follow up|next steps?)[^.]*',
                r'(?:assign|delegate|responsible for)[^.]*'
            ]
            for pattern in action_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                action_items.extend(matches)
            return [item.strip() for item in action_items if len(item.strip()) > 10]
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
        if self.nlp is None:
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

