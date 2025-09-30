import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import List, Dict, Tuple
import re
import os

class QAModel:
    """Question Answering model for contextual queries"""
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2", max_seq_len: int = 384):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
        torch_dtype = torch.float16 if self.device == "cuda" else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        )
        self.model.to(self.device)
        self.model.eval()
        self.max_seq_len = max_seq_len
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer a question given a context"""
        # Tokenize inputs
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_seq_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
        
        # Find answer span
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # Extract answer
        if start_idx <= end_idx:
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        else:
            answer = "No answer found"
        
        return answer.strip()
    
    def get_answer_confidence(self, question: str, context: str) -> Tuple[str, float]:
        """Get answer with confidence score"""
        # Tokenize inputs
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_seq_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = torch.softmax(outputs.start_logits, dim=-1)
            end_scores = torch.softmax(outputs.end_logits, dim=-1)
        
        # Find answer span
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # Calculate confidence
        confidence = (start_scores[0][start_idx] * end_scores[0][end_idx]).item()
        
        # Extract answer
        if start_idx <= end_idx:
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        else:
            answer = "No answer found"
            confidence = 0.0
        
        return answer.strip(), confidence

