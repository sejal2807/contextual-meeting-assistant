import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict
import re
import os

class MeetingSummarizer:
    """Generate meeting summaries using transformer models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
        torch_dtype = torch.float16 if self.device == "cuda" else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        )
        self.model.to(self.device)
        self.model.eval()
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """Generate summary for input text"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=768,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key discussion points"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Generate summary for each sentence group
        key_points = []
        for i in range(0, len(sentences), 5):
            chunk = '. '.join(sentences[i:i+5])
            if len(chunk) > 50:
                summary = self.summarize(chunk, max_length=50, min_length=10)
                key_points.append(summary)
        
        return key_points

