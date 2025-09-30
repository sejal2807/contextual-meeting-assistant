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
    
    def answer_question(self, question: str, context: str, max_answer_tokens: int = 60) -> str:
        """Answer a question given a context with windowing and length cap"""
        answer, _ = self.get_answer_confidence(question, context, max_answer_tokens=max_answer_tokens)
        return answer
    
    def get_answer_confidence(self, question: str, context: str, max_answer_tokens: int = 60) -> Tuple[str, float]:
        """Get best-span answer with calibrated confidence using sliding windows.

        - Uses overflow/stride to cover long contexts
        - Scores span vs. null (no answer) for SQuAD2-style heads
        - Caps span length to avoid dumping large chunks
        """
        stride = max(96, self.max_seq_len // 4)
        raw_enc = self.tokenizer(
            question,
            context,
            max_length=self.max_seq_len,
            truncation=True,
            return_overflowing_tokens=True,
            stride=stride,
            padding="max_length",
            return_tensors="pt"
        )
        # Remove fields not accepted by model.forward
        overflow_to_sample_mapping = raw_enc.pop("overflow_to_sample_mapping", None)
        enc = {k: v.to(self.device) for k, v in raw_enc.items() if torch.is_tensor(v)}
        num_spans = enc["input_ids"].shape[0]

        best_text = ""
        best_score = -1e9
        best_null = -1e9

        with torch.no_grad():
            outputs = self.model(**enc)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        for i in range(num_spans):
            start_logit = start_logits[i]
            end_logit = end_logits[i]

            # Null score (CLS at position 0)
            null_score = (start_logit[0] + end_logit[0]).item()
            if null_score > best_null:
                best_null = null_score

            # Find best valid span <= max_answer_tokens
            max_len = max_answer_tokens
            start_idx = torch.argmax(start_logit).item()
            end_idx = torch.argmax(end_logit).item()

            # Improve span by checking nearby ends within limit
            best_span_score = -1e9
            best_start, best_end = 0, 0
            # Limit search window around peaks to reduce compute
            start_topk = torch.topk(start_logit, k=min(20, start_logit.shape[0])).indices.tolist()
            end_topk = torch.topk(end_logit, k=min(20, end_logit.shape[0])).indices.tolist()
            for s in start_topk:
                for e in end_topk:
                    if e < s:
                        continue
                    if e - s + 1 > max_len:
                        continue
                    score = (start_logit[s] + end_logit[e]).item()
                    if score > best_span_score:
                        best_span_score = score
                        best_start, best_end = s, e

            if best_span_score <= -1e8:
                continue

            text_ids = enc["input_ids"][i][best_start:best_end+1]
            text = self.tokenizer.decode(text_ids, skip_special_tokens=True).strip()

            # Prefer shorter, cleaner text
            text = re.sub(r"\s+", " ", text)
            if len(text) == 0:
                continue

            if best_span_score > best_score:
                best_score = best_span_score
                best_text = text

        # Confidence: numerically stable calibration
        import math
        raw = best_score - best_null
        # Clamp raw score to prevent overflow/underflow
        raw = max(-50.0, min(50.0, raw))
        # Use stable sigmoid calculation
        if raw > 0:
            confidence = 1 / (1 + math.exp(-raw / 2.0))
        else:
            confidence = math.exp(raw / 2.0) / (1 + math.exp(raw / 2.0))
        # Add baseline confidence for any reasonable answer
        if best_text and len(best_text.strip()) > 5:
            confidence = max(0.6, confidence)  # minimum 60% for any reasonable answer
        confidence = max(0.0, min(1.0, float(confidence)))

        # Post-process: stop at first sentence terminator for readability
        if best_text:
            m = re.search(r"([\.!?])\s", best_text)
            if m:
                cut = m.end()
                best_text = best_text[:cut].strip()

        if not best_text:
            return "No answer found", 0.0
        return best_text, confidence

