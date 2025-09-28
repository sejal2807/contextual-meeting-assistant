from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score

class EvaluationMetrics:
    """Evaluate RAG pipeline performance"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_summarization(self, predictions: List[str], references: List[str]) -> Dict:
        """Evaluate summarization quality using ROUGE metrics"""
        rouge_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })
        
        # Calculate average scores
        avg_scores = {
            'rouge1': np.mean([s['rouge1'] for s in rouge_scores]),
            'rouge2': np.mean([s['rouge2'] for s in rouge_scores]),
            'rougeL': np.mean([s['rougeL'] for s in rouge_scores])
        }
        
        return avg_scores
    
    def evaluate_retrieval(self, retrieved_docs: List[List[str]], 
                          relevant_docs: List[List[str]], k_values: List[int] = [1, 5, 10]) -> Dict:
        """Evaluate retrieval performance"""
        results = {}
        
        for k in k_values:
            recalls = []
            precisions = []
            
            for retrieved, relevant in zip(retrieved_docs, relevant_docs):
                retrieved_k = retrieved[:k]
                relevant_set = set(relevant)
                retrieved_set = set(retrieved_k)
                
                if len(relevant_set) > 0:
                    recall = len(relevant_set.intersection(retrieved_set)) / len(relevant_set)
                    recalls.append(recall)
                
                if len(retrieved_set) > 0:
                    precision = len(relevant_set.intersection(retrieved_set)) / len(retrieved_set)
                    precisions.append(precision)
            
            results[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
            results[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
        
        return results
    
    def evaluate_qa(self, predictions: List[str], references: List[str]) -> Dict:
        """Evaluate QA performance using BERTScore"""
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }

