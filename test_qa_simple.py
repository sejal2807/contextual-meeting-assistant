#!/usr/bin/env python3
"""
Simple Q&A test to isolate the issue
"""
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root))

def test_simple_qa():
    print("🧪 Testing simple Q&A...")
    
    try:
        # Test config
        from config.settings import EMBEDDING_MODEL, SUMMARIZATION_MODEL, QA_MODEL
        print(f"✅ Config: {EMBEDDING_MODEL}")
        
        # Test spaCy
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy loaded")
        
        # Test simple text processing
        test_text = "John: We need to decide on the project timeline. Sarah: I think we should aim for Q2 delivery."
        doc = nlp(test_text)
        print(f"✅ spaCy processing: {len(doc)} tokens")
        
        # Test model loading (without full initialization)
        from models.embedding_model import EmbeddingModel
        from models.qa_model import QAModel
        
        print("✅ Model classes imported")
        
        # Test simple embedding
        embedding_model = EmbeddingModel(EMBEDDING_MODEL)
        test_embedding = embedding_model.get_embedding("test sentence")
        print(f"✅ Embedding generated: {len(test_embedding)} dimensions")
        
        # Test QA model
        qa_model = QAModel(QA_MODEL)
        test_answer = qa_model.answer_question("What was discussed?", "John: We discussed the project timeline.")
        print(f"✅ QA test: {test_answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Q&A test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_qa()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
