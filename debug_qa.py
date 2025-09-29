#!/usr/bin/env python3
"""
Debug script to test Q&A functionality
"""
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root))

def test_imports():
    print("Testing imports...")
    try:
        from config.settings import EMBEDDING_MODEL, SUMMARIZATION_MODEL, QA_MODEL
        print(f"✅ Config loaded: {EMBEDDING_MODEL}, {SUMMARIZATION_MODEL}, {QA_MODEL}")
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_spacy():
    print("Testing spaCy...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded")
        return True
    except Exception as e:
        print(f"❌ spaCy failed: {e}")
        return False

def test_models():
    print("Testing model imports...")
    try:
        from models.embedding_model import EmbeddingModel
        from models.summarizer import MeetingSummarizer
        from models.qa_model import QAModel
        print("✅ Model classes imported")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_pipeline():
    print("Testing pipeline...")
    try:
        from pipeline.rag_pipeline import RAGPipeline
        print("✅ Pipeline class imported")
        return True
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

def main():
    print("🔍 Q&A Debug Diagnostic")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("spaCy", test_spacy),
        ("Models", test_models),
        ("Pipeline", test_pipeline)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n{name}:")
        results[name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All tests passed! The issue might be in the Streamlit app logic.")
    else:
        print("\n⚠️ Some tests failed. Fix these issues first.")

if __name__ == "__main__":
    main()
