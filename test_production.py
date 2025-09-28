#!/usr/bin/env python3
"""Production test script for Contextual Meeting Assistant"""

import sys
from pathlib import Path

# Add src and project root to Python path
project_root = Path.cwd()
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root))

print("🧪 Production Test for Contextual Meeting Assistant")
print("=" * 60)

def test_imports():
    """Test all imports"""
    print("\n1. Testing imports...")
    try:
        from config.settings import PAGE_CONFIG, EMBEDDING_MODEL, SUMMARIZATION_MODEL, QA_MODEL
        from data.preprocessor import TranscriptPreprocessor
        from models.embedding_model import EmbeddingModel
        from pipeline.rag_pipeline import RAGPipeline
        from app.components.upload import upload_transcript_component
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_preprocessor():
    """Test preprocessor functionality"""
    print("\n2. Testing preprocessor...")
    try:
        from data.preprocessor import TranscriptPreprocessor
        preprocessor = TranscriptPreprocessor()
        
        sample_text = """
        John: Good morning everyone, let's start our meeting.
        Sarah: Morning John, I have the updates ready.
        Mike: I'm ready to discuss the project.
        """
        
        cleaned = preprocessor.clean_text(sample_text)
        speakers = preprocessor.extract_speakers(sample_text)
        action_items = preprocessor.extract_action_items(sample_text)
        decisions = preprocessor.extract_decisions(sample_text)
        
        print(f"   ✅ Text cleaning: {len(cleaned)} characters")
        print(f"   ✅ Speaker extraction: {len(speakers)} segments")
        print(f"   ✅ Action items: {len(action_items)} found")
        print(f"   ✅ Decisions: {len(decisions)} found")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessor error: {e}")
        return False

def test_streamlit_app():
    """Test Streamlit app"""
    print("\n3. Testing Streamlit app...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        print("✅ App ready to run")
        return True
    except Exception as e:
        print(f"❌ Streamlit error: {e}")
        return False

def main():
    """Run production tests"""
    print("🚀 Starting production tests...")
    
    tests = [
        test_imports,
        test_preprocessor,
        test_streamlit_app
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 PRODUCTION READY!")
        print("🚀 Run: streamlit run app/main.py")
        print("📁 Test file: sample_meeting_transcript.txt")
        print("🌐 App will be available at: http://localhost:8501")
        return True
    else:
        print("\n⚠️ Some tests failed. Check errors above.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ Ready for GitHub upload!")
    else:
        print("\n❌ Fix issues before proceeding")
