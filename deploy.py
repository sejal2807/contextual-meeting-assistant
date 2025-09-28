#!/usr/bin/env python3
"""
Deployment script for Contextual Meeting Assistant
Handles spaCy model installation and setup
"""

import subprocess
import sys
import os
import streamlit as st

def install_spacy_model():
    """Install spaCy English model"""
    try:
        print("📥 Installing spaCy English model...")
        result = subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ spaCy model installed successfully!")
            return True
        else:
            print(f"❌ Error installing spaCy model: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify_setup():
    """Verify all components are working"""
    try:
        # Test imports
        from config.settings import PAGE_CONFIG
        from data.preprocessor import TranscriptPreprocessor
        from models.embedding_model import EmbeddingModel
        from pipeline.rag_pipeline import RAGPipeline
        
        # Test spaCy
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        print("✅ All components verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 Deploying Contextual Meeting Assistant...")
    
    # Install spaCy model
    if install_spacy_model():
        # Verify setup
        if verify_setup():
            print("\n🎉 Deployment successful!")
            print("🚀 Starting Streamlit app...")
            
            # Start Streamlit
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", "app/main.py",
                "--server.port=8501",
                "--server.address=0.0.0.0"
            ])
        else:
            print("\n⚠️ Setup completed but verification failed")
            print("   The app may still work with limited functionality")
    else:
        print("\n❌ Deployment failed. Please check the errors above.")

if __name__ == "__main__":
    main()
