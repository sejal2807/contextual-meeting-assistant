#!/usr/bin/env python3
"""
Setup script to install spaCy model
Run this before starting the Streamlit app
"""

import subprocess
import sys
import os

def install_spacy_model():
    """Install spaCy English model"""
    try:
        print("📥 Installing spaCy English model...")
        
        # Try to download and install the model
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

def verify_installation():
    """Verify spaCy model is working"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model verification successful!")
        return True
    except Exception as e:
        print(f"❌ spaCy model verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Setting up spaCy model for Contextual Meeting Assistant...")
    
    # Install the model
    if install_spacy_model():
        # Verify installation
        if verify_installation():
            print("\n🎉 Setup complete! You can now run:")
            print("   streamlit run app/main.py")
        else:
            print("\n⚠️ Setup completed but verification failed")
            print("   The app may still work with limited functionality")
    else:
        print("\n❌ Setup failed. Please install manually:")
        print("   python -m spacy download en_core_web_sm")
