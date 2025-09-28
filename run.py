#!/usr/bin/env python3
"""
Entry point for the Contextual Meeting Assistant application.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Run Streamlit app
    sys.argv = [
        "streamlit",
        "run",
        "app/main.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    
    sys.exit(stcli.main())
