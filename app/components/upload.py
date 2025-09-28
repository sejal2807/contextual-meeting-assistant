import streamlit as st
from typing import Optional

def upload_transcript_component() -> Optional[str]:
    """Component for uploading meeting transcripts"""
    st.subheader("📁 Upload Meeting Transcript")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a transcript file",
        type=['txt', 'md'],
        help="Upload a meeting transcript in text format"
    )
    
    # Text input
    st.subheader("Or paste transcript directly:")
    transcript_text = st.text_area(
        "Meeting Transcript",
        height=300,
        placeholder="Paste your meeting transcript here..."
    )
    
    if uploaded_file:
        return str(uploaded_file.read(), "utf-8")
    elif transcript_text:
        return transcript_text
    else:
        return None

