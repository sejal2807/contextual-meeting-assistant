import streamlit as st
from typing import Dict, List

def qa_interface_component(pipeline, k: int = 5, show_context: bool = False):
    """Component for question and answer interface"""
    # Question input
    question = st.text_input(
        "Ask a question about the meeting:",
        placeholder="e.g., What were the main decisions made? Who was responsible for the action items?"
    )
    
    if st.button("🔍 Get Answer", type="primary") and question:
        with st.spinner("Generating answer..."):
            try:
                # Get answer using RAG
                response = pipeline.answer_question(question, k=k)
                
                # Display answer
                st.subheader("💡 Answer")
                st.write(response['answer'])
                
                # Display context if requested
                if show_context:
                    st.subheader("📚 Retrieved Context")
                    for i, chunk in enumerate(response['context_chunks'], 1):
                        with st.expander(f"Context {i} (Score: {response['scores'][i-1]:.3f})"):
                            st.write(chunk)
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

def advanced_options_component():
    """Component for advanced QA options"""
    with st.expander("⚙️ Advanced Options"):
        k = st.slider("Number of relevant chunks to retrieve", 1, 10, 5)
        show_context = st.checkbox("Show retrieved context", value=False)
        return k, show_context

