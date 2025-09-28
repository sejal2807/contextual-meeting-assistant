import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline.rag_pipeline import RAGPipeline
from config.settings import PAGE_CONFIG, EMBEDDING_MODEL, SUMMARIZATION_MODEL, QA_MODEL

# Configure page
st.set_page_config(**PAGE_CONFIG)

# Initialize session state
if 'pipeline' not in st.session_state:
    config = {
        'embedding_model': EMBEDDING_MODEL,
        'summarization_model': SUMMARIZATION_MODEL,
        'qa_model': QA_MODEL,
        'faiss_index_type': 'IndexFlatIP'
    }
    st.session_state.pipeline = RAGPipeline(config)

def main():
    st.title("🤖 Contextual Meeting Assistant")
    st.markdown("Upload meeting transcripts and get AI-powered insights, summaries, and Q&A")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.selectbox(
            "Choose a page",
            ["📝 Upload & Process", "📋 Summary & Insights", "❓ Q&A Interface", "📈 Evaluation"]
        )
    
    if page == "📝 Upload & Process":
        upload_and_process_page()
    elif page == "📋 Summary & Insights":
        summary_page()
    elif page == "❓ Q&A Interface":
        qa_page()
    elif page == "📈 Evaluation":
        evaluation_page()

def upload_and_process_page():
    st.header("📝 Upload Meeting Transcript")
    
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
    
    if st.button("🚀 Process Transcript", type="primary"):
        if uploaded_file:
            transcript = str(uploaded_file.read(), "utf-8")
        elif transcript_text:
            transcript = transcript_text
        else:
            st.error("Please upload a file or paste transcript text")
            return
        
        with st.spinner("Processing transcript..."):
            try:
                # Process transcript
                results = st.session_state.pipeline.process_transcript(transcript)
                
                # Store results in session state
                st.session_state.processed_results = results
                st.session_state.transcript_processed = True
                
                st.success("✅ Transcript processed successfully!")
                
                # Display quick stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Chunks Created", results['num_chunks'])
                with col2:
                    st.metric("Action Items", len(results['action_items']))
                with col3:
                    st.metric("Decisions", len(results['decisions']))
                with col4:
                    st.metric("Key Points", len(results['key_points']))
                
            except Exception as e:
                st.error(f"Error processing transcript: {str(e)}")

def summary_page():
    st.header("📋 Meeting Summary & Insights")
    
    if not st.session_state.get('transcript_processed', False):
        st.warning("Please process a transcript first in the Upload & Process page")
        return
    
    results = st.session_state.processed_results
    
    # Summary
    st.subheader("📝 Meeting Summary")
    st.write(results['summary'])
    
    # Key Points
    st.subheader("🎯 Key Discussion Points")
    for i, point in enumerate(results['key_points'], 1):
        st.write(f"{i}. {point}")
    
    # Action Items
    if results['action_items']:
        st.subheader("✅ Action Items")
        for i, item in enumerate(results['action_items'], 1):
            st.write(f"{i}. {item}")
    
    # Decisions
    if results['decisions']:
        st.subheader("🎯 Decisions Made")
        for i, decision in enumerate(results['decisions'], 1):
            st.write(f"{i}. {decision}")

def qa_page():
    st.header("❓ Question & Answer Interface")
    
    if not st.session_state.get('transcript_processed', False):
        st.warning("Please process a transcript first in the Upload & Process page")
        return
    
    # Question input
    question = st.text_input(
        "Ask a question about the meeting:",
        placeholder="e.g., What were the main decisions made? Who was responsible for the action items?"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        k = st.slider("Number of relevant chunks to retrieve", 1, 10, 5)
        show_context = st.checkbox("Show retrieved context", value=False)
    
    if st.button("🔍 Get Answer", type="primary") and question:
        with st.spinner("Generating answer..."):
            try:
                # Get answer using RAG
                response = st.session_state.pipeline.answer_question(question, k=k)
                
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

def evaluation_page():
    st.header("📈 System Evaluation")
    
    st.info("This page shows evaluation metrics for the RAG pipeline")
    
    # Mock evaluation results (replace with actual evaluation)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Retrieval Metrics")
        retrieval_metrics = {
            "Recall@1": 0.72,
            "Recall@5": 0.89,
            "Recall@10": 0.94,
            "F1 Score": 0.85
        }
        
        for metric, value in retrieval_metrics.items():
            st.metric(metric, f"{value:.2f}")
    
    with col2:
        st.subheader("📝 Summarization Metrics")
        summary_metrics = {
            "ROUGE-1": 0.68,
            "ROUGE-2": 0.45,
            "ROUGE-L": 0.72
        }
        
        for metric, value in summary_metrics.items():
            st.metric(metric, f"{value:.2f}")
    
    # Visualization
    st.subheader("📈 Performance Visualization")
    
    # Create sample performance chart
    metrics_df = pd.DataFrame({
        'Metric': list(retrieval_metrics.keys()) + list(summary_metrics.keys()),
        'Score': list(retrieval_metrics.values()) + list(summary_metrics.values()),
        'Category': ['Retrieval'] * len(retrieval_metrics) + ['Summarization'] * len(summary_metrics)
    })
    
    fig = px.bar(metrics_df, x='Metric', y='Score', color='Category',
                 title="System Performance Metrics")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

