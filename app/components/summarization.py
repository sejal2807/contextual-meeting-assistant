import streamlit as st
from typing import Dict, List

def display_summary_component(results: Dict):
    """Component for displaying meeting summary and insights"""
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

def display_processing_stats(results: Dict):
    """Display processing statistics"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Chunks Created", results['num_chunks'])
    with col2:
        st.metric("Action Items", len(results['action_items']))
    with col3:
        st.metric("Decisions", len(results['decisions']))
    with col4:
        st.metric("Key Points", len(results['key_points']))

