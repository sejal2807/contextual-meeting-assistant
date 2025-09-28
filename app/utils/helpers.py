import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List

def display_metrics(metrics: Dict, title: str):
    """Display metrics in a formatted way"""
    st.subheader(title)
    for metric, value in metrics.items():
        st.metric(metric, f"{value:.2f}")

def create_performance_chart(retrieval_metrics: Dict, summary_metrics: Dict):
    """Create performance visualization chart"""
    metrics_df = pd.DataFrame({
        'Metric': list(retrieval_metrics.keys()) + list(summary_metrics.keys()),
        'Score': list(retrieval_metrics.values()) + list(summary_metrics.values()),
        'Category': ['Retrieval'] * len(retrieval_metrics) + ['Summarization'] * len(summary_metrics)
    })
    
    fig = px.bar(metrics_df, x='Metric', y='Score', color='Category',
                 title="System Performance Metrics")
    return fig

def format_transcript_for_display(transcript: str, max_length: int = 500) -> str:
    """Format transcript for display with length limit"""
    if len(transcript) > max_length:
        return transcript[:max_length] + "..."
    return transcript

