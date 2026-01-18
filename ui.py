#!/usr/bin/env python3
"""
Tajik Poetry Analyzer - Multi-Page Application
Main entry point with navigation to all features
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Tajik Poetry Analyzer",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main {max-width: 1200px; margin: 0 auto;}
    h1 {text-align: center; color: #2c3e50;}
    .feature-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    .nav-hint {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.title("Tajik Poetry Analyzer")
st.markdown("Scientific analysis of Tajik poetry with classical and modern approaches")

st.markdown("---")

# Navigation hint
st.markdown("""
<div class="nav-hint">
<strong>Navigation:</strong> Use the sidebar on the left to access different modules.
</div>
""", unsafe_allow_html=True)

# Feature overview
st.header("Available Modules")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
    <h3>1 - Analyze</h3>
    <p>Upload and analyze poetry files (PDF/TXT)</p>
    <ul>
        <li>16 classical Aruz meters</li>
        <li>Free verse detection</li>
        <li>Qafiyeh/Radif analysis</li>
        <li>Excel export</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h3>2 - Library</h3>
    <p>Manage your poetry collection</p>
    <ul>
        <li>Add metadata (author, year, publisher)</li>
        <li>Register volumes</li>
        <li>Edit existing entries</li>
        <li>Browse collection</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
    <h3>3 - Visualize</h3>
    <p>Interactive charts and comparisons</p>
    <ul>
        <li>Word frequency analysis</li>
        <li>Compare volumes</li>
        <li>Timeline visualization</li>
        <li>Theme distribution</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h3>4 - Corpus</h3>
    <p>Manage training data</p>
    <ul>
        <li>Export for NLP training</li>
        <li>Corpus statistics</li>
        <li>Git integration</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick stats if library exists
st.header("Current Status")

try:
    from extended_corpus_manager import TajikLibraryManager
    from corpus_manager import TajikCorpusManager
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simple Corpus")
        try:
            corpus_mgr = TajikCorpusManager()
            stats = corpus_mgr.get_corpus_statistics()
            st.metric("Poems", stats.get("total_poems", 0))
            st.metric("Total Words", stats.get("total_words", 0))
        except Exception as e:
            st.info("No corpus data yet. Use Analyze to add poems.")
    
    with col2:
        st.subheader("Library")
        try:
            library_mgr = TajikLibraryManager()
            lib_stats = library_mgr.get_statistics()
            st.metric("Volumes", lib_stats.get("total_volumes", 0))
            st.metric("Authors", lib_stats.get("authors_count", 0))
        except Exception as e:
            st.info("No library data yet. Use Library to register volumes.")

except ImportError as e:
    st.warning(f"Some modules not available: {e}")

st.markdown("---")

# Footer
st.markdown("""
**Tajik Poetry Analyzer** | Version 2.1  
For scientific research on Tajik/Persian poetry  
License: CC-BY-NC-SA-4.0
""")
