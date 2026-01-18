#!/usr/bin/env python3
"""
Visualization Page
Word frequencies, comparisons between volumes, timeline charts
Uses Plotly for interactive visualizations
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import re
import logging

logger = logging.getLogger(__name__)

# Import Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available")

# Import managers
try:
    from extended_corpus_manager import TajikLibraryManager
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False

try:
    from corpus_manager import TajikCorpusManager
    CORPUS_AVAILABLE = True
except ImportError:
    CORPUS_AVAILABLE = False


# -------------------------------------------------------------------
# Session State
# -------------------------------------------------------------------
def init_viz_state():
    """Initialize visualization session state"""
    defaults = {
        'viz_data_loaded': False,
        'viz_contributions': None,
        'viz_volumes': None,
        'viz_selected_volumes': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -------------------------------------------------------------------
# Data Loading and Processing
# -------------------------------------------------------------------
def load_all_contributions() -> List[Dict]:
    """Load all contributions"""
    contributions = []
    contrib_dir = Path("tajik_corpus/contributions")
    
    if not contrib_dir.exists():
        return contributions
    
    for file in sorted(contrib_dir.glob("*.json")):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                contributions.append(data)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    return contributions


def extract_word_frequencies(text: str, top_n: int = 100) -> List[tuple]:
    """Extract word frequencies from text"""
    # Tajik/Cyrillic word pattern
    words = re.findall(r'[а-яёғӣӯҳқҷА-ЯЁҒӢӮҲҚҶ]+', text.lower())
    
    # Filter stopwords (basic Tajik/Persian stopwords)
    stopwords = {
        'ва', 'дар', 'ба', 'аз', 'ки', 'ин', 'он', 'бо', 'ҳам', 'чун',
        'то', 'ман', 'ту', 'мо', 'шумо', 'у', 'вай', 'худ', 'ҳар', 'як',
        'ё', 'на', 'не', 'чи', 'кай', 'куҷо', 'чаро', 'агар', 'пас',
        'ҳаст', 'буд', 'шуд', 'кард', 'гуфт', 'шавад', 'бошад', 'аст'
    }
    
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    counter = Counter(filtered_words)
    return counter.most_common(top_n)


def group_contributions_by_author(contributions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group contributions by author"""
    grouped = {}
    
    for c in contributions:
        author = c.get('metadata', {}).get('author', 'Unknown')
        if author not in grouped:
            grouped[author] = []
        grouped[author].append(c)
    
    return grouped


def group_contributions_by_collection(contributions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group contributions by collection/volume"""
    grouped = {}
    
    for c in contributions:
        collection = c.get('metadata', {}).get('collection', 
                     c.get('metadata', {}).get('volume_title', 'Unknown'))
        if collection not in grouped:
            grouped[collection] = []
        grouped[collection].append(c)
    
    return grouped


def get_combined_text(contributions: List[Dict]) -> str:
    """Combine text from multiple contributions"""
    texts = []
    for c in contributions:
        text = c.get('raw_text', c.get('normalized_text', ''))
        texts.append(text)
    return '\n'.join(texts)


# -------------------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------------------
def plot_word_frequency_bar(word_freqs: List[tuple], title: str = "Word Frequencies"):
    """Create bar chart of word frequencies"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available for visualization")
        return
    
    if not word_freqs:
        st.info("No data available")
        return
    
    words, counts = zip(*word_freqs[:30])
    
    fig = go.Figure(data=[
        go.Bar(x=list(words), y=list(counts), marker_color='steelblue')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Word",
        yaxis_title="Frequency",
        xaxis_tickangle=-45,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_word_frequency_comparison(data_sets: Dict[str, List[tuple]], top_n: int = 20):
    """Compare word frequencies across multiple collections"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available")
        return
    
    if not data_sets:
        st.info("No data to compare")
        return
    
    # Get all unique words from top N of each set
    all_words = set()
    for freqs in data_sets.values():
        for word, _ in freqs[:top_n]:
            all_words.add(word)
    
    # Create comparison data
    fig = go.Figure()
    
    for name, freqs in data_sets.items():
        freq_dict = dict(freqs)
        words_sorted = sorted(all_words)
        values = [freq_dict.get(w, 0) for w in words_sorted]
        
        fig.add_trace(go.Bar(
            name=name,
            x=words_sorted,
            y=values
        ))
    
    fig.update_layout(
        title="Word Frequency Comparison",
        xaxis_title="Word",
        yaxis_title="Frequency",
        xaxis_tickangle=-45,
        barmode='group',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_theme_distribution(contributions: List[Dict]):
    """Plot theme distribution from analysis data"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available")
        return
    
    # Aggregate themes
    theme_counts = Counter()
    
    for c in contributions:
        tags = c.get('tags', [])
        for tag in tags:
            if tag.startswith('theme:'):
                theme = tag.replace('theme:', '')
                theme_counts[theme] += 1
    
    if not theme_counts:
        st.info("No theme data available")
        return
    
    themes, counts = zip(*theme_counts.most_common(10))
    
    fig = go.Figure(data=[
        go.Pie(labels=list(themes), values=list(counts), hole=0.3)
    ])
    
    fig.update_layout(
        title="Theme Distribution",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_meter_distribution(contributions: List[Dict]):
    """Plot meter distribution"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available")
        return
    
    meter_counts = Counter()
    
    for c in contributions:
        tags = c.get('tags', [])
        for tag in tags:
            if tag.startswith('meter:'):
                meter = tag.replace('meter:', '')
                meter_counts[meter] += 1
    
    if not meter_counts:
        st.info("No meter data available")
        return
    
    meters, counts = zip(*meter_counts.most_common())
    
    fig = go.Figure(data=[
        go.Bar(x=list(meters), y=list(counts), marker_color='darkgreen')
    ])
    
    fig.update_layout(
        title="Meter Distribution",
        xaxis_title="Meter",
        yaxis_title="Count",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_timeline(contributions: List[Dict]):
    """Plot poems over time"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available")
        return
    
    # Group by year
    year_counts = Counter()
    
    for c in contributions:
        year = c.get('metadata', {}).get('publication_year')
        if year:
            year_counts[year] += 1
    
    if not year_counts:
        st.info("No year data available")
        return
    
    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]
    
    fig = go.Figure(data=[
        go.Scatter(x=years, y=counts, mode='lines+markers', line=dict(color='purple'))
    ])
    
    fig.update_layout(
        title="Poems by Publication Year",
        xaxis_title="Year",
        yaxis_title="Number of Poems",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_lexical_diversity_comparison(data_sets: Dict[str, List[Dict]]):
    """Compare lexical diversity across collections"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available")
        return
    
    diversities = {}
    
    for name, contributions in data_sets.items():
        text = get_combined_text(contributions)
        words = re.findall(r'[а-яёғӣӯҳқҷА-ЯЁҒӢӮҲҚҶ]+', text.lower())
        
        if words:
            unique = len(set(words))
            total = len(words)
            diversity = unique / total if total > 0 else 0
            diversities[name] = diversity
    
    if not diversities:
        st.info("No data available")
        return
    
    names = list(diversities.keys())
    values = [diversities[n] * 100 for n in names]  # Convert to percentage
    
    fig = go.Figure(data=[
        go.Bar(x=names, y=values, marker_color='coral')
    ])
    
    fig.update_layout(
        title="Lexical Diversity Comparison",
        xaxis_title="Collection",
        yaxis_title="Diversity (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# Main Page
# -------------------------------------------------------------------
def main():
    init_viz_state()
    
    st.title("Visualizations")
    st.markdown("Interactive charts and analysis")
    
    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for visualizations. Install with: pip install plotly")
        st.stop()
    
    st.markdown("---")
    
    # Load data
    if st.session_state.viz_contributions is None:
        with st.spinner("Loading data..."):
            st.session_state.viz_contributions = load_all_contributions()
    
    contributions = st.session_state.viz_contributions
    
    if not contributions:
        st.warning("No data available. Use the Analyze page to add poems first.")
        st.stop()
    
    st.success(f"Loaded {len(contributions)} poems")
    
    # Refresh button
    if st.button("Reload Data", key="btn_reload_viz"):
        st.session_state.viz_contributions = None
        st.rerun()
    
    st.markdown("---")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Word Frequencies", 
        "Compare Collections", 
        "Distributions",
        "Timeline"
    ])
    
    with tab1:
        st.header("Word Frequency Analysis")
        
        # Group options
        grouped_by_collection = group_contributions_by_collection(contributions)
        collections = list(grouped_by_collection.keys())
        
        if len(collections) > 1:
            selected = st.selectbox(
                "Select collection:",
                ["All"] + collections,
                key="viz_collection_select"
            )
            
            if selected == "All":
                text = get_combined_text(contributions)
            else:
                text = get_combined_text(grouped_by_collection[selected])
        else:
            text = get_combined_text(contributions)
        
        top_n = st.slider("Number of words", 10, 100, 30, key="viz_top_n")
        
        word_freqs = extract_word_frequencies(text, top_n)
        
        plot_word_frequency_bar(word_freqs, f"Top {top_n} Words")
        
        # Show as table
        with st.expander("Show as table"):
            for i, (word, count) in enumerate(word_freqs, 1):
                st.write(f"{i}. {word}: {count}")
    
    with tab2:
        st.header("Compare Collections")
        
        grouped = group_contributions_by_collection(contributions)
        
        if len(grouped) < 2:
            st.info("Need at least 2 collections with different metadata to compare. Add metadata in the Library page.")
        else:
            available = list(grouped.keys())
            
            selected_collections = st.multiselect(
                "Select collections to compare:",
                available,
                default=available[:2] if len(available) >= 2 else available,
                key="viz_compare_select"
            )
            
            if len(selected_collections) >= 2:
                # Word frequency comparison
                st.subheader("Word Frequency Comparison")
                
                data_sets = {}
                for name in selected_collections:
                    text = get_combined_text(grouped[name])
                    data_sets[name] = extract_word_frequencies(text, 50)
                
                plot_word_frequency_comparison(data_sets)
                
                # Lexical diversity comparison
                st.subheader("Lexical Diversity")
                
                comparison_data = {name: grouped[name] for name in selected_collections}
                plot_lexical_diversity_comparison(comparison_data)
            else:
                st.info("Select at least 2 collections to compare")
    
    with tab3:
        st.header("Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Theme Distribution")
            plot_theme_distribution(contributions)
        
        with col2:
            st.subheader("Meter Distribution")
            plot_meter_distribution(contributions)
    
    with tab4:
        st.header("Timeline")
        
        plot_timeline(contributions)
        
        # Statistics
        grouped_by_year = {}
        for c in contributions:
            year = c.get('metadata', {}).get('publication_year')
            if year:
                if year not in grouped_by_year:
                    grouped_by_year[year] = 0
                grouped_by_year[year] += 1
        
        if grouped_by_year:
            st.subheader("Statistics")
            years = sorted(grouped_by_year.keys())
            st.write(f"**Year range:** {min(years)} - {max(years)}")
            st.write(f"**Total years covered:** {len(years)}")
            
            most_productive = max(grouped_by_year.items(), key=lambda x: x[1])
            st.write(f"**Most productive year:** {most_productive[0]} ({most_productive[1]} poems)")


# Run
main()
