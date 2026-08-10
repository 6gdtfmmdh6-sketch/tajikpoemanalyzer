#!/usr/bin/env python3
"""
Visualization Page
Word frequencies, comparisons between volumes, timeline charts
Uses Plotly for interactive visualizations
Extended with R export and comprehensive analysis views
"""

import streamlit as st
import json
import csv
import io
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


# Plotly config: disable broken PNG export, keep other tools
PLOTLY_CONFIG = {
    'modeBarButtonsToRemove': ['toImage'],
    'displaylogo': False
}


def export_chart_to_file(fig, filename: str, key: str):
    """Add download button for chart export as interactive HTML"""
    try:
        html_buffer = io.StringIO()
        fig.write_html(
            html_buffer, 
            include_plotlyjs='cdn',
            full_html=True,
            include_mathjax=False
        )
        st.download_button(
            label="💾 Export",
            data=html_buffer.getvalue(),
            file_name=filename.replace('.png', '.html'),
            mime="text/html",
            key=key
        )
    except Exception as e:
        st.caption(f"⚠️ Export-Fehler: {e}")


def init_viz_state():
    """Initialize visualization session state"""
    defaults = {
        'viz_data_loaded': False,
        'viz_contributions': None,
        'viz_volumes': None,
        'viz_selected_volumes': [],
        'viz_selected_word': None,
        'viz_selected_theme': None,
        'viz_selected_meter': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_all_contributions() -> List[Dict]:
    """Load poems from the unified corpus.

    Previously read tajik_corpus/contributions/*.json, which the single-write
    corpus path no longer produces; the page silently showed nothing.
    """
    try:
        from corpus_core import Corpus
        return Corpus().as_contributions()
    except Exception as e:
        logger.error(f"Could not load corpus: {e}")
        return []

def extract_word_frequencies(text: str, top_n: int = 100) -> List[tuple]:
    """Extract word frequencies from text"""
    words = re.findall(r'[а-яёғӣӯҳқҷА-ЯЁҒӢӮҲҚҶ]+', text.lower())
    
    stopwords = {
        'ва', 'дар', 'ба', 'аз', 'ки', 'ин', 'он', 'бо', 'ҳам', 'чун',
        'то', 'ман', 'ту', 'мо', 'шумо', 'у', 'вай', 'худ', 'ҳар', 'як',
        'ё', 'на', 'не', 'чи', 'кай', 'куҷо', 'чаро', 'агар', 'пас',
        'ҳаст', 'буд', 'шуд', 'кард', 'гуфт', 'шавад', 'бошад', 'аст',
        'ҳамин', 'ҳамон', 'он', 'ин', 'чӣ', 'бар', 'зи', 'ки', 'ра'
    }
    
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    counter = Counter(filtered_words)
    return counter.most_common(top_n)


def find_poems_with_word(contributions: List[Dict], word: str) -> List[Dict]:
    """Find all poems containing a specific word"""
    matching = []
    word_lower = word.lower()
    
    for c in contributions:
        text = c.get('raw_text', '').lower()
        if word_lower in text:
            matching.append(c)
    
    return matching


def find_poems_with_tag(contributions: List[Dict], tag_prefix: str, tag_value: str) -> List[Dict]:
    """Find all poems with a specific tag"""
    matching = []
    full_tag = f"{tag_prefix}:{tag_value}"
    
    for c in contributions:
        tags = c.get('tags', [])
        if full_tag in tags:
            matching.append(c)
    
    return matching


def format_volume_label(batch_data: Dict) -> str:
    """Format volume label from metadata: Author: Title (Year)"""
    poems = batch_data.get('poems', [])
    if not poems:
        return batch_data.get('source_filename', 'Unknown')
    
    # Get metadata from first poem in batch
    meta = poems[0].get('volume_metadata', {})
    author = meta.get('author', '')
    title = meta.get('collection', '')
    year = meta.get('year', '')
    
    if author and title:
        if year:
            return f"{author}: {title} ({year})"
        return f"{author}: {title}"
    elif title:
        if year:
            return f"{title} ({year})"
        return title
    
    # Fallback to filename
    return batch_data.get('source_filename', 'Unknown')


def group_contributions_by_batch(contributions: List[Dict]) -> Dict[str, Dict]:
    """Group contributions by batch with metadata"""
    batches = {}
    
    for c in contributions:
        batch_id = c.get('upload_batch_id', '_ungrouped_')
        source = c.get('source_filename', 'Unknown')
        
        if batch_id not in batches:
            batches[batch_id] = {
                'poems': [],
                'source_filename': source,
                'batch_id': batch_id,
                'volume_metadata': c.get('volume_metadata', {})
            }
        batches[batch_id]['poems'].append(c)
    
    # Add formatted labels
    for batch_id, batch_data in batches.items():
        batch_data['label'] = format_volume_label(batch_data)
    
    return batches


def get_combined_text(contributions: List[Dict]) -> str:
    """Combine text from multiple contributions"""
    texts = []
    for c in contributions:
        text = c.get('raw_text', c.get('normalized_text', ''))
        texts.append(text)
    return '\n'.join(texts)


def extract_analysis_stats(contributions: List[Dict]) -> Dict:
    """Extract comprehensive analysis statistics"""
    stats = {
        'meters': Counter(),
        'forms': Counter(),
        'themes': Counter(),
        'registers': Counter(),
        'free_verse_count': 0,
        'classical_count': 0,
        'with_radif': 0,
        'with_qafiyeh': 0,
        'avg_lines': 0,
        'avg_syllables': 0,
        'total_lines': 0,
        'total_syllables': 0,
        'rhyme_schemes': Counter(),
        'stanza_types': Counter(),
        'enjambement_scores': [],
        'syllable_patterns': [],
        'line_variations': [],
        'prose_poetry_scores': [],
    }
    
    for c in contributions:
        tags = c.get('tags', [])
        analysis = c.get('analysis', {})
        
        for tag in tags:
            if tag.startswith('meter:'):
                meter = tag.replace('meter:', '')
                stats['meters'][meter] += 1
            elif tag.startswith('form:'):
                form = tag.replace('form:', '')
                stats['forms'][form] += 1
                if form == 'free_verse':
                    stats['free_verse_count'] += 1
                else:
                    stats['classical_count'] += 1
            elif tag.startswith('theme:'):
                stats['themes'][tag.replace('theme:', '')] += 1
            elif tag.startswith('register:'):
                stats['registers'][tag.replace('register:', '')] += 1
        
        structural = analysis.get('structural', {})
        if structural:
            lines = structural.get('lines', 0)
            stats['total_lines'] += lines
            
            syllables = structural.get('syllables_per_line', [])
            if syllables:
                valid_syllables = [s for s in syllables if s > 0]
                stats['total_syllables'] += sum(valid_syllables)
                if valid_syllables:
                    stats['syllable_patterns'].append({
                        'poem_id': c.get('poem_id', 'unknown'),
                        'title': c.get('title', 'Untitled'),
                        'pattern': valid_syllables,
                        'avg': sum(valid_syllables) / len(valid_syllables),
                        'variance': max(valid_syllables) - min(valid_syllables) if valid_syllables else 0
                    })
        
        rhyme = analysis.get('rhyme', {})
        if rhyme:
            if rhyme.get('has_radif'):
                stats['with_radif'] += 1
            if rhyme.get('rhyme_scheme'):
                stats['with_qafiyeh'] += 1
                stats['rhyme_schemes'][rhyme.get('rhyme_scheme', 'unknown')] += 1
            if rhyme.get('stanza_structure'):
                stats['stanza_types'][rhyme.get('stanza_structure', 'unknown')] += 1
        
        # Free verse analysis is stored under quality_metrics
        fv = analysis.get('quality_metrics', {}).get('free_verse_analysis', {})
        if fv:
            if fv.get('enjambement_score') is not None:
                stats['enjambement_scores'].append({
                    'poem_id': c.get('poem_id', 'unknown'),
                    'title': c.get('title', 'Untitled'),
                    'score': fv['enjambement_score']
                })
            if fv.get('line_variation_score') is not None:
                stats['line_variations'].append({
                    'poem_id': c.get('poem_id', 'unknown'),
                    'title': c.get('title', 'Untitled'),
                    'score': fv['line_variation_score']
                })
            if fv.get('prose_poetry_score') is not None:
                stats['prose_poetry_scores'].append({
                    'poem_id': c.get('poem_id', 'unknown'),
                    'title': c.get('title', 'Untitled'),
                    'score': fv['prose_poetry_score']
                })
    
    n = len(contributions)
    if n > 0:
        stats['avg_lines'] = stats['total_lines'] / n
        stats['avg_syllables'] = stats['total_syllables'] / stats['total_lines'] if stats['total_lines'] > 0 else 0
    
    return stats


def export_to_r_csv(contributions: List[Dict]) -> str:
    """Export data to CSV format for R Studio"""
    output = io.StringIO()
    
    fieldnames = [
        'poem_id', 'title', 'batch_id', 'source_file', 
        'author', 'collection', 'year',
        'lines', 'total_syllables', 'avg_syllables_per_line',
        'meter', 'form', 'theme', 'register',
        'has_radif', 'has_qafiyeh', 'rhyme_scheme', 'stanza_type',
        'enjambement_score', 'line_variation_score', 'prose_poetry_score',
        'free_verse_confidence', 'free_verse_assessment'
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for c in contributions:
        tags = c.get('tags', [])
        analysis = c.get('analysis', {})
        structural = analysis.get('structural', {})
        rhyme = analysis.get('rhyme', {})
        fv = analysis.get('free_verse_analysis', {})
        vol_meta = c.get('volume_metadata', {})
        
        meter = next((t.replace('meter:', '') for t in tags if t.startswith('meter:')), '')
        form = next((t.replace('form:', '') for t in tags if t.startswith('form:')), '')
        theme = next((t.replace('theme:', '') for t in tags if t.startswith('theme:')), '')
        register = next((t.replace('register:', '') for t in tags if t.startswith('register:')), '')
        
        syllables = structural.get('syllables_per_line', [])
        valid_syllables = [s for s in syllables if s > 0]
        total_syl = sum(valid_syllables)
        avg_syl = total_syl / len(valid_syllables) if valid_syllables else 0
        
        row = {
            'poem_id': c.get('poem_id', ''),
            'title': c.get('title', ''),
            'batch_id': c.get('upload_batch_id', ''),
            'source_file': c.get('source_filename', ''),
            'author': vol_meta.get('author', ''),
            'collection': vol_meta.get('collection', ''),
            'year': vol_meta.get('year', ''),
            'lines': structural.get('lines', 0),
            'total_syllables': total_syl,
            'avg_syllables_per_line': round(avg_syl, 2),
            'meter': meter,
            'form': form,
            'theme': theme,
            'register': register,
            'has_radif': 1 if rhyme.get('has_radif') else 0,
            'has_qafiyeh': 1 if rhyme.get('rhyme_scheme') else 0,
            'rhyme_scheme': rhyme.get('rhyme_scheme', ''),
            'stanza_type': rhyme.get('stanza_structure', ''),
            'enjambement_score': round(fv.get('enjambement_score', 0), 3) if fv.get('enjambement_score') else '',
            'line_variation_score': round(fv.get('line_variation_score', 0), 3) if fv.get('line_variation_score') else '',
            'prose_poetry_score': round(fv.get('prose_poetry_score', 0), 3) if fv.get('prose_poetry_score') else '',
            'free_verse_confidence': round(fv.get('confidence', 0), 3) if fv.get('confidence') else '',
            'free_verse_assessment': fv.get('assessment', '')
        }
        
        writer.writerow(row)
    
    return output.getvalue()


def export_syllable_data_csv(stats: Dict) -> str:
    """Export syllable pattern data for R"""
    output = io.StringIO()
    
    fieldnames = ['poem_id', 'title', 'avg_syllables', 'variance', 'pattern']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for sp in stats['syllable_patterns']:
        writer.writerow({
            'poem_id': sp['poem_id'],
            'title': sp['title'],
            'avg_syllables': round(sp['avg'], 2),
            'variance': sp['variance'],
            'pattern': ','.join(map(str, sp['pattern']))
        })
    
    return output.getvalue()


def export_word_freq_csv(word_freqs: List[tuple], collection_name: str = "") -> str:
    """Export word frequencies for R"""
    output = io.StringIO()
    
    fieldnames = ['rank', 'word', 'frequency', 'collection']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for i, (word, freq) in enumerate(word_freqs, 1):
        writer.writerow({
            'rank': i,
            'word': word,
            'frequency': freq,
            'collection': collection_name
        })
    
    return output.getvalue()


def plot_word_frequency_bar(word_freqs: List[tuple], top_n: int, title: str = "Word Frequencies"):
    """Create bar chart of word frequencies"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available for visualization")
        return
    
    if not word_freqs:
        st.info("No data available")
        return
    
    display_freqs = word_freqs[:top_n]
    words, counts = zip(*display_freqs)
    
    fig = go.Figure(data=[
        go.Bar(x=list(words), y=list(counts), marker_color='steelblue')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Word",
        yaxis_title="Frequency",
        xaxis_tickangle=-45,
        height=500 + (top_n // 20) * 50
    )
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    export_chart_to_file(fig, "word_frequencies.html", "dl_wordfreq")


def plot_clickable_word_cloud(word_freqs: List[tuple], contributions: List[Dict]):
    """Display words as clickable buttons"""
    st.markdown("### Click a word to see poems containing it:")
    
    cols = st.columns(5)
    
    for i, (word, count) in enumerate(word_freqs[:50]):
        col_idx = i % 5
        with cols[col_idx]:
            if st.button(f"{word} ({count})", key=f"word_{word}"):
                st.session_state.viz_selected_word = word


def display_poems_for_word(word: str, contributions: List[Dict]):
    """Display poems containing a specific word"""
    matching = find_poems_with_word(contributions, word)
    
    st.markdown(f"### Poems containing '{word}' ({len(matching)} found)")
    
    for poem in matching[:20]:
        title = poem.get('title', 'Untitled')
        source = poem.get('source_filename', 'Unknown')
        
        with st.expander(f"{title}"):
            st.caption(f"Source: {source}")
            
            text = poem.get('raw_text', '')
            highlighted = re.sub(
                f'({re.escape(word)})',
                r'**\1**',
                text,
                flags=re.IGNORECASE
            )
            st.markdown(highlighted[:1000] + ('...' if len(highlighted) > 1000 else ''))
    
    if len(matching) > 20:
        st.info(f"Showing 20 of {len(matching)} poems")


def plot_word_frequency_comparison(data_sets: Dict[str, List[tuple]], top_n: int = 20, sort_by: str = "frequency"):
    """Compare word frequencies across multiple collections with sorting"""
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
    
    # Sort words based on option
    if sort_by == "alphabetical":
        words_sorted = sorted(all_words)
    elif sort_by == "freq_desc":
        # Sort by total frequency across all sets
        word_totals = {}
        for word in all_words:
            total = sum(dict(freqs).get(word, 0) for freqs in data_sets.values())
            word_totals[word] = total
        words_sorted = sorted(all_words, key=lambda w: word_totals[w], reverse=True)
    elif sort_by == "freq_asc":
        word_totals = {}
        for word in all_words:
            total = sum(dict(freqs).get(word, 0) for freqs in data_sets.values())
            word_totals[word] = total
        words_sorted = sorted(all_words, key=lambda w: word_totals[w])
    else:
        words_sorted = sorted(all_words)
    
    # Plotly default color sequence
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    
    fig = go.Figure()
    
    for i, (name, freqs) in enumerate(data_sets.items()):
        freq_dict = dict(freqs)
        values = [freq_dict.get(w, 0) for w in words_sorted]
        
        fig.add_trace(go.Bar(
            name=name,
            x=words_sorted,
            y=values,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title="Word Frequency Comparison",
        xaxis_title="Word",
        yaxis_title="Frequency",
        xaxis_tickangle=-45,
        barmode='group',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    export_chart_to_file(fig, "word_freq_comparison.html", "dl_wordcomp")


def plot_theme_distribution(contributions: List[Dict], clickable: bool = True):
    """Plot theme distribution with optional click support"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available")
        return
    
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
        go.Pie(
            labels=list(themes), 
            values=list(counts), 
            hole=0.3,
            marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])
        )
    ])
    
    fig.update_layout(title="Theme Distribution", height=400)
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    export_chart_to_file(fig, "theme_distribution.html", "dl_theme")
    
    if clickable:
        st.markdown("### Click a theme to see poems:")
        cols = st.columns(min(5, len(themes)))
        for i, (theme, count) in enumerate(zip(themes, counts)):
            with cols[i % 5]:
                if st.button(f"{theme} ({count})", key=f"theme_{theme}"):
                    st.session_state.viz_selected_theme = theme


def display_poems_for_theme(theme: str, contributions: List[Dict]):
    """Display poems with a specific theme"""
    matching = find_poems_with_tag(contributions, 'theme', theme)
    
    st.markdown(f"### Poems with theme '{theme}' ({len(matching)} found)")
    
    for poem in matching[:20]:
        title = poem.get('title', 'Untitled')
        source = poem.get('source_filename', 'Unknown')
        
        with st.expander(f"{title}"):
            st.caption(f"Source: {source}")
            text = poem.get('raw_text', '')
            st.text(text[:500] + ('...' if len(text) > 500 else ''))
    
    if len(matching) > 20:
        st.info(f"Showing 20 of {len(matching)} poems")


def plot_meter_distribution(contributions: List[Dict], clickable: bool = True):
    """Plot meter distribution"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available")
        return
    
    meter_counts = Counter()
    
    for c in contributions:
        tags = c.get('tags', [])
        
        is_free_verse = 'form:free_verse' in tags
        
        meter_found = False
        for tag in tags:
            if tag.startswith('meter:'):
                meter = tag.replace('meter:', '')
                if is_free_verse and meter != 'free_verse':
                    meter_counts['free_verse'] += 1
                else:
                    meter_counts[meter] += 1
                meter_found = True
                break
        
        if not meter_found and is_free_verse:
            meter_counts['free_verse'] += 1
    
    if not meter_counts:
        st.info("No meter data available")
        return
    
    sorted_meters = meter_counts.most_common()
    meters, counts = zip(*sorted_meters)
    
    colors = ['steelblue' if m == 'free_verse' else 'darkgreen' for m in meters]
    
    fig = go.Figure(data=[
        go.Bar(x=list(meters), y=list(counts), marker_color=colors)
    ])
    
    fig.update_layout(title="Meter Distribution", xaxis_title="Meter", yaxis_title="Count", height=400)
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    export_chart_to_file(fig, "meter_distribution.png", "dl_meter")
    
    if clickable:
        st.markdown("### Click a meter to see poems:")
        cols = st.columns(min(5, len(meters)))
        for i, (meter, count) in enumerate(zip(meters, counts)):
            with cols[i % 5]:
                if st.button(f"{meter} ({count})", key=f"meter_{meter}"):
                    st.session_state.viz_selected_meter = meter


def display_poems_for_meter(meter: str, contributions: List[Dict]):
    """Display poems with a specific meter"""
    matching = find_poems_with_tag(contributions, 'meter', meter)
    
    st.markdown(f"### Poems with meter '{meter}' ({len(matching)} found)")
    
    for poem in matching[:20]:
        title = poem.get('title', 'Untitled')
        source = poem.get('source_filename', 'Unknown')
        analysis = poem.get('analysis', {})
        
        with st.expander(f"{title}"):
            st.caption(f"Source: {source}")
            
            if analysis.get('meter'):
                meter_data = analysis['meter']
                st.write(f"**Confidence:** {meter_data.get('confidence', 'N/A')}")
                st.write(f"**Accuracy:** {meter_data.get('accuracy', 'N/A')}")
            
            text = poem.get('raw_text', '')
            st.text(text[:500] + ('...' if len(text) > 500 else ''))
    
    if len(matching) > 20:
        st.info(f"Showing 20 of {len(matching)} poems")


def plot_syllable_patterns(stats: Dict):
    """Plot syllable pattern analysis"""
    if not PLOTLY_AVAILABLE or not stats['syllable_patterns']:
        st.info("No syllable pattern data available")
        return
    
    # Average syllables distribution
    avgs = [sp['avg'] for sp in stats['syllable_patterns']]
    variances = [sp['variance'] for sp in stats['syllable_patterns']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure(data=[go.Histogram(x=avgs, nbinsx=20, marker_color='teal')])
        fig1.update_layout(
            title="Average Syllables per Line Distribution",
            xaxis_title="Average Syllables",
            yaxis_title="Number of Poems",
            height=350
        )
        st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)
        export_chart_to_file(fig1, "syllables_avg.png", "dl_syl_avg")
    
    with col2:
        fig2 = go.Figure(data=[go.Histogram(x=variances, nbinsx=20, marker_color='coral')])
        fig2.update_layout(
            title="Line Length Variance Distribution",
            xaxis_title="Variance (max-min syllables)",
            yaxis_title="Number of Poems",
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
        export_chart_to_file(fig2, "syllables_variance.png", "dl_syl_var")


def plot_enjambement_analysis(stats: Dict):
    """Plot enjambement score analysis"""
    if not PLOTLY_AVAILABLE or not stats['enjambement_scores']:
        st.info("No enjambement data available")
        return
    
    scores = [e['score'] for e in stats['enjambement_scores']]
    
    fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=20, marker_color='purple')])
    fig.update_layout(
        title="Enjambement Score Distribution",
        xaxis_title="Enjambement Score (0 = none, 1 = heavy)",
        yaxis_title="Number of Poems",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    export_chart_to_file(fig, "enjambement_distribution.png", "dl_enj")
    
    # Top enjambement poems
    sorted_enj = sorted(stats['enjambement_scores'], key=lambda x: x['score'], reverse=True)[:10]
    
    with st.expander("Top 10 poems by enjambement"):
        for e in sorted_enj:
            st.write(f"- {e['title']}: {e['score']:.3f}")


def plot_prose_poetry_analysis(stats: Dict):
    """Plot prose poetry score analysis"""
    if not PLOTLY_AVAILABLE or not stats['prose_poetry_scores']:
        st.info("No prose poetry data available")
        return
    
    scores = [p['score'] for p in stats['prose_poetry_scores']]
    
    fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=20, marker_color='orange')])
    fig.update_layout(
        title="Prose Poetry Score Distribution",
        xaxis_title="Prose Poetry Score (0 = verse, 1 = prose-like)",
        yaxis_title="Number of Poems",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    export_chart_to_file(fig, "prose_poetry_distribution.png", "dl_prose")


def plot_comprehensive_stats(stats: Dict):
    """Display comprehensive analysis statistics"""
    st.subheader("Corpus Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Free Verse", stats['free_verse_count'])
    with col2:
        st.metric("Classical Forms", stats['classical_count'])
    with col3:
        st.metric("With Radif", stats['with_radif'])
    with col4:
        st.metric("With Qafiyeh", stats['with_qafiyeh'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Lines/Poem", f"{stats['avg_lines']:.1f}")
    with col2:
        st.metric("Avg Syllables/Line", f"{stats['avg_syllables']:.1f}")
    with col3:
        st.metric("Total Lines", stats['total_lines'])
    with col4:
        st.metric("Total Syllables", stats['total_syllables'])
    
    if stats['stanza_types']:
        st.markdown("### Stanza Types")
        stanzas, counts = zip(*stats['stanza_types'].most_common(10))
        fig = go.Figure(data=[go.Bar(x=list(stanzas), y=list(counts), marker_color='purple')])
        fig.update_layout(height=300, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        export_chart_to_file(fig, "stanza_types.html", "dl_stanza")
    
    if stats['registers']:
        st.markdown("### Register Distribution")
        registers, counts = zip(*stats['registers'].most_common())
        fig = go.Figure(data=[go.Pie(
            labels=list(registers), 
            values=list(counts), 
            hole=0.4,
            marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880'])
        )])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        export_chart_to_file(fig, "register_distribution.html", "dl_register")


def plot_timeline(contributions: List[Dict]):
    """Plot poems over time"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available")
        return
    
    year_counts = Counter()
    
    for c in contributions:
        year = (
            c.get('volume_metadata', {}).get('year') or
            c.get('metadata', {}).get('volume_year') or
            c.get('metadata', {}).get('publication_year')
        )
        if year:
            year_counts[int(year)] += 1
    
    if not year_counts:
        st.info("No year data available. Add metadata in the Library page.")
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
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    export_chart_to_file(fig, "timeline.html", "dl_timeline")
    
    st.subheader("Timeline Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Year Range", f"{min(years)} - {max(years)}")
    with col2:
        st.metric("Years Covered", len(years))
    with col3:
        most_productive = max(year_counts.items(), key=lambda x: x[1])
        st.metric("Most Productive Year", f"{most_productive[0]} ({most_productive[1]})")


def _calculate_mtld_for_viz(words: List[str], threshold: float = 0.72) -> float:
    """Calculate MTLD (Measure of Textual Lexical Diversity) - McCarthy & Jarvis 2010"""
    if len(words) < 10:
        return 0.0
    
    def mtld_forward(word_list):
        factors = 0.0
        start = 0
        for i in range(1, len(word_list) + 1):
            segment = word_list[start:i]
            ttr = len(set(segment)) / len(segment)
            if ttr <= threshold:
                factors += 1
                start = i
        if start < len(word_list):
            remaining = word_list[start:]
            ttr = len(set(remaining)) / len(remaining)
            factors += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
        return factors
    
    forward = mtld_forward(words)
    backward = mtld_forward(words[::-1])
    avg_factors = (forward + backward) / 2
    
    return len(words) / avg_factors if avg_factors > 0 else float(len(words))


def plot_lexical_diversity_comparison(data_sets: Dict[str, List[Dict]]):
    """Compare lexical diversity (MTLD) across collections"""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not available")
        return
    
    diversities = {}
    
    for name, contributions in data_sets.items():
        text = get_combined_text(contributions)
        words = re.findall(r'[а-яёғӣӯҳқҷА-ЯЁҒӢӮҲҚҶ]+', text.lower())
        
        if words:
            mtld = _calculate_mtld_for_viz(words)
            diversities[name] = mtld
    
    if not diversities:
        st.info("No data available")
        return
    
    names = list(diversities.keys())
    values = [diversities[n] for n in names]
    
    fig = go.Figure(data=[
        go.Bar(x=names, y=values, marker_color='coral')
    ])
    
    fig.update_layout(
        title="Lexical Diversity (MTLD)",
        xaxis_title="Collection",
        yaxis_title="MTLD Score (higher = more diverse)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    export_chart_to_file(fig, "lexical_diversity.html", "dl_lexdiv")


def main():
    init_viz_state()
    
    st.title("Visualizations")
    st.markdown("Interactive charts, analysis, and R export")
    
    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for visualizations. Install with: pip install plotly")
        st.stop()
    
    st.markdown("---")
    
    if st.session_state.viz_contributions is None:
        with st.spinner("Loading data..."):
            st.session_state.viz_contributions = load_all_contributions()
    
    contributions = st.session_state.viz_contributions
    
    if not contributions:
        st.warning("No data available. Use the Analyze page to add poems first.")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Poems", len(contributions))
    with col2:
        batches = group_contributions_by_batch(contributions)
        real_batches = len([b for b in batches if b != '_ungrouped_'])
        st.metric("Batches", real_batches)
    with col3:
        if st.button("Reload Data"):
            st.session_state.viz_contributions = None
            st.session_state.viz_selected_word = None
            st.session_state.viz_selected_theme = None
            st.session_state.viz_selected_meter = None
            st.rerun()
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Word Frequencies", 
        "Compare Collections", 
        "Distributions",
        "Detailed Analysis",
        "Timeline",
        "R Export"
    ])
    
    with tab1:
        st.header("Word Frequency Analysis")
        
        if st.session_state.viz_selected_word:
            display_poems_for_word(st.session_state.viz_selected_word, contributions)
            if st.button("Back to word list"):
                st.session_state.viz_selected_word = None
                st.rerun()
        else:
            grouped_by_batch = group_contributions_by_batch(contributions)
            batch_options = ["All"] + [
                f"{v['label']} ({len(v['poems'])})" 
                for k, v in grouped_by_batch.items() if k != '_ungrouped_'
            ]
            
            selected = st.selectbox("Select volume:", batch_options, key="viz_batch_select")
            
            if selected == "All":
                text = get_combined_text(contributions)
            else:
                for batch_id, batch_data in grouped_by_batch.items():
                    if f"{batch_data['label']} ({len(batch_data['poems'])})" == selected:
                        text = get_combined_text(batch_data['poems'])
                        break
            
            top_n = st.slider("Number of words to display", 10, 100, 30, key="viz_top_n")
            
            word_freqs = extract_word_frequencies(text, top_n)
            
            plot_word_frequency_bar(word_freqs, top_n, f"Top {top_n} Words")
            
            st.markdown("---")
            plot_clickable_word_cloud(word_freqs, contributions)
            
            with st.expander("Show as table"):
                for i, (word, count) in enumerate(word_freqs, 1):
                    st.write(f"{i}. **{word}**: {count}")
    
    with tab2:
        st.header("Compare Batches")
        
        grouped = group_contributions_by_batch(contributions)
        real_batches = {k: v for k, v in grouped.items() if k != '_ungrouped_'}
        
        if len(real_batches) < 2:
            st.info("Need at least 2 batches to compare. Upload more files in the Analyze page.")
        else:
            available = [
                (batch_id, f"{data['label']} ({len(data['poems'])} poems)")
                for batch_id, data in real_batches.items()
            ]
            
            selected_batches = st.multiselect(
                "Select batches to compare:",
                [a[0] for a in available],
                format_func=lambda x: next(a[1] for a in available if a[0] == x),
                default=[available[0][0], available[1][0]] if len(available) >= 2 else [],
                key="viz_compare_select"
            )
            
            if len(selected_batches) >= 2:
                st.subheader("Word Frequency Comparison")
                
                col1, col2 = st.columns(2)
                with col1:
                    compare_top_n = st.slider("Words to compare", 10, 100, 30, key="compare_top_n")
                with col2:
                    sort_option = st.selectbox(
                        "Sort by",
                        ["freq_desc", "freq_asc", "alphabetical"],
                        format_func=lambda x: {
                            "freq_desc": "Frequency (high to low)",
                            "freq_asc": "Frequency (low to high)",
                            "alphabetical": "Alphabetical"
                        }[x],
                        key="compare_sort"
                    )
                
                data_sets = {}
                for batch_id in selected_batches:
                    batch_data = real_batches[batch_id]
                    name = batch_data['label'][:40]
                    text = get_combined_text(batch_data['poems'])
                    data_sets[name] = extract_word_frequencies(text, compare_top_n)
                
                plot_word_frequency_comparison(data_sets, compare_top_n, sort_option)
                
                st.subheader("Lexical Diversity")
                
                comparison_data = {
                    real_batches[b]['label'][:40]: real_batches[b]['poems'] 
                    for b in selected_batches
                }
                plot_lexical_diversity_comparison(comparison_data)
            else:
                st.info("Select at least 2 batches to compare")
    
    with tab3:
        st.header("Distributions")
        
        if st.session_state.viz_selected_theme:
            display_poems_for_theme(st.session_state.viz_selected_theme, contributions)
            if st.button("Back to themes", key="back_theme"):
                st.session_state.viz_selected_theme = None
                st.rerun()
        elif st.session_state.viz_selected_meter:
            display_poems_for_meter(st.session_state.viz_selected_meter, contributions)
            if st.button("Back to meters", key="back_meter"):
                st.session_state.viz_selected_meter = None
                st.rerun()
        else:
            # Batch filter for distributions
            grouped_dist = group_contributions_by_batch(contributions)
            batch_opts_dist = ["All"] + [
                f"{v['label']} ({len(v['poems'])})" 
                for k, v in grouped_dist.items() if k != '_ungrouped_'
            ]
            selected_dist = st.selectbox("Select volume:", batch_opts_dist, key="dist_batch_select")
            
            if selected_dist == "All":
                filtered_dist = contributions
            else:
                filtered_dist = contributions
                for batch_id, batch_data in grouped_dist.items():
                    if f"{batch_data['label']} ({len(batch_data['poems'])})" == selected_dist:
                        filtered_dist = batch_data['poems']
                        break
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Theme Distribution")
                plot_theme_distribution(filtered_dist, clickable=True)
            
            with col2:
                st.subheader("Meter Distribution")
                plot_meter_distribution(filtered_dist, clickable=True)
    
    with tab4:
        st.header("Detailed Analysis")
        
        # Batch filter for detailed analysis
        grouped_detail = group_contributions_by_batch(contributions)
        batch_opts_detail = ["All"] + [
            f"{v['label']} ({len(v['poems'])})" 
            for k, v in grouped_detail.items() if k != '_ungrouped_'
        ]
        selected_detail = st.selectbox("Select volume:", batch_opts_detail, key="detail_batch_select")
        
        if selected_detail == "All":
            filtered_detail = contributions
        else:
            filtered_detail = contributions
            for batch_id, batch_data in grouped_detail.items():
                if f"{batch_data['label']} ({len(batch_data['poems'])})" == selected_detail:
                    filtered_detail = batch_data['poems']
                    break
        
        stats = extract_analysis_stats(filtered_detail)
        
        plot_comprehensive_stats(stats)
        
        st.markdown("---")
        st.subheader("Syllable Patterns")
        plot_syllable_patterns(stats)
        
        st.markdown("---")
        st.subheader("Enjambement Analysis")
        plot_enjambement_analysis(stats)
        
        st.markdown("---")
        st.subheader("Prose Poetry Scores")
        plot_prose_poetry_analysis(stats)
    
    with tab5:
        st.header("Timeline")
        plot_timeline(contributions)
    
    with tab6:
        st.header("Export for R Studio")
        st.info("Download data in CSV format for analysis in R, SPSS, or Excel.")
        
        stats = extract_analysis_stats(contributions)
        
        st.subheader("Available Exports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Main Dataset")
            st.write("Complete poem-level data with all analysis results")
            
            main_csv = export_to_r_csv(contributions)
            st.download_button(
                label="Download poems.csv",
                data=main_csv,
                file_name="tajik_poems_analysis.csv",
                mime="text/csv"
            )
            
            st.markdown("### Syllable Patterns")
            st.write("Detailed syllable pattern data per poem")
            
            syl_csv = export_syllable_data_csv(stats)
            st.download_button(
                label="Download syllables.csv",
                data=syl_csv,
                file_name="tajik_syllable_patterns.csv",
                mime="text/csv"
            )
        
        with col2:
            st.markdown("### Word Frequencies")
            st.write("Word frequency data for selected collection")
            
            grouped_by_batch = group_contributions_by_batch(contributions)
            batch_options = ["All"] + [
                f"{v['label']}" 
                for k, v in grouped_by_batch.items() if k != '_ungrouped_'
            ]
            
            export_batch = st.selectbox("Collection:", batch_options, key="export_batch")
            
            if export_batch == "All":
                text = get_combined_text(contributions)
            else:
                for batch_id, batch_data in grouped_by_batch.items():
                    if batch_data['label'] == export_batch:
                        text = get_combined_text(batch_data['poems'])
                        break
            
            word_freqs = extract_word_frequencies(text, 500)
            word_csv = export_word_freq_csv(word_freqs, export_batch)
            
            st.download_button(
                label="Download word_frequencies.csv",
                data=word_csv,
                file_name=f"tajik_word_freq_{export_batch.replace(' ', '_')[:20]}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        st.markdown("""
        ### R Code Example
        ```r
        # Load the data
        poems <- read.csv("tajik_poems_analysis.csv")
        syllables <- read.csv("tajik_syllable_patterns.csv")
        words <- read.csv("tajik_word_freq_All.csv")
        
        # Basic analysis
        summary(poems$avg_syllables_per_line)
        table(poems$meter)
        table(poems$form)
        
        # Correlation between enjambement and prose poetry score
        cor(poems$enjambement_score, poems$prose_poetry_score, use="complete.obs")
        
        # Plot meter distribution
        library(ggplot2)
        ggplot(poems, aes(x=meter)) + geom_bar() + theme_minimal()
        ```
        """)


main()
