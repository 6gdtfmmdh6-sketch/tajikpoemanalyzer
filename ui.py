#!/usr/bin/env python3
"""
Enhanced Web-UI for Tajik Poetry Analyzer
Supports both classical and enhanced analysis with free verse detection

Features:
1. Classical ʿArūḍ analysis (16 meters)
2. Enhanced analysis with free verse detection
3. Modern verse metrics
4. PDF and OCR support
5. Scientific quality validation
6. Corpus management with library functions
7. Chronological analysis across periods
"""

import streamlit as st
from pathlib import Path
import tempfile
import re
import json
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from dataclasses import asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from consolidated analyzer
try:
    from analyzer import (
        TajikPoemAnalyzer,
        EnhancedTajikPoemAnalyzer,
        AnalysisConfig,
        MeterConfidence,
        QualityValidator,
        ExcelReporter,
    )
    ANALYZER_AVAILABLE = True
    logger.info("Analyzer loaded successfully")
except ImportError as e:
    logger.error(f"Analyzer not available: {e}")
    ANALYZER_AVAILABLE = False

# NEW: Import extended library manager
try:
    from extended_corpus_manager import (
        TajikLibraryManager,
        VolumeMetadata,
        Genre,
        Period,
        Genre,
        Period
    )
    LIBRARY_MANAGER_AVAILABLE = True
    logger.info("Extended Library Manager loaded successfully")
except ImportError as e:
    logger.warning(f"Extended Library Manager not available: {e}")
    LIBRARY_MANAGER_AVAILABLE = False

# Import basic corpus manager as fallback
try:
    from corpus_manager import TajikCorpusManager
    CORPUS_MANAGER_AVAILABLE = True
    logger.info("Basic Corpus Manager loaded successfully")
except ImportError as e:
    logger.warning(f"Basic Corpus Manager not available: {e}")
    CORPUS_MANAGER_AVAILABLE = False

try:
    from pdf_handler import read_file_with_pdf_support
except ImportError:
    st.error("Error: Could not import pdf_handler. Please ensure pdf_handler.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Tajik Poetry Analyzer",
    page_icon="📖",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main {max-width: 1200px; margin: 0 auto;}
    h1 {text-align: center; color: #2c3e50;}
    .stButton>button {width: 100%;}
    .analyzer-badge {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    .free-verse-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    .classical-badge {
        background-color: #4a6fa5;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    .radif-badge {
        background-color: #9b59b6;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    .library-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 20px 0;
    }
    .timeline-chart {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
class TajikCyrillicConfig(AnalysisConfig):
    """Configuration specific to Tajik Cyrillic poetry"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tajik_cyrillic_alphabet = set(
            'АБВГҒДЕЁЖЗИӢЙКҚЛМНОПРСТУӮФХҲЧҶШЪЭЮЯ'
            'абвгғдеёжзиӣйкқлмнопрстуӯфхҳчҷшъэюя'
            '0123456789 .,!?;:-–—()[]{}"\'«»'
        )
        self.min_poem_lines = 3
        self.max_poem_lines = 100


class UIPoemSplitter:
    """Poem splitter for UI"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.UIPoemSplitter")
        
    def get_split_suggestions(self, text: str) -> List[int]:
        """Returns line indices where a new poem is likely to start"""
        lines = text.split('\n')
        suggestions = []
        
        for i, line in enumerate(lines):
            score = 0
            
            if self._looks_like_title(line):
                score += 2
            
            if i > 0 and not lines[i-1].strip() and len(line.strip()) > 0:
                score += 1.5
            
            if re.match(r'^[\*\-=]{3,}$', line.strip()):
                suggestions.append(max(0, i-1))
                continue
                
            if re.match(r'^\s*[\d]+[\.\)]\s*[A-ZА-Я]', line):
                score += 1
            
            if score >= 1.5:
                suggestions.append(i)
        
        if suggestions:
            filtered = [suggestions[0]]
            for s in suggestions[1:]:
                if s - filtered[-1] > 3:
                    filtered.append(s)
            suggestions = filtered
        
        return suggestions

    def _looks_like_title(self, line: str) -> bool:
        """Simple heuristic to recognize title lines"""
        line = line.strip()
        if not line or len(line) > 150:
            return False
        if line.endswith(('.', '!', '?', ':', ',')):
            return False
        if not line[0].isupper():
            return False
        if line.isupper():
            return False
        return True


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
@st.cache_resource
def load_classical_analyzer():
    """Initialize classical analyzer (cached)"""
    config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
    return TajikPoemAnalyzer(config=config)


@st.cache_resource
def load_enhanced_analyzer():
    """Initialize enhanced analyzer (cached)"""
    config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
    return EnhancedTajikPoemAnalyzer(config=config, enable_corpus=False)


def split_poems_auto(text: str) -> list:
    """Split text into poems automatically"""
    if '*****' in text:
        poems = [p.strip() for p in text.split('*****')]
    elif '\n\n\n' in text:
        poems = [p.strip() for p in text.split('\n\n\n')]
    else:
        poems = [p.strip() for p in text.split('\n\n')]
    return [p for p in poems if len(p) > 50]


def split_text_at_indices(text: str, split_indices: List[int]) -> List[str]:
    """Split text at specified line indices"""
    all_lines = text.split('\n')
    poems = []
    start_idx = 0
    
    for split_idx in sorted(split_indices):
        poem_lines = all_lines[start_idx:split_idx]
        poem_text = '\n'.join(poem_lines).strip()
        if poem_text:
            poems.append(poem_text)
        start_idx = split_idx
    
    final_poem = '\n'.join(all_lines[start_idx:]).strip()
    if final_poem:
        poems.append(final_poem)
    
    return poems


def display_classical_results(analysis, poem_num: int, poem_text: str):
    """Display classical analysis results"""
    structural = analysis.structural
    content = analysis.content
    validation = analysis.quality_metrics
    
    with st.expander(f"Poem {poem_num} - Classical Analysis - {content.total_words} words", expanded=True):
        # Content
        st.subheader("Content")
        st.text(poem_text[:500] + "..." if len(poem_text) > 500 else poem_text)
        
        st.markdown("---")
        
        # ʿArūḍ Meter Analysis
        st.subheader("ʿArūḍ Meter Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            meter_name = structural.aruz_analysis.identified_meter
            st.metric("Identified Meter", meter_name.title())
        
        with col2:
            confidence = structural.meter_confidence.value
            st.metric("Confidence", f"{confidence.title()}")
        
        with col3:
            prosodic = structural.prosodic_consistency
            st.metric("Prosodic Consistency", f"{prosodic:.1%}")
        
        # Pattern display
        if structural.aruz_analysis.pattern_match:
            st.write(f"**Pattern:** `{structural.aruz_analysis.pattern_match}`")
            if structural.aruz_analysis.variations_detected:
                st.write(f"**Variations:** {', '.join(structural.aruz_analysis.variations_detected)}")
        
        st.markdown("---")
        
        # Structural Analysis
        st.subheader("Structural Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Lines:** {structural.lines}")
            st.write(f"**Average Syllables per Line:** {structural.avg_syllables:.1f}")
            st.write(f"**Stanza Form:** {structural.stanza_structure}")
        
        with col2:
            st.write(f"**Rhyme Pattern:** {structural.rhyme_pattern}")
        
        st.markdown("---")
        
        # Content Analysis
        st.subheader("Content Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Words", content.total_words)
            st.metric("Unique Words", content.unique_words)
        
        with col2:
            st.metric("Lexical Diversity", f"{content.lexical_diversity:.1%}")
            st.metric("Register", content.stylistic_register.title())
        
        with col3:
            st.metric("Neologisms", len(content.neologisms))
            st.metric("Archaisms", len(content.archaisms))
        
        # Word Frequencies
        if content.word_frequencies:
            st.write("**Top Words:**")
            top_words = ", ".join([f"{w}({c})" for w, c in content.word_frequencies[:10]])
            st.write(top_words)
        
        # Neologisms
        if content.neologisms:
            st.write(f"**Neologisms:** {', '.join(content.neologisms[:10])}")
        
        # Archaisms
        if content.archaisms:
            st.write(f"**Archaisms:** {', '.join(content.archaisms)}")
        
        # Themes
        active_themes = [k for k, v in content.theme_distribution.items() if v > 0]
        if active_themes:
            st.write(f"**Themes:** {', '.join(active_themes)}")
        
        st.markdown("---")
        
        # Rhyme Analysis with Radīf Detection
        st.subheader("Rhyme Analysis (Qāfiyeh/Radīf)")
        
        # Check for global Radīf
        radif_values = [r.radif for r in structural.rhyme_scheme if r.radif]
        if radif_values and len(set(radif_values)) == 1:
            global_radif = radif_values[0]
            st.success(f"🔁 **Global Radīf Detected:** `{global_radif}` (appears in {len(radif_values)}/{len(structural.rhyme_scheme)} lines)")
            st.info("Meter analysis was performed on lines with Radīf removed for accuracy.")
        
        if structural.rhyme_scheme:
            for i, rhyme in enumerate(structural.rhyme_scheme[:5]):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Line {i+1}**")
                with col2:
                    st.write(f"Qāfiyeh: `{rhyme.qafiyeh}`")
                with col3:
                    st.write(f"Radīf: `{rhyme.radif or '—'}`")
                with col4:
                    st.write(f"Type: {rhyme.rhyme_type}")
            
            if len(structural.rhyme_scheme) > 5:
                st.caption(f"... and {len(structural.rhyme_scheme) - 5} more lines")
        
        st.markdown("---")
        
        # Quality Validation
        st.subheader("Quality Validation")
        
        quality_score = validation.get('quality_score', 0)
        reliability = validation.get('reliability', 'unknown')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Quality Score", f"{quality_score:.0%}")
        with col2:
            st.metric("Reliability", reliability.title())
        
        warnings = validation.get('warnings', [])
        if warnings:
            st.warning("**Warnings:**")
            for w in warnings:
                st.write(f"{w}")
        
        recommendations = validation.get('recommendations', [])
        if recommendations:
            st.info("**Recommendations:**")
            for r in recommendations:
                st.write(f"{r}")


def display_enhanced_results(analysis, poem_num: int, poem_text: str):
    """Display enhanced analysis results with free verse detection"""
    structural = analysis.structural
    content = analysis.content
    validation = analysis.quality_metrics
    
    # Create title
    title = f"Poem {poem_num} - Enhanced Analysis - {content.total_words} words"
    
    with st.expander(title, expanded=True):
        # Display badge inside expander
        if structural.is_free_verse:
            st.markdown('<span class="free-verse-badge">Free Verse Detected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="classical-badge">Classical Form</span>', unsafe_allow_html=True)
        
        # Content
        st.subheader("Content")
        st.text(poem_text[:500] + "..." if len(poem_text) > 500 else poem_text)
        
        st.markdown("---")
        
        # Analysis Summary
        st.subheader("Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if structural.is_free_verse:
                st.metric("Form", "Free Verse")
                st.metric("Free Verse Confidence", f"{structural.free_verse_confidence:.0%}")
            else:
                meter_name = structural.aruz_analysis.identified_meter
                st.metric("Identified Meter", meter_name.title())
        
        with col2:
            confidence = structural.meter_confidence.value
            st.metric("Confidence", confidence.title())
            prosodic = structural.prosodic_consistency
            st.metric("Prosodic Consistency", f"{prosodic:.1%}")
        
        with col3:
            st.metric("Lines", structural.lines)
            st.metric("Average Syllables", f"{structural.avg_syllables:.1f}")
        
        st.markdown("---")
        
        # Free Verse Analysis (if applicable)
        if structural.is_free_verse and structural.modern_metrics:
            st.subheader("Modern Verse Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Enjambement Ratio", f"{structural.modern_metrics.enjambement_ratio:.1%}")
                st.metric("Line Variation", f"{structural.modern_metrics.line_length_variation:.2f}")
                st.metric("Prose Tendency", f"{structural.modern_metrics.prose_poetry_score:.1%}")
            
            with col2:
                st.metric("Visual Complexity", f"{structural.modern_metrics.visual_structure_score:.1%}")
                st.metric("Syntactic Parallelism", f"{structural.modern_metrics.syntactic_parallelism:.1%}")
                st.metric("Lexical Repetition", f"{structural.modern_metrics.lexical_repetition_score:.1%}")
            
            # Free verse assessment
            if "free_verse_analysis" in validation:
                fv_assessment = validation["free_verse_analysis"].get("assessment", "")
                if fv_assessment:
                    assessment_display = fv_assessment.replace('_', ' ').title()
                    st.info(f"**Free Verse Assessment:** {assessment_display}")
            
            st.markdown("---")
        
        # Classical Metrics (always shown)
        st.subheader("Classical Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Stanza Form:** {structural.stanza_structure}")
            st.write(f"**Rhyme Pattern:** {structural.rhyme_pattern}")
            if structural.aruz_analysis.pattern_match:
                st.write(f"**Meter Pattern:** `{structural.aruz_analysis.pattern_match}`")
        
        with col2:
            st.write(f"**Total Syllables:** {sum(structural.syllables_per_line)}")
            st.write(f"**Syllables per Line:** {', '.join(map(str, structural.syllables_per_line))}")
            if structural.aruz_analysis.variations_detected:
                st.write(f"**Meter Variations:** {', '.join(structural.aruz_analysis.variations_detected)}")
        
        st.markdown("---")
        
        # Rhyme Analysis with Radīf Detection
        st.subheader("Rhyme Analysis (Qāfiyeh/Radīf)")
        
        # Check for global Radīf
        radif_values = [r.radif for r in structural.rhyme_scheme if r.radif]
        if radif_values and len(set(radif_values)) == 1:
            global_radif = radif_values[0]
            st.success(f"🔁 **Global Radīf Detected:** `{global_radif}` (appears in {len(radif_values)}/{len(structural.rhyme_scheme)} lines)")
            st.info("Meter analysis was performed on lines with Radīf removed for accuracy.")
        
        if structural.rhyme_scheme:
            for i, rhyme in enumerate(structural.rhyme_scheme[:5]):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Line {i+1}**")
                with col2:
                    st.write(f"Qāfiyeh: `{rhyme.qafiyeh}`")
                with col3:
                    st.write(f"Radīf: `{rhyme.radif or '—'}`")
                with col4:
                    st.write(f"Type: {rhyme.rhyme_type}")
            
            if len(structural.rhyme_scheme) > 5:
                st.caption(f"... and {len(structural.rhyme_scheme) - 5} more lines")
        
        st.markdown("---")
        
        # Content Analysis
        st.subheader("Content Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Words", content.total_words)
            st.metric("Unique Words", content.unique_words)
        
        with col2:
            st.metric("Lexical Diversity", f"{content.lexical_diversity:.1%}")
            st.metric("Stylistic Register", content.stylistic_register.title())
        
        with col3:
            st.metric("Neologisms", len(content.neologisms))
            st.metric("Archaisms", len(content.archaisms))
        
        # Word Frequencies
        if content.word_frequencies:
            st.write("**Top 10 Words:**")
            top_words = ", ".join([f"{w}({c})" for w, c in content.word_frequencies[:10]])
            st.write(top_words)
        
        # Themes
        active_themes = [k for k, v in content.theme_distribution.items() if v > 0]
        if active_themes:
            st.write(f"**Themes:** {', '.join(active_themes)}")
        
        st.markdown("---")
        
        # Quality Validation
        st.subheader("Quality Validation")
        
        quality_score = validation.get('quality_score', 0)
        reliability = validation.get('reliability', 'unknown')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Quality Score", f"{quality_score:.0%}")
        with col2:
            st.metric("Reliability", reliability.title())
        
        warnings = validation.get('warnings', [])
        if warnings:
            st.warning("**Warnings:**")
            for w in warnings:
                st.write(f"{w}")
        
        recommendations = validation.get('recommendations', [])
        if recommendations:
            st.info("**Recommendations:**")
            for r in recommendations:
                st.write(f"{r}")


# -------------------------------------------------------------------
# Library Management Functions
# -------------------------------------------------------------------
def collect_volume_metadata():
    """Collect metadata for a poetry volume"""
    st.markdown("---")
    st.header("📖 Volume Metadata Collection")
    
    # Use session state to preserve values
    if 'author_name' not in st.session_state:
        st.session_state.author_name = ""
    if 'volume_title' not in st.session_state:
        st.session_state.volume_title = ""
    if 'publication_year' not in st.session_state:
        st.session_state.publication_year = 2023
    
    col1, col2 = st.columns(2)
    
    with col1:
        author_name = st.text_input("Author / Full Name*", 
                                   value=st.session_state.author_name,
                                   key="author_name_input")
        st.session_state.author_name = author_name
        
        author_birth = st.number_input("Birth Year (optional)", 
                                      min_value=1000, 
                                      max_value=2024, 
                                      value=None,
                                      step=1,
                                      help="e.g., 1945",
                                      key="author_birth_input")
        
        author_death = st.number_input("Death Year (optional)", 
                                      min_value=1000, 
                                      max_value=2024, 
                                      value=None,
                                      step=1,
                                      help="e.g., 2020",
                                      key="author_death_input")
        
        volume_title = st.text_input("Title of Poetry Volume*", 
                                    value=st.session_state.volume_title,
                                    key="volume_title_input")
        st.session_state.volume_title = volume_title
        
        publication_year = st.number_input("Publication Year*", 
                                         min_value=1800, 
                                         max_value=2024, 
                                         value=st.session_state.publication_year,
                                         step=1,
                                         key="publication_year_input")
        st.session_state.publication_year = publication_year
    
    with col2:
        publisher = st.text_input("Publisher (optional)", key="publisher_input")
        city = st.text_input("Place of Publication (optional)", key="city_input")
        isbn = st.text_input("ISBN (optional)", key="isbn_input")
        pages = st.number_input("Number of Pages (optional)", 
                               min_value=1, 
                               max_value=1000, 
                               value=None,
                               step=1,
                               key="pages_input")
        
        # Genres (multiple selection)
        if LIBRARY_MANAGER_AVAILABLE:
            if 'selected_genres' not in st.session_state:
                st.session_state.selected_genres = []
            
            genre_options = [g.value for g in Genre]
            selected_genres = st.multiselect(
                "Literary Genres",
                options=genre_options,
                default=st.session_state.selected_genres,
                key="genres_input"
            )
            st.session_state.selected_genres = selected_genres
            genres = [Genre(g) for g in selected_genres]
        else:
            genres = []
            st.info("Genre selection requires extended library manager")
    
    # Period inference
    st.markdown("### Historical Period")
    
    if LIBRARY_MANAGER_AVAILABLE:
        period_options = {p.value: p for p in Period}
        default_period = None
        
        # Auto-infer period
        if publication_year:
            if publication_year < 1920:
                default_period = Period.CLASSICAL.value
            elif 1920 <= publication_year < 1940:
                default_period = Period.SOVIET_EARLY.value
            elif 1940 <= publication_year < 1970:
                default_period = Period.SOVIET_MID.value
            elif 1970 <= publication_year < 1991:
                default_period = Period.SOVIET_LATE.value
            elif 1991 <= publication_year < 2000:
                default_period = Period.INDEPENDENCE.value
            else:
                default_period = Period.CONTEMPORARY.value
        
        default_index = 0
        if default_period:
            period_keys = list(period_options.keys())
            default_index = period_keys.index(default_period) if default_period in period_keys else 0
        
        # Store selected period in session state
        if 'selected_period_value' not in st.session_state:
            st.session_state.selected_period_value = list(period_options.keys())[default_index]
        
        selected_period_value = st.selectbox(
            "Historical Period*",
            options=list(period_options.keys()),
            index=list(period_options.keys()).index(st.session_state.selected_period_value),
            key="period_input"
        )
        st.session_state.selected_period_value = selected_period_value
        selected_period = period_options[selected_period_value]
    else:
        selected_period = None
        st.info("Period classification requires extended library manager")
    
    # Source type
    if 'source_type' not in st.session_state:
        st.session_state.source_type = "printed"
    
    source_type = st.radio(
        "Source Type",
        options=["printed", "manuscript", "digital"],
        index=["printed", "manuscript", "digital"].index(st.session_state.source_type),
        horizontal=True,
        key="source_type_input"
    )
    st.session_state.source_type = source_type
    
    # Additional notes
    if 'notes' not in st.session_state:
        st.session_state.notes = ""
    
    notes = st.text_area("Additional Notes (optional)", 
                        value=st.session_state.notes,
                        height=100,
                        key="notes_input")
    st.session_state.notes = notes
    
    # Validate required fields
    if not author_name or not volume_title:
        st.warning("Please fill in the required fields marked with *")
        return None
    
    if LIBRARY_MANAGER_AVAILABLE:
        # Create metadata object
        metadata = VolumeMetadata(
            author_name=author_name,
            author_birth_year=int(author_birth) if author_birth else None,
            author_death_year=int(author_death) if author_death else None,
            volume_title=volume_title,
            publication_year=int(publication_year),
            publisher=publisher if publisher else None,
            city=city if city else None,
            genres=genres,
            period=selected_period,
            isbn=isbn if isbn else None,
            pages=int(pages) if pages else None,
            source_type=source_type,
            notes=notes if notes else None
        )
        
        # Preview
        with st.expander("Metadata Preview"):
            st.json(metadata.to_dict())
        
        return metadata
    else:
        # Basic metadata structure if extended manager not available
        return {
            "author_name": author_name,
            "author_birth_year": int(author_birth) if author_birth else None,
            "author_death_year": int(author_death) if author_death else None,
            "volume_title": volume_title,
            "publication_year": int(publication_year),
            "publisher": publisher if publisher else None,
            "city": city if city else None,
            "isbn": isbn if isbn else None,
            "pages": int(pages) if pages else None,
            "source_type": source_type,
            "notes": notes if notes else None
        }
        def display_library_management(all_results: List[Dict[str, Any]]):
    """Display library management section"""
    st.markdown("---")
    st.header("📚 Library & Corpus Management")
    
    # Initialize session state for metadata
    if 'metadata_collected' not in st.session_state:
        st.session_state.metadata_collected = False
    if 'volume_metadata' not in st.session_state:
        st.session_state.volume_metadata = None
    if 'library_submitted' not in st.session_state:
        st.session_state.library_submitted = False
    
    # Tab selection for different management options
    tab1, tab2, tab3 = st.tabs(["📖 Add to Library", "📊 Library Statistics", "📤 Export & Share"])
    
    with tab1:
        # Only show form if metadata not already collected
        if not st.session_state.metadata_collected:
            # Volume metadata collection
            metadata = collect_volume_metadata()
            
            if metadata:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("➕ Add Volume to Library", type="primary", key="add_to_lib_btn"):
                        # Save metadata to session state
                        st.session_state.volume_metadata = metadata
                        st.session_state.metadata_collected = True
                        # Don't rerun here - let the next part handle it
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", key="cancel_lib_btn"):
                        st.session_state.metadata_collected = False
                        st.session_state.volume_metadata = None
                        st.rerun()
        
        # Process the saved metadata
        elif st.session_state.metadata_collected and st.session_state.volume_metadata and not st.session_state.library_submitted:
            metadata = st.session_state.volume_metadata
            
            st.success("✅ Volume metadata collected! Ready to add to library.")
            
            # Show metadata summary
            with st.expander("📋 Metadata Summary", expanded=True):
                if LIBRARY_MANAGER_AVAILABLE:
                    st.json(metadata.to_dict())
                else:
                    st.json(metadata)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm and Add to Library", type="primary", key="confirm_lib_btn"):
                    successful_results = [r for r in all_results if r.get('success', False)]
                    
                    if not successful_results:
                        st.error("No successful analyses to add to library.")
                        st.session_state.library_submitted = False
                    else:
                        if LIBRARY_MANAGER_AVAILABLE:
                            try:
                                # Prepare poems data
                                poems_data = []
                                for result in successful_results:
                                    # Convert analysis to dict if it's an object
                                    analysis_data = result['analysis']
                                    if hasattr(analysis_data, '__dict__'):
                                        analysis_dict = asdict(analysis_data)
                                    else:
                                        analysis_dict = analysis_data
                                    
                                    poems_data.append({
                                        'content': result['poem_text'],
                                        'analysis': analysis_dict,
                                        'poem_num': result['poem_num']
                                    })
                                
                                # Initialize library manager
                                library_manager = TajikLibraryManager()
                                
                                # Register volume
                                with st.spinner("Registering volume in library..."):
                                    volume_id = library_manager.register_volume(metadata, poems_data)
                                    
                                    # Generate timeline report
                                    html_report = library_manager.generate_timeline_report("html")
                                    
                                    # Save report
                                    report_path = Path(library_manager.library_path) / "timeline" / f"report_{volume_id}.html"
                                    report_path.parent.mkdir(exist_ok=True)
                                    with open(report_path, 'w', encoding='utf-8') as f:
                                        f.write(html_report)
                                    
                                    st.session_state.library_submitted = True
                                    st.session_state.volume_id = volume_id
                                    st.session_state.html_report = html_report
                                    st.session_state.library_manager = library_manager
                                    
                                    st.success(f"✅ Volume successfully registered: {volume_id}")
                                    st.rerun()
                            
                            except Exception as e:
                                st.error(f"Error registering volume: {e}")
                                logger.error(f"Library registration error: {e}")
                                st.session_state.library_submitted = False
                        else:
                            # Fallback to basic corpus manager
                            if CORPUS_MANAGER_AVAILABLE:
                                # Reset metadata state since we're switching to basic mode
                                st.session_state.metadata_collected = False
                                st.session_state.volume_metadata = None
                                st.rerun()
                            else:
                                st.warning("Neither library manager nor basic corpus manager is available.")
                                st.session_state.library_submitted = False
            
            with col2:
                if st.button("🔄 Edit Metadata", key="edit_metadata_btn"):
                    st.session_state.metadata_collected = False
                    st.session_state.volume_metadata = None
                    st.rerun()
        
        # Show success message after submission
        elif st.session_state.library_submitted:
            st.success("✅ Volume successfully added to library!")
            
            # Show quick statistics
            if hasattr(st.session_state, 'library_manager'):
                try:
                    corpus = st.session_state.library_manager.load_corpus()
                    stats = corpus["statistics"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Volumes", stats["total_volumes"])
                    with col2:
                        st.metric("Total Poems", stats["total_poems"])
                    with col3:
                        earliest = stats["publication_years"]["min"] or "—"
                        st.metric("Earliest Volume", earliest)
                    with col4:
                        latest = stats["publication_years"]["max"] or "—"
                        st.metric("Latest Volume", latest)
                    
                    # Show timeline visualization
                    if hasattr(st.session_state, 'html_report'):
                        st.markdown("### Timeline Visualization")
                        st.components.v1.html(st.session_state.html_report, height=600, scrolling=True)
                    
                    # Export options
                    st.markdown("### 📥 Download Options")
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        if st.button("📊 Download Timeline Report"):
                            html_report = st.session_state.html_report
                            st.download_button(
                                label="Download HTML",
                                data=html_report,
                                file_name=f"timeline_{st.session_state.volume_id}.html",
                                mime="text/html"
                            )
                    
                    with col_dl2:
                        if st.button("📁 Export Library Data"):
                            corpus = st.session_state.library_manager.load_corpus()
                            json_data = json.dumps(corpus, ensure_ascii=False, indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"tajik_library_{datetime.now().strftime('%Y%m%d')}.json",
                                mime="application/json"
                            )
                    
                except Exception as e:
                    st.error(f"Error displaying results: {e}")
            
            # Option to add another volume
            if st.button("➕ Add Another Volume"):
                # Reset all library-related session state
                for key in ['metadata_collected', 'volume_metadata', 'library_submitted', 
                           'volume_id', 'html_report', 'library_manager']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    with tab2:
        # Library statistics (unchanged)
        if LIBRARY_MANAGER_AVAILABLE:
            try:
                library_manager = TajikLibraryManager()
                corpus = library_manager.load_corpus()
                stats = corpus["statistics"]
                
                st.subheader("📊 Library Statistics")
                
                # Overall metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Volumes", stats["total_volumes"])
                    st.metric("Poems", stats["total_poems"])
                with col2:
                    st.metric("Lines", stats.get("total_lines", 0))
                    st.metric("Words", stats.get("total_words", 0))
                with col3:
                    st.metric("Unique Words", stats.get("unique_words", 0))
                    st.metric("Authors", stats.get("authors_count", 0))
                
                # Timeline distribution
                st.subheader("Timeline Distribution")
                
                # By period
                if "period_distribution" in stats and stats["period_distribution"]:
                    st.write("**By Historical Period:**")
                    for period, count in sorted(stats["period_distribution"].items()):
                        st.write(f"- {period}: {count} volumes")
                
                # By year
                if "publication_years" in stats and stats["publication_years"].get("distribution"):
                    st.write("**Publication Year Range:**")
                    years = stats["publication_years"]
                    st.write(f"- From: {years.get('min', '—')}")
                    st.write(f"- To: {years.get('max', '—')}")
                    st.write(f"- Years covered: {len(years.get('distribution', {}))}")
                
                # Meter distribution
                if "meter_distribution" in stats and stats["meter_distribution"]:
                    st.subheader("Meter Distribution")
                    for meter, count in sorted(stats["meter_distribution"].items(), key=lambda x: -x[1])[:10]:
                        st.write(f"- {meter}: {count}")
                
                # Theme distribution
                if "theme_distribution" in stats and stats["theme_distribution"]:
                    st.subheader("Theme Distribution")
                    for theme, count in sorted(stats["theme_distribution"].items(), key=lambda x: -x[1])[:5]:
                        st.write(f"- {theme}: {count}")
                        
            except Exception as e:
                st.error(f"Error loading library statistics: {e}")
        else:
            st.info("Extended library statistics require the extended library manager.")
    
    with tab3:
        # Export options (unchanged)
        st.subheader("Export Options")
        
        if LIBRARY_MANAGER_AVAILABLE:
            try:
                library_manager = TajikLibraryManager()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📁 Export Complete Library (JSON)"):
                        corpus = library_manager.load_corpus()
                        json_data = json.dumps(corpus, ensure_ascii=False, indent=2)
                        
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"tajik_poetry_library_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )
                
                with col2:
                    if st.button("📊 Generate Timeline Report"):
                        html_report = library_manager.generate_timeline_report("html")
                        
                        st.download_button(
                            label="Download HTML Report",
                            data=html_report,
                            file_name=f"timeline_report_{datetime.now().strftime('%Y%m%d')}.html",
                            mime="text/html"
                        )
                
                # Git export
                st.subheader("Git Export for Collaboration")
                if st.button("🔗 Prepare Git Export"):
                    try:
                        export_path = library_manager.export_contributions_for_git()
                        st.success(f"Export prepared: `{export_path}`")
                        
                        # Show Git commands
                        git_commands = f"""
# Git Commands for Sharing:
1. Navigate to your repository:
   cd /path/to/your/tajik-poetry-repo

2. Create new branch:
   git checkout -b new-contributions-{datetime.now().strftime('%Y%m%d')}

3. Copy exported files:
   cp "{export_path}" ./contributions/

4. Commit changes:
   git add contributions/
   git commit -m "New Tajik poetry contributions from {datetime.now().strftime('%Y-%m-%d')}"

5. Push to GitHub:
   git push origin new-contributions-{datetime.now().strftime('%Y%m%d')}
"""
                        
                        st.code(git_commands, language="bash")
                    except Exception as e:
                        st.error(f"Export preparation failed: {e}")
                        
            except Exception as e:
                st.error(f"Export error: {e}")
        else:
            st.info("Extended export features require the extended library manager.")


def display_basic_corpus_section(all_results: List[Dict[str, Any]]):
    """Display basic corpus management section (fallback)"""
    st.markdown("---")
    st.header("📚 Basic Corpus Management")
    
    if not CORPUS_MANAGER_AVAILABLE:
        st.warning("Basic Corpus Manager is not available.")
        return
    
    # Initialize Corpus Manager
    corpus_manager = TajikCorpusManager()
    
    st.markdown("### Contribute to the Tajik Poetry Corpus")
    
    # User information (optional)
    with st.expander("User Information (optional)"):
        username = st.text_input("GitHub Username (optional)", key="corpus_username")
        email = st.text_input("Email (optional)", key="corpus_email")
        license_accepted = st.checkbox(
            "I accept the CC-BY-NC-SA 4.0 license for my contribution",
            value=True,
            key="corpus_license"
        )
    
    # Prepare contributions
    contributions = []
    for result in all_results:
        if result.get('success', False):
            # Create analysis_result in required format
            first_line = result['poem_text'].split('\n')[0].strip()
            title = first_line[:50] if len(first_line) > 50 else first_line
            
            analysis_result = {
                "poem_id": f"P{result['poem_num']:03d}",
                "title": title,
                "content": result['poem_text'],
                "analysis": result['analysis'],
                "validation": result.get('validation', result['analysis'].quality_metrics if hasattr(result['analysis'], 'quality_metrics') else {})
            }
            
            # User info
            user_info = {}
            if username:
                user_info["username"] = username
            if email:
                user_info["email"] = email
            
            try:
                # Prepare contribution
                contribution = corpus_manager.prepare_contribution(
                    analysis_result=analysis_result,
                    raw_text=result['poem_text'],
                    user_info=user_info if user_info else {"anonymous": True}
                )
                
                # Accept license
                if license_accepted:
                    contribution["metadata"]["license_accepted"] = True
                
                contributions.append(contribution)
                
            except Exception as e:
                logger.error(f"Error preparing contribution: {e}")
    
    if not contributions:
        st.info("No analyzed poems available for corpus contribution.")
        return
    
    st.success(f"✅ {len(contributions)} poem(s) prepared for corpus contribution.")
    
    # Options for user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Contributions Locally", type="primary"):
            with st.spinner("Saving contributions..."):
                saved_count = 0
                for contribution in contributions:
                    try:
                        corpus_manager.save_contribution(contribution)
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"Error saving contribution: {e}")
                st.success(f"{saved_count} contribution(s) saved!")
    
    with col2:
        if st.button("📤 Export for Git"):
            with st.spinner("Exporting..."):
                try:
                    # First save all contributions
                    for contribution in contributions:
                        corpus_manager.save_contribution(contribution)
                    
                    export_path = corpus_manager.export_contributions_for_git()
                    st.success(f"Export prepared: `{export_path}`")
                    
                    # Show Git commands
                    git_commands = corpus_manager.generate_git_commands()
                    with st.expander("Show Git Commands"):
                        st.code(git_commands, language="bash")
                except Exception as e:
                    st.error(f"Export failed: {e}")
    
    with col3:
        if st.button("📈 Show Statistics"):
            try:
                # Load and display basic statistics
                corpus_path = Path("./tajik_corpus/corpus/master.json")
                if corpus_path.exists():
                    with open(corpus_path, 'r', encoding='utf-8') as f:
                        corpus_data = json.load(f)
                    
                    stats = corpus_data.get("statistics", {})
                    
                    st.metric("Total Poems", stats.get("total_poems", 0))
                    st.metric("Total Lines", stats.get("total_lines", 0))
                    st.metric("Total Words", stats.get("total_words", 0))
                else:
                    st.info("No corpus data found yet.")
                    
            except Exception as e:
                st.error(f"Error loading statistics: {e}")


# -------------------------------------------------------------------
# Main Application
# -------------------------------------------------------------------
def main():
    if not ANALYZER_AVAILABLE:
        st.error("Error: Analyzer not available. Please ensure analyzer.py is in the same directory.")
        st.stop()
    
    # Initialize session state
    if 'splitters' not in st.session_state:
        st.session_state.splitters = []
    if 'all_lines' not in st.session_state:
        st.session_state.all_lines = []
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'proceed_to_analysis' not in st.session_state:
        st.session_state.proceed_to_analysis = False
    if 'final_poems' not in st.session_state:
        st.session_state.final_poems = []
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Enhanced"
    
    st.title("📖 Tajik Poetry Analyzer")
    st.markdown("Advanced scientific analysis of Tajik poetry with classical and modern approaches")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Analysis Mode")
        
        analysis_mode = st.radio(
            "Select analysis mode:",
            options=["Classical (ʿArūḍ only)", "Enhanced (with free verse detection)"],
            index=1,
            key="analysis_mode_selector"
        )
        
        st.session_state.analysis_mode = analysis_mode
        
        st.markdown("---")
        st.header("Features")
        
        if analysis_mode == "Classical (ʿArūḍ only)":
            st.write("• 16 Classical ʿArūḍ Meters")
            st.write("• Qāfiyeh/Radīf Detection")
            st.write("• Prosodic Weight Calculation")
            st.write("• Classical Form Recognition")
            st.write("• Scientific Validation")
        else:
            st.write("• Free Verse Detection")
            st.write("• Modern Verse Metrics")
            st.write("• Enjambement Analysis")
            st.write("• Line Variation Analysis")
            st.write("• Prose-Poetry Assessment")
            st.write("• All Classical Features")
        
        st.markdown("---")
        st.header("Library Features")
        
        if LIBRARY_MANAGER_AVAILABLE:
            st.write("• Volume Metadata Collection")
            st.write("• Chronological Analysis")
            st.write("• Historical Periods")
            st.write("• Genre Classification")
            st.write("• Timeline Visualization")
        elif CORPUS_MANAGER_AVAILABLE:
            st.write("• Basic Corpus Contribution")
            st.write("• Git Export")
            st.write("• Local Storage")
        else:
            st.write("• No library features available")
        
        st.markdown("---")
        st.header("Supported Classical Meters")
        meters = ["ṭawīl", "basīṭ", "wāfir", "kāmil", "mutaqārib", "hazaj", 
                  "rajaz", "ramal", "sarīʿ", "munsarih", "khafīf", "muḍāriʿ",
                  "muqtaḍab", "mujtath", "mutadārik", "madīd"]
        st.write(", ".join(meters))

    # Main area
    st.header("📄 Upload File")

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT file",
        type=['pdf', 'txt'],
        help="Supports normal and scanned PDFs"
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        try:
            with st.spinner("Extracting text from file..."):
                text = read_file_with_pdf_support(tmp_path)
                st.session_state.extracted_text = text
                st.success(f"✅ Text extracted: {len(text)} characters")

            with st.expander("Show extracted text"):
                st.text_area("Content", text, height=200)

            # Poem splitting section
            if not st.session_state.proceed_to_analysis:
                st.header("✂️ Poem Splitting")
                
                split_mode = st.radio(
                    "How do you want to split the poems?",
                    options=["Automatic", "Manual with preview"],
                    index=0
                )

                if split_mode == "Manual with preview":
                    if not st.session_state.all_lines or st.session_state.all_lines[0] != text.split('\n')[0]:
                        splitter = UIPoemSplitter()
                        all_lines = text.split('\n')
                        
                        proposed = splitter.get_split_suggestions(text)
                        if not proposed:
                            proposed = [i for i, line in enumerate(all_lines) if line.strip() == '']
                        if not proposed and len(all_lines) > 10:
                            proposed = list(range(10, len(all_lines), 20))
                        
                        st.session_state.splitters = proposed
                        st.session_state.all_lines = all_lines
                    
                    col_left, col_right = st.columns([3, 1])
                    
                    with col_left:
                        st.subheader("Text with split suggestions")
                        display_text = ""
                        for i, line in enumerate(st.session_state.all_lines):
                            if i in st.session_state.splitters:
                                display_text += f"\n--- SPLITTER (before line {i+1}) ---\n"
                            display_text += line + "\n"
                        st.text_area("Preview", display_text, height=400)
                    
                    with col_right:
                        st.subheader("Control Splitters")
                        
                        selected_position = st.slider(
                            "Line index",
                            0,
                            len(st.session_state.all_lines)-1,
                            value=0 if not st.session_state.splitters else min(st.session_state.splitters)
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if selected_position in st.session_state.splitters:
                                if st.button("❌ Remove"):
                                    st.session_state.splitters.remove(selected_position)
                                    st.rerun()
                            else:
                                if st.button("➕ Add"):
                                    st.session_state.splitters.append(selected_position)
                                    st.session_state.splitters.sort()
                                    st.rerun()
                        
                        with col2:
                            if st.button("🗑️ Clear all"):
                                st.session_state.splitters = []
                                st.rerun()
                        
                        st.markdown(f"**Splitters:** {', '.join(map(str, sorted(st.session_state.splitters)))}")
                        
                        if st.button("✅ Confirm and Analyze", type="primary"):
                            poems = split_text_at_indices(text, st.session_state.splitters)
                            st.session_state.final_poems = poems
                            st.session_state.proceed_to_analysis = True
                            st.rerun()
                    
                    st.stop()
                
                else:  # Automatic
                    poems = split_poems_auto(text)
                    st.info(f"Found {len(poems)} poems")
                    
                    if st.button("✅ Confirm and Analyze", type="primary"):
                        st.session_state.final_poems = poems
                        st.session_state.proceed_to_analysis = True
                        st.rerun()
            
            # Analysis section
            if st.session_state.proceed_to_analysis:
                poems = st.session_state.final_poems
                
                if not poems:
                    st.warning("No poems to analyze.")
                    st.session_state.proceed_to_analysis = False
                    st.rerun()
                
                st.header("🔍 Analysis")
                st.info(f"Analyzing {len(poems)} poem(s) in {analysis_mode} mode...")
                
                if st.button("🚀 Start Analysis", type="primary"):
                    # Load appropriate analyzer
                    if analysis_mode == "Classical (ʿArūḍ only)":
                        analyzer = load_classical_analyzer()
                    else:
                        analyzer = load_enhanced_analyzer()
                    
                    progress_bar = st.progress(0)
                    results_container = st.container()
                    
                    all_results = []
                    
                    for i, poem_text in enumerate(poems):
                        progress_bar.progress((i + 1) / len(poems))
                        
                        try:
                            if analysis_mode == "Classical (ʿArūḍ only)":
                                analysis = analyzer.analyze_poem(poem_text)
                                all_results.append({
                                    'poem_text': poem_text,
                                    'poem_num': i+1,
                                    'analysis': analysis,
                                    'success': True,
                                    'mode': 'classical'
                                })
                            else:
                                analysis = analyzer.analyze_poem(poem_text)
                                all_results.append({
                                    'poem_text': poem_text,
                                    'poem_num': i+1,
                                    'analysis': analysis,
                                    'success': True,
                                    'mode': 'enhanced'
                                })
                        except Exception as e:
                            logger.error(f"Error in poem {i+1}: {e}")
                            all_results.append({
                                'poem_text': poem_text,
                                'poem_num': i+1,
                                'error': str(e),
                                'success': False,
                                'mode': analysis_mode
                            })
                    
                    progress_bar.empty()
                    
                    with results_container:
                        st.markdown("---")
                        st.header("📊 Results")
                        
                        col1, col2, col3 = st.columns(3)
                        successful = sum(1 for r in all_results if r['success'])
                        
                        with col1:
                            st.metric("Total Poems", len(all_results))
                        with col2:
                            st.metric("Successful Analyses", successful)
                        with col3:
                            st.metric("Failed Analyses", len(all_results) - successful)
                        
                        st.markdown("---")
                        
                        for result in all_results:
                            if not result['success']:
                                st.error(f"Poem {result['poem_num']}: {result['error']}")
                                continue
                            
                            if result['mode'] == 'classical':
                                display_classical_results(
                                    result['analysis'],
                                    result['poem_num'],
                                    result['poem_text']
                                )
                            else:
                                display_enhanced_results(
                                    result['analysis'],
                                    result['poem_num'],
                                    result['poem_text']
                                )
                    
                    st.success("✅ Analysis completed!")
                    
                    # Generate Excel Report
                    if successful > 0:
                        st.markdown("---")
                        st.subheader("📥 Download Report")
                        
                        try:
                            # Prepare data for ExcelReporter
                            excel_data = []
                            for result in all_results:
                                if result['success']:
                                    # Extract title from first line
                                    first_line = result['poem_text'].split('\n')[0].strip()
                                    title = first_line[:50] if len(first_line) > 50 else first_line
                                    
                                    # Get validation data
                                    if result['mode'] == 'classical':
                                        validation = QualityValidator.validate_analysis(result['analysis'])
                                    else:
                                        validation = result['analysis'].quality_metrics
                                    
                                    excel_data.append({
                                        'poem_id': f"P{result['poem_num']:03d}",
                                        'title': title,
                                        'content': result['poem_text'],
                                        'analysis': result['analysis'],
                                        'validation': validation
                                    })
                            
                            # Create Excel report
                            excel_reporter = ExcelReporter()
                            mode_suffix = "classical" if analysis_mode == "Classical (ʿArūḍ only)" else "enhanced"
                            excel_filename = f"tajik_poetry_{mode_suffix}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                            excel_path = Path(tempfile.gettempdir()) / excel_filename
                            excel_reporter.create_report(excel_data, str(excel_path))
                            
                            # Provide download button
                            with open(excel_path, 'rb') as f:
                                excel_bytes = f.read()
                            
                            st.download_button(
                                label="📊 Download Excel Report",
                                data=excel_bytes,
                                file_name=excel_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.info(f"Report contains {len(excel_data)} poems with detailed analysis.")
                            
                        except Exception as e:
                            logger.error(f"Error creating Excel report: {e}")
                            st.error(f"Could not create Excel report: {e}")
                    
                    # Library/Corpus Management Section
                    successful_results = [r for r in all_results if r.get('success', False)]
                    if successful_results:
                        if LIBRARY_MANAGER_AVAILABLE:
                            display_library_management(successful_results)
                        elif CORPUS_MANAGER_AVAILABLE:
                            display_basic_corpus_section(successful_results)
                        else:
                            st.info("Library/corpus features not available.")
                    
                    if st.button("🔄 Start over"):
                        st.session_state.splitters = []
                        st.session_state.all_lines = []
                        st.session_state.proceed_to_analysis = False
                        st.session_state.final_poems = []
                        st.rerun()

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    else:
        # Landing page when no file uploaded
        st.info("Please upload a PDF or TXT file to begin.")
        
        st.markdown("---")
        st.subheader("🎯 Analysis Capabilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔬 Classical Analysis:**")
            st.markdown("""
            - 16 Classical ʿArūḍ Meters
            - Qāfiyeh (rhyme) & Radīf (refrain) detection
            - Prosodic weight calculation (Heavy/Light syllables)
            - Phonetic transcription (IPA)
            - Scientific quality validation
            - PDF & OCR support for scanned documents
            """)
        
        with col2:
            st.markdown("**🚀 Enhanced Analysis:**")
            st.markdown("""
            - Free verse detection
            - Modern verse metrics
            - Enjambement analysis
            - Line variation assessment
            - Prose-poetry scoring
            - Visual structure analysis
            - All classical features included
            """)
        
        st.markdown("---")
        
        if LIBRARY_MANAGER_AVAILABLE or CORPUS_MANAGER_AVAILABLE:
            st.subheader("📚 Library & Corpus Features")
            
            if LIBRARY_MANAGER_AVAILABLE:
                st.markdown("""
                - **Volume metadata** (author, year, publisher, genres)
                - **Historical period classification** (6 periods)
                - **Timeline visualization** across decades
                - **Genre analysis** and distribution
                - **Stylistic evolution tracking**
                - **Export to JSON/HTML/Git**
                """)
            elif CORPUS_MANAGER_AVAILABLE:
                st.markdown("""
                - **Basic corpus contribution**
                - **Local storage of analyses**
                - **Git export functionality**
                - **Collaborative research support**
                """)
        
        st.markdown("---")
        st.subheader("🎓 Research Applications")
        st.markdown("""
        - **Literary Studies**: Analysis of classical and modern Tajik poetry
        - **Linguistics**: Phonetic and prosodic analysis of Tajik language
        - **Digital Humanities**: Computational analysis of poetic structures
        - **Comparative Literature**: Comparison of Persianate poetic traditions
        - **Text Analysis**: Statistical analysis of poetic content and themes
        - **Historical Research**: Tracking stylistic evolution over time
        """)
        
        # Quick start instructions
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            1. **Prepare your text**: Use AI-assisted transcription for best results
            2. **Upload**: PDF or TXT file with Tajik Cyrillic text
            3. **Split poems**: Automatic or manual poem separation
            4. **Analyze**: Choose classical or enhanced analysis
            5. **Export**: Download Excel reports or add to library
            
            **For best results**: Ensure proper Tajik characters (ӣ, ӯ, ҷ, ҳ, қ, ғ)
            """)


if __name__ == "__main__":
    main()
