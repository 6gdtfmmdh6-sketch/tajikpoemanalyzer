#!/usr/bin/env python3
"""
Simple Web-UI for Tajik Poetry Analyzer
Supports PDF upload and analysis with Enhanced ʿArūḍ Analysis

This UI integrates:
- Basic TajikPoemAnalyzer (app2.py)
- Enhanced analyzer with proper ʿArūḍ meter analysis (enhanced_tajik_analyzer.py)
"""
import streamlit as st
from pathlib import Path
import tempfile
import re
from typing import List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import analyzers - try enhanced first, fall back to basic
USE_ENHANCED = False
ENHANCED_AVAILABLE = False

try:
    from enhanced_tajik_analyzer import (
        EnhancedTajikPoemAnalyzer,
        AruzMeterAnalyzer,
        AdvancedRhymeAnalyzer,
        MeterConfidence,
        EnhancedStructuralAnalysis
    )
    from app2 import AnalysisConfig, TajikPoemAnalyzer
    ENHANCED_AVAILABLE = True
    USE_ENHANCED = True
    logger.info("Enhanced analyzer loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced analyzer not available: {e}")
    try:
        from app2 import TajikPoemAnalyzer, AnalysisConfig, PoemData
        logger.info("Basic analyzer loaded")
    except ImportError:
        st.error("Error: Could not import TajikPoemAnalyzer. Please ensure app2.py is in the same directory.")
        st.stop()

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

# CSS for simple design
st.markdown("""
<style>
    .main {max-width: 1200px; margin: 0 auto;}
    h1 {text-align: center; color: #2c3e50;}
    .stButton>button {width: 100%;}
    .metric-box {
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .enhanced-badge {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    .basic-badge {
        background-color: #6c757d;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Configuration and Helper Classes
# -------------------------------------------------------------------
class TajikCyrillicConfig(AnalysisConfig):
    """Configuration specific to Tajik Cyrillic poetry"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tajik_cyrillic_alphabet = set(
            'АБВГҒДЕЁЖЗИӢЙКҚЛМНОПРСТУӮФХҲЧҶШЪЭЮЯ'
            'абвгғдеёжзиӣйкқлмнопрстуӯфхҳчҷшъэюя'
            '0123456789'
            ' .,!?;:-–—()[]{}"\'«»'
        )
        self.min_poem_lines = 3
        self.max_poem_lines = 100
        self.title_max_length = 100


class EnhancedPoemSplitter:
    """Advanced poem splitter for Tajik Cyrillic poetry collections"""
    
    def __init__(self, config: TajikCyrillicConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EnhancedPoemSplitter")
        
    def get_split_suggestions(self, text: str) -> List[int]:
        """Returns line indices where a new poem is likely to start."""
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
            
            if i > 0 and not lines[i-1].strip() and line.strip() and line.strip()[0].isupper():
                score += 0.5
            
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
        """Simple heuristic to recognize title lines."""
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
def load_analyzer(use_enhanced: bool = True):
    """Initialize analyzer (cached)"""
    config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
    
    if use_enhanced and ENHANCED_AVAILABLE:
        try:
            return EnhancedTajikPoemAnalyzer(config=config), True
        except Exception as e:
            logger.warning(f"Failed to load enhanced analyzer: {e}")
    
    return TajikPoemAnalyzer(config=config), False


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


def display_enhanced_results(result: Dict[str, Any], poem_num: int, poem_text: str):
    """Display results from enhanced analyzer"""
    structural = result.get('structural_analysis')
    validation = result.get('validation', {})
    
    with st.expander(f"📜 Poem {poem_num} - {len(poem_text.split())} words", expanded=True):
        # Content
        st.subheader("Content")
        st.text(poem_text[:500] + "..." if len(poem_text) > 500 else poem_text)
        
        st.markdown("---")
        
        # ʿArūḍ Meter Analysis (Enhanced Feature)
        st.subheader("🎯 ʿArūḍ Meter Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            meter_name = structural.aruz_analysis.identified_meter if hasattr(structural, 'aruz_analysis') else "unknown"
            st.metric("Identified Meter", meter_name.title())
        
        with col2:
            confidence = structural.meter_confidence.value if hasattr(structural, 'meter_confidence') else "unknown"
            confidence_color = {
                'high': '🟢', 'medium': '🟡', 'low': '🟠', 'none': '🔴'
            }.get(confidence, '⚪')
            st.metric("Confidence", f"{confidence_color} {confidence.title()}")
        
        with col3:
            prosodic = structural.prosodic_consistency if hasattr(structural, 'prosodic_consistency') else 0
            st.metric("Prosodic Consistency", f"{prosodic:.1%}")
        
        # Pattern display
        if hasattr(structural, 'aruz_analysis') and structural.aruz_analysis.pattern_match:
            st.write(f"**Pattern:** `{structural.aruz_analysis.pattern_match}`")
            if structural.aruz_analysis.variations_detected:
                st.write(f"**Variations:** {', '.join(structural.aruz_analysis.variations_detected)}")
        
        st.markdown("---")
        
        # Structural Analysis
        st.subheader("📊 Structural Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Lines:** {structural.lines}")
            st.write(f"**Avg Syllables/Line:** {structural.avg_syllables:.1f}")
            if hasattr(structural, 'syllable_analysis'):
                syl_info = structural.syllable_analysis
                if 'heavy_syllables' in syl_info:
                    st.write(f"**Heavy Syllables:** {syl_info.get('heavy_syllables', 0)}")
                    st.write(f"**Light Syllables:** {syl_info.get('light_syllables', 0)}")
        
        with col2:
            st.write(f"**Rhyme Pattern:** {structural.rhyme_pattern}")
        
        st.markdown("---")
        
        # Rhyme Analysis (Enhanced Feature)
        st.subheader("🎵 Rhyme Analysis (Qāfiyeh/Radīf)")
        
        if hasattr(structural, 'rhyme_scheme') and structural.rhyme_scheme:
            for i, rhyme in enumerate(structural.rhyme_scheme[:5]):  # Show first 5 lines
                with st.container():
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
        st.subheader("✅ Quality Validation")
        
        quality_score = validation.get('quality_score', 0)
        reliability = validation.get('reliability_level', 'unknown')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Quality Score", f"{quality_score:.1%}")
        with col2:
            reliability_color = {'high': '🟢', 'medium': '🟡', 'low': '🟠', 'unreliable': '🔴'}.get(reliability, '⚪')
            st.metric("Reliability", f"{reliability_color} {reliability.title()}")
        
        # Warnings
        warnings = validation.get('warnings', [])
        if warnings:
            st.warning("**Warnings:**")
            for w in warnings:
                st.write(f"⚠️ {w}")
        
        # Recommendations
        recommendations = validation.get('recommended_actions', [])
        if recommendations:
            st.info("**Recommendations:**")
            for r in recommendations:
                st.write(f"💡 {r}")


def display_basic_results(analysis, poem_num: int, poem_text: str):
    """Display results from basic analyzer"""
    with st.expander(f"📜 Poem {poem_num} - {len(poem_text.split())} words"):
        # Content
        st.subheader("Content")
        st.text(poem_text[:500] + "..." if len(poem_text) > 500 else poem_text)
        
        st.markdown("---")
        
        # Structural analysis
        st.subheader("Structural Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Lines:** {analysis.structural.lines}")
            st.write(f"**Syllables/Line:** {analysis.structural.avg_syllables:.1f}")
            st.write(f"**Stanza Form:** {analysis.structural.stanza_structure}")
        
        with col2:
            st.write(f"**Rhyme Pattern:** {analysis.structural.rhyme_pattern}")
        
        st.markdown("---")
        
        # Content analysis
        st.subheader("Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Most Frequent Words:**")
            if hasattr(analysis.content, 'word_frequencies'):
                for word, count in analysis.content.word_frequencies[:5]:
                    st.write(f"- {word}: {count}x")
        
        with col2:
            st.write("**Themes:**")
            if hasattr(analysis.content, 'theme_distribution'):
                themes = [k for k, v in analysis.content.theme_distribution.items() if v > 0]
                if themes:
                    for theme in themes:
                        st.write(f"- {theme}")
                else:
                    st.write("No themes recognized")


# -------------------------------------------------------------------
# Main Application
# -------------------------------------------------------------------
def main():
    # Initialize session state variables
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
    if 'use_enhanced' not in st.session_state:
        st.session_state.use_enhanced = ENHANCED_AVAILABLE
    
    st.title("Tajik Poetry Analyzer")
    
    # Show analyzer mode badge
    if st.session_state.use_enhanced and ENHANCED_AVAILABLE:
        st.markdown('<span class="enhanced-badge">🚀 Enhanced ʿArūḍ Mode</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="basic-badge">📊 Basic Mode</span>', unsafe_allow_html=True)
    
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Analyzer mode toggle
        if ENHANCED_AVAILABLE:
            st.session_state.use_enhanced = st.toggle(
                "Use Enhanced ʿArūḍ Analyzer",
                value=st.session_state.use_enhanced,
                help="Enable advanced ʿArūḍ meter analysis with 16 classical Arabic-Persian meters"
            )
        else:
            st.warning("Enhanced analyzer not available")
            st.caption("Make sure enhanced_tajik_analyzer.py is present")
        
        st.markdown("---")
        st.header("ℹ️ Info")
        st.write("Scientific analysis of Tajik/Persian poetry")
        
        st.markdown("---")
        st.write("**Features:**")
        
        if st.session_state.use_enhanced and ENHANCED_AVAILABLE:
            st.write("✅ 16 Classical ʿArūḍ Meters")
            st.write("✅ Qāfiyeh/Radīf Detection")
            st.write("✅ Phonetic Transcription")
            st.write("✅ Prosodic Consistency")
            st.write("✅ Scientific Validation")
            st.write("✅ PDF & OCR support")
        else:
            st.write("- Basic metric analysis")
            st.write("- Rhyme scheme detection")
            st.write("- Thematic analysis")
            st.write("- PDF & OCR support")

    # Main area
    st.header("📁 Upload File")

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=['pdf', 'txt'],
        help="Supports normal and scanned PDFs"
    )

    if uploaded_file is not None:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        try:
            # Extract text
            with st.spinner("Extracting text from file..."):
                text = read_file_with_pdf_support(tmp_path)
                st.session_state.extracted_text = text
                st.success(f"✅ Text extracted: {len(text)} characters")

            # Show text
            with st.expander("📄 Show extracted text"):
                st.text_area("Content", text, height=200, key="extracted_text_area")

            # -----------------------------------------------------------
            # Poem splitting section
            # -----------------------------------------------------------
            if not st.session_state.proceed_to_analysis:
                st.header("✂️ Poem Splitting")
                
                split_mode = st.radio(
                    "How do you want to split the poems?",
                    options=["Automatic (simple blank line search)", "Manual with preview and correction"],
                    index=0
                )

                if split_mode == "Manual with preview and correction":
                    if not st.session_state.all_lines or st.session_state.all_lines[0] != text.split('\n')[0]:
                        config = TajikCyrillicConfig()
                        splitter = EnhancedPoemSplitter(config)
                        all_lines = text.split('\n')
                        
                        proposed_split_indices = splitter.get_split_suggestions(text)
                        if not proposed_split_indices:
                            proposed_split_indices = [i for i, line in enumerate(all_lines) if line.strip() == '']
                        if not proposed_split_indices and len(all_lines) > 10:
                            proposed_split_indices = list(range(10, len(all_lines), 20))
                        
                        st.session_state.splitters = proposed_split_indices
                        st.session_state.all_lines = all_lines
                    
                    col_left, col_right = st.columns([3, 1])
                    
                    with col_left:
                        st.subheader("Text with split suggestions")
                        display_text = ""
                        for i, line in enumerate(st.session_state.all_lines):
                            if i in st.session_state.splitters:
                                display_text += f"\n--- **SPLITTER** (before line {i+1}) ---\n"
                            display_text += line + "\n"
                        st.text_area("Preview", display_text, height=400, key="display_area")
                    
                    with col_right:
                        st.subheader("Control Splitters")
                        
                        current_splitters = st.session_state.splitters
                        
                        selected_position = st.slider(
                            "Line index for splitter",
                            0,
                            len(st.session_state.all_lines)-1,
                            value=0 if not current_splitters else min(current_splitters),
                            key="splitter_slider"
                        )
                        
                        col_add_remove, col_clear = st.columns(2)
                        with col_add_remove:
                            if selected_position in current_splitters:
                                if st.button("Remove splitter"):
                                    st.session_state.splitters.remove(selected_position)
                                    st.rerun()
                            else:
                                if st.button("Add splitter"):
                                    st.session_state.splitters.append(selected_position)
                                    st.session_state.splitters.sort()
                                    st.rerun()
                        
                        with col_clear:
                            if st.button("Clear all"):
                                st.session_state.splitters = []
                                st.rerun()
                        
                        st.markdown("---")
                        st.markdown(f"**Current splitters at lines:** {', '.join(map(str, sorted(st.session_state.splitters)))}")
                        
                        # Confirm and proceed to analysis
                        if st.button("🚀 Confirm splitting & start analysis", type="primary"):
                            poems = split_text_at_indices(text, st.session_state.splitters)
                            st.session_state.final_poems = poems
                            st.session_state.proceed_to_analysis = True
                            st.rerun()
                    
                    st.stop()
                
                else:  # Automatic mode
                    poems = split_poems_auto(text)
                    st.info(f"📊 Automatically split: {len(poems)} poems")
                    
                    if st.button("✅ Confirm and proceed to analysis", type="primary"):
                        st.session_state.final_poems = poems
                        st.session_state.proceed_to_analysis = True
                        st.rerun()
            
            # -----------------------------------------------------------
            # Analysis section
            # -----------------------------------------------------------
            if st.session_state.proceed_to_analysis:
                poems = st.session_state.final_poems
                
                if not poems:
                    st.warning("No poems to analyze. Please adjust split points.")
                    st.session_state.proceed_to_analysis = False
                    st.rerun()
                
                st.header("🔬 Analysis Ready")
                st.info(f"Found {len(poems)} poem(s) for analysis")
                
                # Show which analyzer will be used
                if st.session_state.use_enhanced and ENHANCED_AVAILABLE:
                    st.success("🚀 Using Enhanced ʿArūḍ Analyzer with 16 classical meters")
                else:
                    st.info("📊 Using Basic Analyzer")
                
                if st.button("▶️ Start Analysis", type="primary"):
                    analyzer, is_enhanced = load_analyzer(st.session_state.use_enhanced)
                    
                    # Progress Bar
                    progress_bar = st.progress(0)
                    results_container = st.container()
                    
                    all_results = []
                    
                    for i, poem_text in enumerate(poems):
                        progress_bar.progress((i + 1) / len(poems))
                        
                        try:
                            if is_enhanced:
                                # Use enhanced analyzer
                                analysis = analyzer.analyze_poem_enhanced(poem_text)
                                all_results.append({
                                    'poem_text': poem_text,
                                    'poem_num': i+1,
                                    'analysis': analysis,
                                    'success': True,
                                    'enhanced': True
                                })
                            else:
                                # Use basic analyzer
                                analysis = analyzer.analyze_poem(poem_text)
                                all_results.append({
                                    'poem_text': poem_text,
                                    'poem_num': i+1,
                                    'analysis': analysis,
                                    'success': True,
                                    'enhanced': False
                                })
                        except Exception as e:
                            logger.error(f"Error in poem {i+1}: {e}")
                            all_results.append({
                                'poem_text': poem_text,
                                'poem_num': i+1,
                                'error': str(e),
                                'success': False,
                                'enhanced': is_enhanced
                            })
                    
                    progress_bar.empty()
                    
                    # Display results
                    with results_container:
                        st.markdown("---")
                        st.header("📈 Analysis Results")
                        
                        # Overview
                        col1, col2, col3, col4 = st.columns(4)
                        successful = sum(1 for r in all_results if r['success'])
                        
                        with col1:
                            st.metric("Total Poems", len(all_results))
                        with col2:
                            st.metric("Successful", successful)
                        with col3:
                            st.metric("Failed", len(all_results) - successful)
                        with col4:
                            mode = "Enhanced" if is_enhanced else "Basic"
                            st.metric("Mode", mode)
                        
                        st.markdown("---")
                        
                        # Individual poems
                        for result in all_results:
                            if not result['success']:
                                st.error(f"❌ Poem {result['poem_num']}: {result['error']}")
                                continue

                            if result.get('enhanced', False):
                                display_enhanced_results(
                                    result['analysis'],
                                    result['poem_num'],
                                    result['poem_text']
                                )
                            else:
                                display_basic_results(
                                    result['analysis'],
                                    result['poem_num'],
                                    result['poem_text']
                                )
                    
                    st.success("✅ Analysis completed!")
                    
                    # Reset button
                    if st.button("🔄 Start over with new splitting"):
                        st.session_state.splitters = []
                        st.session_state.all_lines = []
                        st.session_state.proceed_to_analysis = False
                        st.session_state.final_poems = []
                        st.rerun()

        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()

    else:
        st.info("👆 Please upload a PDF or TXT file to begin.")
        
        # Show example of what the analyzer can do
        st.markdown("---")
        st.subheader("🎯 What this analyzer can do:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Basic Analysis:**
            - Syllable counting
            - Basic rhyme scheme detection
            - Word frequency analysis
            - Theme detection
            """)
        
        with col2:
            if ENHANCED_AVAILABLE:
                st.markdown("""
                **Enhanced ʿArūḍ Analysis:**
                - 16 Classical Arabic-Persian meters
                - Qāfiyeh (rhyme) & Radīf (refrain) detection
                - Prosodic weight calculation (Heavy/Light)
                - Phonetic transcription
                - Scientific quality validation
                """)
            else:
                st.warning("Enhanced analyzer not available. Add enhanced_tajik_analyzer.py for full ʿArūḍ analysis.")


if __name__ == "__main__":
    main()
