#!/usr/bin/env python3
"""
Simple Web-UI for Tajik Poetry Analyzer
Supports PDF upload and analysis with ʿArūḍ Analysis (16 classical meters)

Uses consolidated analyzer.py with:
- 16 Classical ʿArūḍ meters
- Qāfiyeh/Radīf detection
- Prosodic weight calculation
- Scientific quality validation
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

# Import from consolidated analyzer
try:
    from analyzer import (
        TajikPoemAnalyzer,
        AnalysisConfig,
        PoemData,
        AruzMeterAnalyzer,
        AdvancedRhymeAnalyzer,
        MeterConfidence,
        StructuralAnalysis,
        EnhancedPoemSplitter,
        QualityValidator
    )
    ANALYZER_AVAILABLE = True
    logger.info("Analyzer loaded successfully")
except ImportError as e:
    logger.error(f"Analyzer not available: {e}")
    ANALYZER_AVAILABLE = False

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
def load_analyzer():
    """Initialize analyzer (cached)"""
    config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
    return TajikPoemAnalyzer(config=config)


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


def display_results(analysis, poem_num: int, poem_text: str):
    """Display analysis results"""
    structural = analysis.structural
    validation = analysis.quality_metrics
    
    with st.expander(f"📜 Poem {poem_num} - {len(poem_text.split())} words", expanded=True):
        # Content
        st.subheader("Content")
        st.text(poem_text[:500] + "..." if len(poem_text) > 500 else poem_text)
        
        st.markdown("---")
        
        # ʿArūḍ Meter Analysis
        st.subheader("🎯 ʿArūḍ Meter Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            meter_name = structural.aruz_analysis.identified_meter
            st.metric("Identified Meter", meter_name.title())
        
        with col2:
            confidence = structural.meter_confidence.value
            confidence_color = {
                'high': '🟢', 'medium': '🟡', 'low': '🟠', 'none': '🔴'
            }.get(confidence, '⚪')
            st.metric("Confidence", f"{confidence_color} {confidence.title()}")
        
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
        st.subheader("📊 Structural Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Lines:** {structural.lines}")
            st.write(f"**Avg Syllables/Line:** {structural.avg_syllables:.1f}")
            st.write(f"**Stanza Form:** {structural.stanza_structure}")
        
        with col2:
            st.write(f"**Rhyme Pattern:** {structural.rhyme_pattern}")
        
        st.markdown("---")
        
        # Rhyme Analysis
        st.subheader("🎵 Rhyme Analysis (Qāfiyeh/Radīf)")
        
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
        st.subheader("✅ Quality Validation")
        
        quality_score = validation.get('quality_score', 0)
        reliability = validation.get('reliability', 'unknown')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Quality Score", f"{quality_score:.0%}")
        with col2:
            reliability_color = {'high': '🟢', 'medium': '🟡', 'low': '🟠'}.get(reliability, '⚪')
            st.metric("Reliability", f"{reliability_color} {reliability.title()}")
        
        warnings = validation.get('warnings', [])
        if warnings:
            st.warning("**Warnings:**")
            for w in warnings:
                st.write(f"⚠️ {w}")
        
        recommendations = validation.get('recommendations', [])
        if recommendations:
            st.info("**Recommendations:**")
            for r in recommendations:
                st.write(f"💡 {r}")


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
    
    st.title("Tajik Poetry Analyzer")
    st.markdown('<span class="analyzer-badge">🚀 ʿArūḍ Analysis with 16 Classical Meters</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Features")
        st.write("✅ 16 Classical ʿArūḍ Meters")
        st.write("✅ Qāfiyeh/Radīf Detection")
        st.write("✅ Phonetic Transcription")
        st.write("✅ Prosodic Consistency")
        st.write("✅ Scientific Validation")
        st.write("✅ PDF & OCR support")
        
        st.markdown("---")
        st.header("📚 Supported Meters")
        meters = ["ṭawīl", "basīṭ", "wāfir", "kāmil", "mutaqārib", "hazaj", 
                  "rajaz", "ramal", "sarīʿ", "munsarih", "khafīf", "muḍāriʿ",
                  "muqtaḍab", "mujtath", "mutadārik", "madīd"]
        st.write(", ".join(meters))

    # Main area
    st.header("📁 Upload File")

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
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

            with st.expander("📄 Show extracted text"):
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
                                display_text += f"\n--- **SPLITTER** (before line {i+1}) ---\n"
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
                                if st.button("Remove"):
                                    st.session_state.splitters.remove(selected_position)
                                    st.rerun()
                            else:
                                if st.button("Add"):
                                    st.session_state.splitters.append(selected_position)
                                    st.session_state.splitters.sort()
                                    st.rerun()
                        
                        with col2:
                            if st.button("Clear all"):
                                st.session_state.splitters = []
                                st.rerun()
                        
                        st.markdown(f"**Splitters:** {', '.join(map(str, sorted(st.session_state.splitters)))}")
                        
                        if st.button("🚀 Confirm & Analyze", type="primary"):
                            poems = split_text_at_indices(text, st.session_state.splitters)
                            st.session_state.final_poems = poems
                            st.session_state.proceed_to_analysis = True
                            st.rerun()
                    
                    st.stop()
                
                else:  # Automatic
                    poems = split_poems_auto(text)
                    st.info(f"📊 Found {len(poems)} poems")
                    
                    if st.button("✅ Confirm & Analyze", type="primary"):
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
                
                st.header("🔬 Analysis")
                st.info(f"Analyzing {len(poems)} poem(s)...")
                
                if st.button("▶️ Start Analysis", type="primary"):
                    analyzer = load_analyzer()
                    
                    progress_bar = st.progress(0)
                    results_container = st.container()
                    
                    all_results = []
                    
                    for i, poem_text in enumerate(poems):
                        progress_bar.progress((i + 1) / len(poems))
                        
                        try:
                            analysis = analyzer.analyze_poem(poem_text)
                            all_results.append({
                                'poem_text': poem_text,
                                'poem_num': i+1,
                                'analysis': analysis,
                                'success': True
                            })
                        except Exception as e:
                            logger.error(f"Error in poem {i+1}: {e}")
                            all_results.append({
                                'poem_text': poem_text,
                                'poem_num': i+1,
                                'error': str(e),
                                'success': False
                            })
                    
                    progress_bar.empty()
                    
                    with results_container:
                        st.markdown("---")
                        st.header("📈 Results")
                        
                        col1, col2, col3 = st.columns(3)
                        successful = sum(1 for r in all_results if r['success'])
                        
                        with col1:
                            st.metric("Total", len(all_results))
                        with col2:
                            st.metric("Successful", successful)
                        with col3:
                            st.metric("Failed", len(all_results) - successful)
                        
                        st.markdown("---")
                        
                        for result in all_results:
                            if not result['success']:
                                st.error(f"❌ Poem {result['poem_num']}: {result['error']}")
                                continue
                            
                            display_results(
                                result['analysis'],
                                result['poem_num'],
                                result['poem_text']
                            )
                    
                    st.success("✅ Analysis completed!")
                    
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
        st.info("👆 Please upload a PDF or TXT file to begin.")
        
        st.markdown("---")
        st.subheader("🎯 What this analyzer can do:")
        
        st.markdown("""
        - **16 Classical ʿArūḍ Meters** (ṭawīl, basīṭ, wāfir, kāmil, etc.)
        - **Qāfiyeh (rhyme) & Radīf (refrain) detection**
        - **Prosodic weight calculation** (Heavy/Light syllables)
        - **Phonetic transcription** (IPA)
        - **Scientific quality validation**
        - **PDF & OCR support** for scanned documents
        """)


if __name__ == "__main__":
    main()
