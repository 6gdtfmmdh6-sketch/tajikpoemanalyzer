#!/usr/bin/env python3
"""
Simple Web-UI for Tajik Poetry Analyzer
Supports PDF upload and analysis
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil
import re
from typing import List, Optional
import logging

# Import original analyzer
try:
    from app2 import TajikPoemAnalyzer, AnalysisConfig, PoemData
except ImportError:
    st.error("Error: Could not import TajikPoemAnalyzer. Please ensure app2.py is in the same directory.")
    st.stop()

try:
    from pdf_handler import read_file_with_pdf_support
except ImportError:
    st.error("Error: Could not import pdf_handler. Please ensure pdf_handler.py is in the same directory.")
    st.stop()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Configuration and Helper Classes
# -------------------------------------------------------------------
class TajikCyrillicConfig(AnalysisConfig):
    """Configuration specific to Tajik Cyrillic poetry"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Tajik Cyrillic alphabet
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
        """
        Returns line indices where a new poem is likely to start.
        These suggestions are shown to the user for manual confirmation/correction.
        """
        lines = text.split('\n')
        suggestions = []
        
        for i, line in enumerate(lines):
            score = 0
            
            # 1. Title-like lines (strong signal)
            if self._looks_like_title(line):
                score += 2
            
            # 2. Empty line followed by a title-like line
            if i > 0 and not lines[i-1].strip() and len(line.strip()) > 0:
                score += 1.5
            
            # 3. Lines with poem markers like "***" or "---"
            if re.match(r'^[\*\-=]{3,}$', line.strip()):
                # Suggest split before this line
                suggestions.append(max(0, i-1))
                continue
                
            # 4. Line numbers (e.g., "1." or "(2)")
            if re.match(r'^\s*[\d]+[\.\)]\s*[A-ZА-Я]', line):
                score += 1
            
            # 5. Uppercase at the beginning of the line after an empty line
            if i > 0 and not lines[i-1].strip() and line.strip() and line.strip()[0].isupper():
                score += 0.5
            
            if score >= 1.5:  # Threshold
                suggestions.append(i)
        
        # Remove suggestions that are too close (within 3 lines)
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
        
        # Does not end with punctuation
        if line.endswith(('.', '!', '?', ':', ',')):
            return False
        
        # Starts with an uppercase letter
        if not line[0].isupper():
            return False
        
        # Not written entirely in uppercase (not a "SCREAM")
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
    # Split by ***** or multiple blank lines
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
    
    # Add last poem
    final_poem = '\n'.join(all_lines[start_idx:]).strip()
    if final_poem:
        poems.append(final_poem)
    
    return poems

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
    
    st.title("Tajik Poetry Analyzer")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Info")
        st.write("Scientific analysis of Tajik/Persian poetry")
        st.markdown("---")
        st.write("**Features:**")
        st.write("- Aruz metric analysis")
        st.write("- Rhyme scheme detection")
        st.write("- Phonetic transcription")
        st.write("- Thematic analysis")
        st.write("- PDF & OCR support")

    # Main area
    st.header("Upload File")

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
                st.success(f"Text extracted: {len(text)} characters")

            # Show text
            with st.expander("Show extracted text"):
                st.text_area("Content", text, height=200, key="extracted_text_area")

            # -----------------------------------------------------------
            # Poem splitting section - ONLY SHOW IF NOT PROCEEDING TO ANALYSIS
            # -----------------------------------------------------------
            if not st.session_state.proceed_to_analysis:
                st.header("📐 Poem Splitting")
                
                # Let the user choose the splitting mode
                split_mode = st.radio(
                    "How do you want to split the poems?",
                    options=["Automatic (simple blank line search)", "Manual with preview and correction"],
                    index=0
                )

                if split_mode == "Manual with preview and correction":
                    # Initialize the splitter and suggestions if needed
                    if not st.session_state.all_lines or st.session_state.all_lines[0] != text.split('\n')[0]:
                        config = TajikCyrillicConfig()
                        splitter = EnhancedPoemSplitter(config)
                        all_lines = text.split('\n')
                        
                        # Generate suggestions
                        proposed_split_indices = splitter.get_split_suggestions(text)
                        # Fallback: if no suggestions, use empty lines
                        if not proposed_split_indices:
                            proposed_split_indices = [i for i, line in enumerate(all_lines) if line.strip() == '']
                        # Fallback: if still none, use regular intervals
                        if not proposed_split_indices and len(all_lines) > 10:
                            proposed_split_indices = list(range(10, len(all_lines), 20))
                        
                        st.session_state.splitters = proposed_split_indices
                        st.session_state.all_lines = all_lines
                    
                    # Interactive display and editing
                    col_left, col_right = st.columns([3, 1])
                    
                    with col_left:
                        st.subheader("Text with split suggestions")
                        display_text = ""
                        for i, line in enumerate(st.session_state.all_lines):
                            # Add a marked splitter if this index is in the splitter list
                            if i in st.session_state.splitters:
                                display_text += f"\n--- 🟥 **SPLITTER** (before line {i+1}) ---\n"
                            display_text += line + "\n"
                        st.text_area("Preview", display_text, height=400, key="display_area")
                    
                    with col_right:
                        st.subheader("Control Splitters")
                        
                        # Select a splitter to edit or add a new one
                        current_splitters = st.session_state.splitters
                        
                        # Slider for moving or selecting a new position
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
                                if st.button("❌ Remove splitter"):
                                    st.session_state.splitters.remove(selected_position)
                                    st.rerun()
                            else:
                                if st.button("✅ Add splitter"):
                                    st.session_state.splitters.append(selected_position)
                                    st.session_state.splitters.sort()
                                    st.rerun()
                        
                        with col_clear:
                            if st.button("🗑️ Clear all"):
                                st.session_state.splitters = []
                                st.rerun()
                        
                        st.markdown("---")
                        st.markdown(f"**Current splitters at lines:** {', '.join(map(str, sorted(st.session_state.splitters)))}")
                        
                        # Confirm and proceed to analysis
                        if st.button("🚀 Confirm splitting & start analysis", type="primary"):
                            # Split text according to confirmed splitters
                            poems = split_text_at_indices(text, st.session_state.splitters)
                            
                            st.session_state.final_poems = poems
                            st.session_state.proceed_to_analysis = True
                            st.rerun()
                    
                    # If we're in manual mode, stop further execution
                    st.stop()
                
                else:  # Automatic mode
                    poems = split_poems_auto(text)
                    st.info(f"Automatically split: {len(poems)} poems")
                    
                    if st.button("Confirm and proceed to analysis", type="primary"):
                        st.session_state.final_poems = poems
                        st.session_state.proceed_to_analysis = True
                        st.rerun()
            
            # -----------------------------------------------------------
            # Analysis section (only reached if proceed_to_analysis is True)
            # -----------------------------------------------------------
            if st.session_state.proceed_to_analysis:
                poems = st.session_state.final_poems
                
                if not poems:
                    st.warning("No poems to analyze. Please adjust split points.")
                    st.session_state.proceed_to_analysis = False
                    st.rerun()
                
                st.header("Analysis Ready")
                st.info(f"Found {len(poems)} poem(s) for analysis")
                
                if st.button("Start Analysis", type="primary"):
                    analyzer = load_analyzer()
                    
                    # Progress Bar
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
                    
                    # Display results
                    with results_container:
                        st.markdown("---")
                        st.header("Analysis Results")
                        
                        # Overview
                        col1, col2, col3 = st.columns(3)
                        successful = sum(1 for r in all_results if r['success'])
                        
                        with col1:
                            st.metric("Total Poems", len(all_results))
                        with col2:
                            st.metric("Successful", successful)
                        with col3:
                            st.metric("Failed", len(all_results) - successful)
                        
                        st.markdown("---")
                        
                        # Individual poems
                        for result in all_results:
                            if not result['success']:
                                st.error(f"Poem {result['poem_num']}: {result['error']}")
                                continue

                            analysis = result['analysis']
                            poem_text = result['poem_text']
                            poem_num = result['poem_num']

                            with st.expander(f"Poem {poem_num} - {len(poem_text.split())} words"):
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
                                    if hasattr(analysis.structural, 'aruz_analysis'):
                                        st.write(f"**Meter:** {analysis.structural.aruz_analysis.identified_meter}")
                                        st.write(f"**Confidence:** {analysis.structural.aruz_analysis.confidence.value}")
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

                                if hasattr(analysis.content, 'neologisms') and analysis.content.neologisms:
                                    st.write(f"**Neologisms:** {', '.join(analysis.content.neologisms[:5])}")

                                if hasattr(analysis.content, 'archaisms') and analysis.content.archaisms:
                                    st.write(f"**Archaisms:** {', '.join(analysis.content.archaisms[:5])}")

                                st.markdown("---")

                                # Quality
                                if hasattr(analysis, 'quality_metrics'):
                                    st.subheader("Quality")
                                    quality_cols = st.columns(4)
                                    metrics = list(analysis.quality_metrics.items())[:4]
                                    for col, (metric, score) in zip(quality_cols, metrics):
                                        if isinstance(score, (int, float)):
                                            col.metric(metric.replace('_', ' ').title(), f"{score:.2f}")
                    
                    st.success("Analysis completed")
                    
                    # Add button to restart with new splitting
                    if st.button("Start over with new splitting"):
                        # Reset session state
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
        st.info("Please upload a PDF or TXT file to begin.")


if __name__ == "__main__":
    main()
