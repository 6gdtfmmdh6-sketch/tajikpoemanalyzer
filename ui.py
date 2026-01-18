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

FIXED: Proper session_state handling for buttons and downloads
"""

import streamlit as st
from pathlib import Path
import tempfile
import re
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from consolidated analyzer
try:
    from analyzer import (
        TajikPoemAnalyzer,
        EnhancedTajikPoemAnalyzer,
        AnalysisConfig,
        PoemData,
        AruzMeterAnalyzer,
        AdvancedRhymeAnalyzer,
        MeterConfidence,
        StructuralAnalysis,
        EnhancedStructuralAnalysis,
        EnhancedComprehensiveAnalysis,
        ModernVerseMetrics,
        EnhancedPoemSplitter,
        QualityValidator,
        ExcelReporter,
        RadifAnalysis,
        EnhancedRadifDetector
    )
    ANALYZER_AVAILABLE = True
    logger.info("Analyzer loaded successfully")
except ImportError as e:
    logger.error(f"Analyzer not available: {e}")
    ANALYZER_AVAILABLE = False

# Import Corpus Manager
try:
    from corpus_manager import TajikCorpusManager
    CORPUS_MANAGER_AVAILABLE = True
    logger.info("Corpus Manager loaded successfully")
except ImportError as e:
    logger.error(f"Corpus Manager not available: {e}")
    CORPUS_MANAGER_AVAILABLE = False

try:
    from pdf_handler import read_file_with_pdf_support
except ImportError:
    st.error("Error: Could not import pdf_handler.")
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
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'splitters': [],
        'all_lines': [],
        'extracted_text': "",
        'proceed_to_analysis': False,
        'final_poems': [],
        'analysis_mode': "Enhanced",
        # NEW: Results storage for persistent display
        'analysis_results': None,
        'excel_bytes': None,
        'excel_filename': None,
        'analysis_completed': False,
        # Corpus state
        'corpus_saved': False,
        'corpus_exported': False,
        'corpus_stats': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


class UIPoemSplitter:
    """Poem splitter for UI"""
    
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
# Display Functions
# -------------------------------------------------------------------
def display_classical_results(analysis, poem_num: int, poem_text: str):
    """Display classical analysis results"""
    structural = analysis.structural
    content = analysis.content
    validation = analysis.quality_metrics
    
    with st.expander(f"Poem {poem_num} - {content.total_words} words", expanded=False):
        st.text(poem_text[:500] + "..." if len(poem_text) > 500 else poem_text)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Meter", structural.aruz_analysis.identified_meter.title())
        with col2:
            st.metric("Confidence", structural.meter_confidence.value.title())
        with col3:
            st.metric("Lines", structural.lines)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Avg Syllables/Line:** {structural.avg_syllables:.1f}")
            st.write(f"**Rhyme Pattern:** {structural.rhyme_pattern}")
        with col2:
            st.write(f"**Lexical Diversity:** {content.lexical_diversity:.1%}")
            st.write(f"**Neologisms:** {len(content.neologisms)}")
        
        if content.word_frequencies:
            st.write("**Top Words:** " + ", ".join([f"{w}({c})" for w, c in content.word_frequencies[:5]]))


def display_enhanced_results(analysis: EnhancedComprehensiveAnalysis, poem_num: int, poem_text: str):
    """Display enhanced analysis results"""
    structural = analysis.structural
    content = analysis.content
    
    badge = "Free Verse" if structural.is_free_verse else "Classical"
    
    with st.expander(f"Poem {poem_num} - {badge} - {content.total_words} words", expanded=False):
        st.text(poem_text[:500] + "..." if len(poem_text) > 500 else poem_text)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if structural.is_free_verse:
                st.metric("Form", "Free Verse")
            else:
                st.metric("Meter", structural.aruz_analysis.identified_meter.title())
        with col2:
            st.metric("Confidence", structural.meter_confidence.value.title())
        with col3:
            st.metric("Lines", structural.lines)
        
        if structural.is_free_verse and structural.modern_metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Enjambement:** {structural.modern_metrics.enjambement_ratio:.1%}")
                st.write(f"**Line Variation:** {structural.modern_metrics.line_length_variation:.2f}")
            with col2:
                st.write(f"**Prose Score:** {structural.modern_metrics.prose_poetry_score:.1%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Avg Syllables/Line:** {structural.avg_syllables:.1f}")
            st.write(f"**Rhyme Pattern:** {structural.rhyme_pattern}")
        with col2:
            st.write(f"**Lexical Diversity:** {content.lexical_diversity:.1%}")
            st.write(f"**Neologisms:** {len(content.neologisms)}")


def run_analysis(poems: List[str], analysis_mode: str) -> List[Dict]:
    """Run analysis on poems and return results"""
    if analysis_mode == "Classical (ʿArūḍ only)":
        analyzer = load_classical_analyzer()
    else:
        analyzer = load_enhanced_analyzer()
    
    all_results = []
    
    for i, poem_text in enumerate(poems):
        try:
            analysis = analyzer.analyze_poem(poem_text)
            mode = 'classical' if analysis_mode == "Classical (ʿArūḍ only)" else 'enhanced'
            all_results.append({
                'poem_text': poem_text,
                'poem_num': i+1,
                'analysis': analysis,
                'success': True,
                'mode': mode
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
    
    return all_results


def generate_excel_report(all_results: List[Dict], analysis_mode: str) -> tuple:
    """Generate Excel report and return (bytes, filename)"""
    excel_data = []
    for result in all_results:
        if result['success']:
            first_line = result['poem_text'].split('\n')[0].strip()
            title = first_line[:50] if len(first_line) > 50 else first_line
            
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
    
    excel_reporter = ExcelReporter()
    mode_suffix = "classical" if analysis_mode == "Classical (ʿArūḍ only)" else "enhanced"
    excel_filename = f"tajik_poetry_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_path = Path(tempfile.gettempdir()) / excel_filename
    excel_reporter.create_report(excel_data, str(excel_path))
    
    with open(excel_path, 'rb') as f:
        excel_bytes = f.read()
    
    return excel_bytes, excel_filename


# -------------------------------------------------------------------
# Corpus Section
# -------------------------------------------------------------------
def display_corpus_section():
    """Display Corpus Management section"""
    if not st.session_state.analysis_results:
        return
    
    st.markdown("---")
    st.header("Corpus Management")
    
    if not CORPUS_MANAGER_AVAILABLE:
        st.warning("Corpus Manager not available.")
        return
    
    successful_results = [r for r in st.session_state.analysis_results if r.get('success', False)]
    if not successful_results:
        return
    
    st.info(f"{len(successful_results)} poem(s) available for corpus contribution.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Save to Corpus", key="corpus_save"):
            corpus_manager = TajikCorpusManager()
            saved = 0
            for result in successful_results:
                try:
                    first_line = result['poem_text'].split('\n')[0].strip()
                    title = first_line[:50] if len(first_line) > 50 else first_line
                    
                    contribution = corpus_manager.prepare_contribution(
                        analysis_result={
                            "poem_id": f"P{result['poem_num']:03d}",
                            "title": title,
                            "content": result['poem_text'],
                            "analysis": result['analysis'],
                            "validation": result['analysis'].quality_metrics
                        },
                        raw_text=result['poem_text'],
                        user_info={"anonymous": True}
                    )
                    corpus_manager.save_contribution(contribution)
                    saved += 1
                except Exception as e:
                    logger.error(f"Error saving: {e}")
            st.session_state.corpus_saved = True
            st.rerun()
    
    with col2:
        if st.button("Export for Git", key="corpus_export"):
            corpus_manager = TajikCorpusManager()
            try:
                export_path = corpus_manager.export_contributions_for_git()
                st.session_state.corpus_exported = True
                st.session_state.export_path = str(export_path)
                st.rerun()
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col3:
        if st.button("Show Statistics", key="corpus_stats"):
            corpus_manager = TajikCorpusManager()
            try:
                st.session_state.corpus_stats = corpus_manager.get_corpus_statistics()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display persistent state
    if st.session_state.corpus_saved:
        st.success("Contributions saved!")
    
    if st.session_state.corpus_exported:
        st.success(f"Export ready: {st.session_state.get('export_path', '')}")
    
    if st.session_state.corpus_stats:
        stats = st.session_state.corpus_stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Poems", stats.get("total_poems", 0))
        with col2:
            st.metric("Words", stats.get("total_words", 0))
        with col3:
            st.metric("Unique Words", stats.get("unique_words", 0))


# -------------------------------------------------------------------
# Main Application
# -------------------------------------------------------------------
def main():
    if not ANALYZER_AVAILABLE:
        st.error("Analyzer not available. Ensure analyzer.py exists.")
        st.stop()
    
    init_session_state()
    
    st.title("Tajik Poetry Analyzer")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Analysis Mode")
        analysis_mode = st.radio(
            "Select mode:",
            options=["Classical (ʿArūḍ only)", "Enhanced (with free verse detection)"],
            index=1
        )
        st.session_state.analysis_mode = analysis_mode
        
        st.markdown("---")
        st.header("Meters")
        meters = ["ṭawīl", "basīṭ", "wāfir", "kāmil", "mutaqārib", "hazaj", 
                  "rajaz", "ramal", "sarīʿ", "munsarih", "khafīf", "muḍāriʿ",
                  "muqtaḍab", "mujtath", "mutadārik", "madīd"]
        st.caption(", ".join(meters))

    # File upload
    st.header("Upload File")
    uploaded_file = st.file_uploader("PDF or TXT", type=['pdf', 'txt'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        try:
            # Extract text
            with st.spinner("Extracting text..."):
                text = read_file_with_pdf_support(tmp_path)
                st.session_state.extracted_text = text
                st.success(f"Extracted: {len(text)} characters")

            with st.expander("Show text"):
                st.text_area("Content", text, height=200)

            # --- POEM SPLITTING ---
            if not st.session_state.proceed_to_analysis:
                st.header("Split Poems")
                
                split_mode = st.radio("Method:", ["Automatic", "Manual"], index=0)

                if split_mode == "Manual":
                    if not st.session_state.all_lines or st.session_state.all_lines[0] != text.split('\n')[0]:
                        splitter = UIPoemSplitter()
                        all_lines = text.split('\n')
                        proposed = splitter.get_split_suggestions(text)
                        if not proposed:
                            proposed = [i for i, line in enumerate(all_lines) if line.strip() == '']
                        st.session_state.splitters = proposed
                        st.session_state.all_lines = all_lines
                    
                    # Display with splitters
                    display_text = ""
                    for i, line in enumerate(st.session_state.all_lines):
                        if i in st.session_state.splitters:
                            display_text += f"\n--- SPLIT ---\n"
                        display_text += line + "\n"
                    st.text_area("Preview", display_text, height=300)
                    
                    # Controls
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pos = st.number_input("Line #", 0, len(st.session_state.all_lines)-1, 0)
                    with col2:
                        if st.button("Add split"):
                            if pos not in st.session_state.splitters:
                                st.session_state.splitters.append(pos)
                                st.session_state.splitters.sort()
                                st.rerun()
                    with col3:
                        if st.button("Remove split"):
                            if pos in st.session_state.splitters:
                                st.session_state.splitters.remove(pos)
                                st.rerun()
                    
                    st.write(f"Splits at: {st.session_state.splitters}")
                    
                    if st.button("Confirm & Continue", type="primary"):
                        poems = split_text_at_indices(text, st.session_state.splitters)
                        st.session_state.final_poems = poems
                        st.session_state.proceed_to_analysis = True
                        st.rerun()
                    
                else:  # Automatic
                    poems = split_poems_auto(text)
                    st.info(f"Found {len(poems)} poems")
                    
                    if st.button("Confirm & Continue", type="primary"):
                        st.session_state.final_poems = poems
                        st.session_state.proceed_to_analysis = True
                        st.rerun()

            # --- ANALYSIS SECTION ---
            if st.session_state.proceed_to_analysis:
                poems = st.session_state.final_poems
                
                if not poems:
                    st.warning("No poems found.")
                    st.session_state.proceed_to_analysis = False
                    st.rerun()
                
                st.header("Analysis")
                st.info(f"{len(poems)} poem(s) ready for analysis")
                
                # START ANALYSIS BUTTON
                if st.button("Start Analysis", type="primary"):
                    with st.spinner("Analyzing..."):
                        progress = st.progress(0)
                        
                        # Run analysis
                        results = run_analysis(poems, analysis_mode)
                        st.session_state.analysis_results = results
                        
                        # Generate Excel
                        successful = [r for r in results if r['success']]
                        if successful:
                            try:
                                excel_bytes, excel_filename = generate_excel_report(results, analysis_mode)
                                st.session_state.excel_bytes = excel_bytes
                                st.session_state.excel_filename = excel_filename
                            except Exception as e:
                                logger.error(f"Excel error: {e}")
                        
                        st.session_state.analysis_completed = True
                        progress.empty()
                    st.rerun()
                
                # --- DISPLAY RESULTS (outside button block!) ---
                if st.session_state.analysis_completed and st.session_state.analysis_results:
                    results = st.session_state.analysis_results
                    
                    st.markdown("---")
                    st.header("Results")
                    
                    # Summary
                    successful = sum(1 for r in results if r['success'])
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", len(results))
                    with col2:
                        st.metric("Successful", successful)
                    with col3:
                        st.metric("Failed", len(results) - successful)
                    
                    # Download button (OUTSIDE analysis button!)
                    if st.session_state.excel_bytes:
                        st.download_button(
                            label="Download Excel Report",
                            data=st.session_state.excel_bytes,
                            file_name=st.session_state.excel_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )
                    
                    st.markdown("---")
                    
                    # Individual results
                    for result in results:
                        if not result['success']:
                            st.error(f"Poem {result['poem_num']}: {result['error']}")
                            continue
                        
                        if result['mode'] == 'classical':
                            display_classical_results(result['analysis'], result['poem_num'], result['poem_text'])
                        else:
                            display_enhanced_results(result['analysis'], result['poem_num'], result['poem_text'])
                    
                    # Corpus section
                    display_corpus_section()
                    
                    # Reset button
                    st.markdown("---")
                    if st.button("Start Over"):
                        for key in ['splitters', 'all_lines', 'proceed_to_analysis', 'final_poems',
                                   'analysis_results', 'excel_bytes', 'excel_filename', 
                                   'analysis_completed', 'corpus_saved', 'corpus_exported', 'corpus_stats']:
                            st.session_state[key] = None if 'results' in key or 'bytes' in key or 'stats' in key else ([] if 'list' in str(type(st.session_state.get(key, []))) else False)
                        st.session_state.splitters = []
                        st.session_state.all_lines = []
                        st.session_state.proceed_to_analysis = False
                        st.session_state.final_poems = []
                        st.session_state.analysis_results = None
                        st.session_state.analysis_completed = False
                        st.rerun()

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    else:
        st.info("Upload a PDF or TXT file to begin.")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Classical Analysis:**")
            st.markdown("""
            - 16 ʿArūḍ Meters
            - Qāfiyeh/Radīf detection
            - Prosodic analysis
            - Quality validation
            """)
        
        with col2:
            st.markdown("**Enhanced Analysis:**")
            st.markdown("""
            - Free verse detection
            - Modern metrics
            - Enjambement analysis
            - All classical features
            """)


if __name__ == "__main__":
    main()
