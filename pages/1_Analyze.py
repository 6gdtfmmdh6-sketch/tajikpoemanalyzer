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

# Note: page_config is set in main ui.py

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
        # Results storage
        'analysis_results': None,
        'excel_bytes': None,
        'excel_filename': None,
        'analysis_completed': False,
        # Corpus state
        'corpus_saved': False,
        'corpus_exported': False,
        'corpus_stats': None,
        # NEW: File info for batch creation
        'uploaded_filename': None,
        'batch_metadata': {},
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
    """Display enhanced analysis results with all details"""
    structural = analysis.structural
    content = analysis.content
    
    badge = "Free Verse" if structural.is_free_verse else "Classical"
    
    with st.expander(f"Poem {poem_num} - {badge} - {content.total_words} words", expanded=False):
        st.text(poem_text[:500] + "..." if len(poem_text) > 500 else poem_text)
        
        st.markdown("---")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if structural.is_free_verse:
                st.metric("Form", "Free Verse")
            else:
                st.metric("Meter", structural.aruz_analysis.identified_meter.title())
        with col2:
            st.metric("Confidence", structural.meter_confidence.value.title())
        with col3:
            st.metric("Lines", structural.lines)
        with col4:
            st.metric("Words", content.total_words)
        
        st.markdown("---")
        
        # Structural Analysis
        st.markdown("**Structural Analysis**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Avg Syllables/Line: {structural.avg_syllables:.1f}")
            st.write(f"Rhyme Pattern: {structural.rhyme_pattern}")
        with col2:
            st.write(f"Stanza Type: {structural.stanza_type if hasattr(structural, 'stanza_type') else 'N/A'}")
            # Radif info
            if hasattr(structural, 'radif_analysis') and structural.radif_analysis:
                radif = structural.radif_analysis
                if hasattr(radif, 'has_radif') and radif.has_radif:
                    st.write(f"Radif: {radif.radif_text if hasattr(radif, 'radif_text') else 'Yes'}")
                else:
                    st.write("Radif: None")
        with col3:
            # Show syllable pattern
            if hasattr(structural, 'syllables_per_line') and structural.syllables_per_line:
                pattern = structural.syllables_per_line[:10]
                st.write(f"Syllable Pattern: {pattern}{'...' if len(structural.syllables_per_line) > 10 else ''}")
        
        # Free Verse Metrics (if applicable)
        if structural.is_free_verse and structural.modern_metrics:
            st.markdown("---")
            st.markdown("**Free Verse Metrics**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Enjambement Ratio: {structural.modern_metrics.enjambement_ratio:.1%}")
            with col2:
                st.write(f"Line Length Variation: {structural.modern_metrics.line_length_variation:.2f}")
            with col3:
                st.write(f"Prose Poetry Score: {structural.modern_metrics.prose_poetry_score:.1%}")
            
            # Free verse assessment if available
            if hasattr(analysis, 'quality_metrics') and analysis.quality_metrics:
                fv = getattr(analysis.quality_metrics, 'free_verse_assessment', None)
                if fv:
                    st.write(f"Assessment: {fv}")
        
        # Content Analysis
        st.markdown("---")
        st.markdown("**Content Analysis**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Lexical Diversity: {content.lexical_diversity:.1%}")
            st.write(f"Unique Words: {content.unique_words}")
        with col2:
            st.write(f"Neologisms Found: {len(content.neologisms)}")
            if content.neologisms:
                st.write(f"Examples: {', '.join(content.neologisms[:5])}")
        
        # Top words
        if content.word_frequencies:
            st.markdown("---")
            st.markdown("**Top Words**")
            top_words = content.word_frequencies[:10]
            word_str = ", ".join([f"{w} ({c})" for w, c in top_words])
            st.write(word_str)
        
        # Themes
        if hasattr(content, 'themes') and content.themes:
            st.write(f"Themes: {', '.join(content.themes[:5])}")
        
        # Quality metrics
        if hasattr(analysis, 'quality_metrics') and analysis.quality_metrics:
            qm = analysis.quality_metrics
            st.markdown("---")
            st.markdown("**Quality Metrics**")
            col1, col2 = st.columns(2)
            with col1:
                if hasattr(qm, 'overall_confidence'):
                    st.write(f"Overall Confidence: {qm.overall_confidence:.1%}")
                if hasattr(qm, 'syllable_consistency'):
                    st.write(f"Syllable Consistency: {qm.syllable_consistency:.1%}")
            with col2:
                if hasattr(qm, 'rhyme_quality'):
                    st.write(f"Rhyme Quality: {qm.rhyme_quality:.1%}")
                if hasattr(qm, 'warnings') and qm.warnings:
                    st.write(f"Warnings: {len(qm.warnings)}")
                    for w in qm.warnings[:3]:
                        st.caption(f"- {w}")


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


def generate_excel_report(all_results: List[Dict], analysis_mode: str, source_filename: str = None) -> tuple:
    """Generate Excel report, save to exports/, and return (bytes, filename)"""
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
    
    # Create descriptive filename from source
    if source_filename:
        base_name = Path(source_filename).stem
        base_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)[:30]
    else:
        base_name = "analysis"
    
    excel_filename = f"{base_name}_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Save to exports/ directory (persistent)
    exports_dir = Path("./exports")
    exports_dir.mkdir(exist_ok=True)
    excel_path = exports_dir / excel_filename
    excel_reporter.create_report(excel_data, str(excel_path))
    
    with open(excel_path, 'rb') as f:
        excel_bytes = f.read()
    
    logger.info(f"Excel report saved to {excel_path}")
    return excel_bytes, excel_filename


def get_previous_exports() -> List[Dict]:
    """Get list of previous Excel exports"""
    exports_dir = Path("./exports")
    if not exports_dir.exists():
        return []
    
    exports = []
    for f in sorted(exports_dir.glob("*.xlsx"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            stat = f.stat()
            exports.append({
                'filename': f.name,
                'path': str(f),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
        except Exception:
            pass
    return exports[:10]  # Return last 10


# -------------------------------------------------------------------
# Corpus Section
# -------------------------------------------------------------------
def generate_batch_id(filename: str) -> str:
    """Generate a batch ID from filename"""
    import re
    # Clean filename: remove extension, replace spaces/special chars
    base = Path(filename).stem
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', base.lower())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"batch_{clean}_{timestamp}"


def display_corpus_section():
    """Display Corpus Management section with metadata and batch support"""
    if not st.session_state.analysis_results:
        return
    
    st.markdown("---")
    st.header("Save to Library")
    
    if not CORPUS_MANAGER_AVAILABLE:
        st.warning("Corpus Manager not available.")
        return
    
    successful_results = [r for r in st.session_state.analysis_results if r.get('success', False)]
    if not successful_results:
        return
    
    # Already saved?
    if st.session_state.corpus_saved:
        st.success(f"Saved {len(successful_results)} poems to library!")
        st.info("Go to Library page to view and manage your poems.")
        
        # Show statistics
        if st.button("Show Corpus Statistics", key="btn_show_stats_after"):
            corpus_manager = TajikCorpusManager()
            stats = corpus_manager.get_corpus_statistics()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Poems", stats.get("total_poems", 0))
            with col2:
                st.metric("Total Words", stats.get("total_words", 0))
            with col3:
                st.metric("Unique Words", stats.get("unique_words", 0))
        return
    
    # Show what will be saved
    st.info(f"{len(successful_results)} poem(s) ready to save")
    
    # Source file info
    source_filename = st.session_state.get('uploaded_filename', 'unknown.txt')
    st.write(f"**Source file:** `{source_filename}`")
    
    # Metadata form
    st.markdown("### Volume Metadata (optional but recommended)")
    st.caption("This metadata will be applied to all poems from this file.")
    
    col1, col2 = st.columns(2)
    with col1:
        author = st.text_input("Author", key="corpus_author", placeholder="e.g. Dilorom Soliboeva")
        collection = st.text_input("Collection/Volume Title", key="corpus_collection", placeholder="e.g. Tufonhoi sokit")
    with col2:
        year = st.number_input("Publication Year", min_value=1900, max_value=2030, value=2000, key="corpus_year")
        publisher = st.text_input("Publisher", key="corpus_publisher", placeholder="e.g. Adib")
    
    st.markdown("---")
    
    # Save button
    if st.button("Save All to Library", type="primary", key="btn_corpus_save"):
        corpus_manager = TajikCorpusManager()
        saved = 0
        errors = []
        
        # Generate batch ID
        batch_id = generate_batch_id(source_filename)
        
        # Prepare metadata
        volume_metadata = {
            'author': author if author else None,
            'collection': collection if collection else None,
            'year': year if year else None,
            'publisher': publisher if publisher else None,
        }
        
        with st.spinner(f"Saving {len(successful_results)} poems..."):
            for result in successful_results:
                try:
                    first_line = result['poem_text'].split('\n')[0].strip()
                    title = first_line[:50] if len(first_line) > 50 else first_line
                    
                    # Prepare contribution with batch info
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
                    
                    # ADD batch info and metadata to contribution
                    contribution['source_filename'] = source_filename
                    contribution['upload_batch_id'] = batch_id
                    contribution['volume_metadata'] = volume_metadata
                    
                    # Update metadata section too
                    if 'metadata' in contribution:
                        contribution['metadata']['volume_title'] = collection
                        contribution['metadata']['volume_year'] = year
                    
                    # Save
                    corpus_manager.save_contribution(contribution)
                    saved += 1
                    
                except Exception as e:
                    logger.error(f"Error saving poem {result['poem_num']}: {e}")
                    errors.append(f"Poem {result['poem_num']}: {str(e)}")
        
        if saved > 0:
            st.session_state.corpus_saved = True
            st.session_state.batch_metadata = {
                'batch_id': batch_id,
                'source_filename': source_filename,
                'poems_saved': saved,
                'volume_metadata': volume_metadata
            }
            st.rerun()
        
        if errors:
            for err in errors:
                st.error(err)


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
        
        # Previous Exports Section
        st.markdown("---")
        st.header("Previous Exports")
        previous_exports = get_previous_exports()
        if previous_exports:
            for exp in previous_exports[:5]:
                with open(exp['path'], 'rb') as f:
                    st.download_button(
                        label=f"{exp['filename'][:25]}...",
                        data=f.read(),
                        file_name=exp['filename'],
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_{exp['filename']}"
                    )
        else:
            st.caption("No previous exports")

    # File upload
    st.header("Upload File")
    uploaded_file = st.file_uploader("PDF or TXT", type=['pdf', 'txt'])

    if uploaded_file is not None:
        # Store filename for batch creation
        st.session_state.uploaded_filename = uploaded_file.name
        
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
                                source_file = st.session_state.get('uploaded_filename', 'analysis')
                                excel_bytes, excel_filename = generate_excel_report(results, analysis_mode, source_file)
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
                        st.subheader("Download Report")
                        st.download_button(
                            label="Download Excel Report",
                            data=st.session_state.excel_bytes,
                            file_name=st.session_state.excel_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                            key="download_excel"
                        )
                    else:
                        st.warning("Excel report not available")
                    
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
                        # Reset all session state
                        st.session_state.splitters = []
                        st.session_state.all_lines = []
                        st.session_state.proceed_to_analysis = False
                        st.session_state.final_poems = []
                        st.session_state.analysis_results = None
                        st.session_state.excel_bytes = None
                        st.session_state.excel_filename = None
                        st.session_state.analysis_completed = False
                        st.session_state.corpus_saved = False
                        st.session_state.corpus_exported = False
                        st.session_state.corpus_stats = None
                        st.session_state.uploaded_filename = None
                        st.session_state.batch_metadata = {}
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


# Run main when page is loaded
main()
