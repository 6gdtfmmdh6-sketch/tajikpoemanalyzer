#!/usr/bin/env python3
"""
Tajik Poetry Analyzer - Enhanced Streamlit UI
Fixed state management and navigation for better user experience
"""

import streamlit as st
from pathlib import Path
import tempfile
import re
import json
from datetime import datetime
from typing import List, Dict, Any
import logging

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

# Import corpus managers
try:
    from extended_corpus_manager import TajikLibraryManager
    EXTENDED_CORPUS_AVAILABLE = True
except ImportError:
    EXTENDED_CORPUS_AVAILABLE = False

try:
    from corpus_manager import TajikCorpusManager
    BASIC_CORPUS_AVAILABLE = True
except ImportError:
    BASIC_CORPUS_AVAILABLE = False

try:
    from pdf_handler import read_file_with_pdf_support
    PDF_HANDLER_AVAILABLE = True
except ImportError:
    PDF_HANDLER_AVAILABLE = False
    st.error("Error: Could not import pdf_handler. Please ensure pdf_handler.py is in the same directory.")

# -------------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_tab': 'upload',
        'extracted_text': '',
        'poems': [],
        'analysis_results': [],
        'analysis_mode': 'Enhanced',
        'corpus_type': 'both',
        'uploaded_file_name': '',
        'show_details': {},
        'corpus_metadata': {},
        'excel_data': None,
        'excel_filename': '',
        'library_initialized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def split_poems_auto(text: str) -> List[str]:
    """Split text into poems automatically"""
    if not text:
        return []
    
    # Try different separators
    separators = [
        r'\*{5,}',          # *****
        r'-{5,}',           # -----
        r'={5,}',           # =====
        r'_{5,}',           # _____
        r'\n\s*\n\s*\n+',   # Multiple blank lines
    ]
    
    pattern = '|'.join(separators)
    poems = [p.strip() for p in re.split(pattern, text)]
    
    # Filter out very short blocks
    return [p for p in poems if len(p) > 50]

def extract_title_from_poem(poem_text: str) -> str:
    """Extract title from poem text"""
    lines = poem_text.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        # Check if first line looks like a title
        if (len(first_line) < 100 and 
            not first_line.endswith(('.', '!', '?', ',', ':', ';')) and
            first_line[0].isupper()):
            return first_line[:50]
    return f"Poem {len(st.session_state.poems) + 1}"

def validate_poem_content(poem_text: str) -> bool:
    """Validate poem content before analysis"""
    if not poem_text or len(poem_text.strip()) < 50:
        return False
    
    # Count Tajik Cyrillic characters
    tajik_chars = set('АБВГҒДЕЁЖЗИӢЙКҚЛМНОПРСТУӮФХҲЧҶШЪЭЮЯабвгғдеёжзиӣйкқлмнопрстуӯфхҳчҷшъэюяӣӯ')
    found_chars = any(c in tajik_chars for c in poem_text)
    
    return found_chars

# -------------------------------------------------------------------
# Analysis Functions
# -------------------------------------------------------------------
def analyze_poem_enhanced(poem_text: str, poem_num: int) -> Dict[str, Any]:
    """Analyze poem using enhanced analyzer"""
    try:
        config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
        analyzer = EnhancedTajikPoemAnalyzer(config=config, enable_corpus=False)
        analysis = analyzer.analyze_poem(poem_text)
        
        return {
            'poem_text': poem_text,
            'poem_num': poem_num,
            'title': extract_title_from_poem(poem_text),
            'analysis': analysis,
            'mode': 'enhanced',
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing poem {poem_num}: {e}")
        return {
            'poem_text': poem_text,
            'poem_num': poem_num,
            'title': extract_title_from_poem(poem_text),
            'error': str(e),
            'mode': 'enhanced',
            'success': False,
            'timestamp': datetime.now().isoformat()
        }

def analyze_poem_classical(poem_text: str, poem_num: int) -> Dict[str, Any]:
    """Analyze poem using classical analyzer"""
    try:
        config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
        analyzer = TajikPoemAnalyzer(config=config)
        analysis = analyzer.analyze_poem(poem_text)
        
        return {
            'poem_text': poem_text,
            'poem_num': poem_num,
            'title': extract_title_from_poem(poem_text),
            'analysis': analysis,
            'mode': 'classical',
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing poem {poem_num}: {e}")
        return {
            'poem_text': poem_text,
            'poem_num': poem_num,
            'title': extract_title_from_poem(poem_text),
            'error': str(e),
            'mode': 'classical',
            'success': False,
            'timestamp': datetime.now().isoformat()
        }

# -------------------------------------------------------------------
# Display Functions
# -------------------------------------------------------------------
def display_poem_summary(result: Dict[str, Any]) -> None:
    """Display summary of poem analysis"""
    if not result['success']:
        st.error(f"Poem {result['poem_num']}: Analysis failed - {result['error']}")
        return
    
    analysis = result['analysis']
    title = result['title']
    
    # Create columns for summary
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.write(f"**{title}**")
        st.write(f"Lines: {analysis.structural.lines}")
        st.write(f"Words: {analysis.content.total_words}")
    
    with col2:
        meter = analysis.structural.aruz_analysis.identified_meter
        st.write(f"Meter: {meter}")
        st.write(f"Rhyme: {analysis.structural.rhyme_pattern}")
    
    with col3:
        if hasattr(analysis.structural, 'is_free_verse') and analysis.structural.is_free_verse:
            st.markdown("<span style='color: #ff6b6b;'>Free Verse</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #4a6fa5;'>Classical</span>", unsafe_allow_html=True)
        
        quality = analysis.quality_metrics.get('quality_score', 0)
        if quality > 0.8:
            st.markdown("<span style='color: #28a745;'>High Quality</span>", unsafe_allow_html=True)
        elif quality > 0.6:
            st.markdown("<span style='color: #ffc107;'>Medium Quality</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #dc3545;'>Low Quality</span>", unsafe_allow_html=True)

def display_poem_details(result: Dict[str, Any]) -> None:
    """Display detailed analysis of a poem"""
    if not result['success']:
        return
    
    analysis = result['analysis']
    
    # Structural Analysis
    st.subheader("Structural Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Lines:** {analysis.structural.lines}")
        st.write(f"**Average Syllables:** {analysis.structural.avg_syllables:.1f}")
        st.write(f"**Stanza Form:** {analysis.structural.stanza_structure}")
        
        # Check for Radīf
        rhyme_scheme = analysis.structural.rhyme_scheme
        radif_values = [r.radif for r in rhyme_scheme if r.radif]
        if radif_values and len(set(radif_values)) == 1:
            st.write(f"**Radīf:** {radif_values[0]}")
    
    with col2:
        st.write(f"**Meter:** {analysis.structural.aruz_analysis.identified_meter}")
        st.write(f"**Confidence:** {analysis.structural.meter_confidence.value}")
        st.write(f"**Rhyme Pattern:** {analysis.structural.rhyme_pattern}")
    
    # Content Analysis
    st.subheader("Content Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Total Words:** {analysis.content.total_words}")
        st.write(f"**Unique Words:** {analysis.content.unique_words}")
    
    with col2:
        st.write(f"**Lexical Diversity:** {analysis.content.lexical_diversity:.1%}")
        st.write(f"**Register:** {analysis.content.stylistic_register}")
    
    with col3:
        if analysis.content.neologisms:
            st.write(f"**Neologisms:** {len(analysis.content.neologisms)}")
        if analysis.content.archaisms:
            st.write(f"**Archaisms:** {len(analysis.content.archaisms)}")
    
    # Themes
    active_themes = [k for k, v in analysis.content.theme_distribution.items() if v > 0]
    if active_themes:
        st.write(f"**Themes:** {', '.join(active_themes)}")
    
    # Quality Metrics
    st.subheader("Quality Assessment")
    quality_metrics = analysis.quality_metrics
    if quality_metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Quality Score:** {quality_metrics.get('quality_score', 0):.0%}")
        with col2:
            st.write(f"**Reliability:** {quality_metrics.get('reliability', 'unknown')}")
        
        warnings = quality_metrics.get('warnings', [])
        if warnings:
            st.warning("**Warnings:** " + "; ".join(warnings))

# -------------------------------------------------------------------
# Tab Functions
# -------------------------------------------------------------------
def render_upload_tab():
    """Render the upload and analysis tab"""
    st.header("Upload and Analyze")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT file",
        type=['pdf', 'txt'],
        help="Supports normal and scanned PDFs",
        key="file_uploader"
    )
    
    if uploaded_file:
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # Save to temp file and process
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        
        try:
            # Extract text
            if PDF_HANDLER_AVAILABLE:
                with st.spinner("Extracting text from file..."):
                    text = read_file_with_pdf_support(tmp_path)
                    st.session_state.extracted_text = text
                    st.success(f"Text extracted: {len(text)} characters")
            else:
                # For TXT files
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                st.session_state.extracted_text = text
            
            # Show extracted text preview
            with st.expander("View extracted text"):
                st.text_area("Content", text[:2000] + "..." if len(text) > 2000 else text, height=200)
            
            # Analysis configuration
            st.subheader("Analysis Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_mode = st.radio(
                    "Analysis Mode",
                    ["Enhanced (with free verse detection)", "Classical (Arūḍ only)"],
                    key="analysis_mode_radio"
                )
                st.session_state.analysis_mode = analysis_mode
            
            with col2:
                split_method = st.selectbox(
                    "Poem Splitting Method",
                    ["Automatic (by separators)", "Manual (by blank lines)"],
                    key="split_method"
                )
            
            # Split poems
            if st.button("Split Poems and Preview", key="split_button"):
                text = st.session_state.extracted_text
                
                if split_method == "Automatic (by separators)":
                    poems = split_poems_auto(text)
                else:
                    # Split by double blank lines
                    poems = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                
                st.session_state.poems = poems
                st.success(f"Found {len(poems)} poems")
                
                # Show preview
                with st.expander("Poem Preview"):
                    for i, poem in enumerate(poems[:5]):  # Show first 5
                        st.write(f"**Poem {i+1}** ({len(poem)} characters):")
                        st.text(poem[:200] + "..." if len(poem) > 200 else poem)
                        st.write("---")
            
            # Start analysis button
            if st.session_state.poems and st.button("Start Analysis", type="primary", key="analyze_button"):
                with st.spinner("Analyzing poems..."):
                    results = []
                    
                    for i, poem in enumerate(st.session_state.poems):
                        # Validate poem content
                        if not validate_poem_content(poem):
                            results.append({
                                'poem_text': poem,
                                'poem_num': i + 1,
                                'title': f"Poem {i + 1}",
                                'error': "Invalid content (too short or no Tajik characters)",
                                'success': False
                            })
                            continue
                        
                        # Choose analyzer based on mode
                        if "Enhanced" in st.session_state.analysis_mode:
                            result = analyze_poem_enhanced(poem, i + 1)
                        else:
                            result = analyze_poem_classical(poem, i + 1)
                        
                        results.append(result)
                    
                    st.session_state.analysis_results = results
                    
                    # Calculate success rate
                    successful = sum(1 for r in results if r['success'])
                    st.success(f"Analysis complete: {successful}/{len(results)} successful")
                    
                    # Switch to results tab
                    st.session_state.current_tab = 'results'
                    st.rerun()
        
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()
    
    else:
        # Show instructions when no file uploaded
        st.info("Please upload a PDF or TXT file to begin analysis.")
        
        st.markdown("### Analysis Capabilities")
        st.markdown("""
        - **Classical Analysis**: 16 Arūḍ meters, Qāfiyeh/Radīf detection, prosodic analysis
        - **Enhanced Analysis**: Free verse detection, modern metrics, enjambement analysis
        - **PDF Support**: Both normal and scanned PDFs with OCR
        - **Content Analysis**: Lexical diversity, themes, neologisms, archaisms
        """)

def render_results_tab():
    """Render the results tab"""
    st.header("Analysis Results")
    
    if not st.session_state.analysis_results:
        st.info("No analysis results available. Please upload and analyze a file first.")
        return
    
    # Summary statistics
    results = st.session_state.analysis_results
    successful = sum(1 for r in results if r['success'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Poems", len(results))
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Failed", len(results) - successful)
    
    # Export options at the top
    st.subheader("Export Options")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("Generate Excel Report", key="excel_button"):
            generate_excel_report()
    
    with export_col2:
        if st.button("Add to Corpus", type="primary", key="corpus_button"):
            st.session_state.current_tab = 'corpus'
            st.rerun()
    
    with export_col3:
        if st.button("New Analysis", key="new_analysis_button"):
            # Reset analysis results
            st.session_state.analysis_results = []
            st.session_state.poems = []
            st.session_state.current_tab = 'upload'
            st.rerun()
    
    # Results display
    st.subheader("Poem Analysis")
    
    for i, result in enumerate(results):
        # Create expander for each poem
        with st.expander(f"{result['title']} - {'✓' if result['success'] else '✗'}", 
                        expanded=(i == 0)):  # First one expanded by default
            if result['success']:
                display_poem_details(result)
            else:
                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                st.text_area("Poem Text", result['poem_text'][:500] + "..." 
                           if len(result['poem_text']) > 500 else result['poem_text'], 
                           height=150)

def render_corpus_tab():
    """Render the corpus management tab"""
    st.header("Corpus Management")
    
    if not st.session_state.analysis_results:
        st.warning("No analysis results available to add to corpus. Please analyze poems first.")
        return
    
    successful_results = [r for r in st.session_state.analysis_results if r['success']]
    
    if not successful_results:
        st.error("No successful analyses to add to corpus.")
        return
    
    # Corpus type selection
    st.subheader("Corpus Configuration")
    
    corpus_type = st.radio(
        "Select corpus type:",
        ["Linguistic Corpus (for OCR/KI training)", 
         "Literary Corpus (for literary analysis)",
         "Both"],
        key="corpus_type_radio"
    )
    
    # Initialize corpus managers based on availability
    if EXTENDED_CORPUS_AVAILABLE and corpus_type in ["Literary Corpus (for literary analysis)", "Both"]:
        st.info("Extended literary corpus features available.")
        render_extended_corpus_interface(successful_results, corpus_type)
    elif BASIC_CORPUS_AVAILABLE:
        st.info("Basic corpus features available.")
        render_basic_corpus_interface(successful_results, corpus_type)
    else:
        st.error("No corpus manager available. Please ensure corpus_manager.py or extended_corpus_manager.py is available.")

def render_extended_corpus_interface(results: List[Dict], corpus_type: str):
    """Render interface for extended corpus manager"""
    
    # Metadata collection
    st.subheader("Volume Metadata")
    
    with st.form(key="volume_metadata_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            author_name = st.text_input("Author Name*", key="author_name")
            volume_title = st.text_input("Volume Title*", key="volume_title")
            publication_year = st.number_input("Publication Year*", 
                                             min_value=1800, 
                                             max_value=datetime.now().year,
                                             value=2023,
                                             key="pub_year")
        
        with col2:
            publisher = st.text_input("Publisher", key="publisher")
            city = st.text_input("City of Publication", key="city")
            isbn = st.text_input("ISBN", key="isbn")
        
        # Genres (simplified for now)
        genres = st.multiselect(
            "Genres",
            ["Ghazal", "Qasida", "Rubaiyat", "Free Verse", "Lyric", "Epic", "Satire"],
            key="genres"
        )
        
        submitted = st.form_submit_button("Add to Literary Corpus")
        
        if submitted:
            if not author_name or not volume_title:
                st.error("Author name and volume title are required.")
            else:
                try:
                    # Prepare poems data
                    poems_data = []
                    for result in results:
                        analysis = result['analysis']
                        # Convert analysis to dict if needed
                        if hasattr(analysis, '__dict__'):
                            analysis_dict = analysis.__dict__
                        else:
                            analysis_dict = analysis
                        
                        poems_data.append({
                            'content': result['poem_text'],
                            'analysis': analysis_dict
                        })
                    
                    # Create metadata dictionary
                    metadata = {
                        'author_name': author_name,
                        'volume_title': volume_title,
                        'publication_year': publication_year,
                        'publisher': publisher if publisher else None,
                        'city': city if city else None,
                        'genres': genres,
                        'isbn': isbn if isbn else None
                    }
                    
                    # Initialize library manager
                    library_manager = TajikLibraryManager()
                    
                    # Register volume (simplified - would need proper VolumeMetadata object)
                    # For now, we'll create a basic registration
                    st.info("Extended corpus integration would be implemented here.")
                    st.success(f"Prepared {len(poems_data)} poems for literary corpus.")
                    
                except Exception as e:
                    st.error(f"Error preparing corpus data: {str(e)}")

def render_basic_corpus_interface(results: List[Dict], corpus_type: str):
    """Render interface for basic corpus manager"""
    
    st.subheader("Basic Corpus Contribution")
    
    # User info (optional)
    with st.expander("Contributor Information (optional)"):
        username = st.text_input("GitHub Username", key="github_username")
        email = st.text_input("Email", key="email")
    
    license_accepted = st.checkbox(
        "I accept the CC-BY-NC-SA 4.0 license for this contribution",
        value=True,
        key="license_check"
    )
    
    if st.button("Save to Corpus", type="primary", key="save_corpus_button"):
        if not license_accepted:
            st.error("You must accept the license to contribute.")
        else:
            try:
                corpus_manager = TajikCorpusManager()
                saved_count = 0
                
                for result in results:
                    # Prepare contribution
                    user_info = {}
                    if username:
                        user_info["username"] = username
                    if email:
                        user_info["email"] = email
                    
                    # Prepare analysis_result structure
                    analysis_result = {
                        "poem_id": f"P{result['poem_num']:03d}",
                        "title": result['title'],
                        "content": result['poem_text'],
                        "analysis": result['analysis'],
                        "validation": result['analysis'].quality_metrics if hasattr(result['analysis'], 'quality_metrics') else {}
                    }
                    
                    contribution = corpus_manager.prepare_contribution(
                        analysis_result=analysis_result,
                        raw_text=result['poem_text'],
                        user_info=user_info if user_info else {"anonymous": True}
                    )
                    
                    contribution["metadata"]["license_accepted"] = True
                    
                    # Save contribution
                    corpus_manager.save_contribution(contribution)
                    saved_count += 1
                
                st.success(f"Successfully saved {saved_count} poems to corpus.")
                
                # Show corpus statistics
                stats = corpus_manager.get_corpus_statistics()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Poems", stats.get("total_poems", 0))
                with col2:
                    st.metric("Total Lines", stats.get("total_lines", 0))
                with col3:
                    st.metric("Total Words", stats.get("total_words", 0))
                
            except Exception as e:
                st.error(f"Error saving to corpus: {str(e)}")

def generate_excel_report():
    """Generate Excel report from analysis results"""
    try:
        results = st.session_state.analysis_results
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            st.error("No successful analyses to export.")
            return
        
        # Prepare data for ExcelReporter
        excel_data = []
        for result in successful_results:
            excel_data.append({
                'poem_id': f"P{result['poem_num']:03d}",
                'title': result['title'],
                'content': result['poem_text'],
                'analysis': result['analysis'],
                'validation': result['analysis'].quality_metrics if hasattr(result['analysis'], 'quality_metrics') else {}
            })
        
        # Create Excel report
        excel_reporter = ExcelReporter()
        mode = "enhanced" if "Enhanced" in st.session_state.analysis_mode else "classical"
        filename = f"tajik_poetry_{mode}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp_path = Path(tmp.name)
            excel_reporter.create_report(excel_data, str(tmp_path))
            
            # Provide download button
            with open(tmp_path, 'rb') as f:
                excel_bytes = f.read()
            
            st.download_button(
                label="Download Excel Report",
                data=excel_bytes,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )
            
            st.success(f"Excel report generated: {filename}")
        
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()
            
    except Exception as e:
        st.error(f"Error generating Excel report: {str(e)}")
        logger.error(f"Excel report error: {e}")

# -------------------------------------------------------------------
# Main Application
# -------------------------------------------------------------------
def main():
    """Main application function"""
    
    # Initialize session state
    init_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="Tajik Poetry Analyzer",
        page_icon="📖",
        layout="wide"
    )
    
    # Title and description
    st.title("Tajik Poetry Analyzer")
    st.markdown("Advanced scientific analysis of Tajik poetry with classical and modern approaches")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Tab selection
        tabs = {
            "upload": "📤 Upload & Analyze",
            "results": "📊 View Results",
            "corpus": "📚 Manage Corpus"
        }
        
        selected_tab = st.radio(
            "Go to:",
            list(tabs.values()),
            key="tab_selector"
        )
        
        # Map back to tab key
        for key, value in tabs.items():
            if value == selected_tab:
                st.session_state.current_tab = key
                break
        
        st.markdown("---")
        
        # Analysis mode selection
        st.header("Analysis Mode")
        analysis_mode = st.radio(
            "Select mode:",
            ["Enhanced (with free verse detection)", "Classical (Arūḍ only)"],
            key="sidebar_analysis_mode"
        )
        st.session_state.analysis_mode = analysis_mode
        
        st.markdown("---")
        
        # Status information
        st.header("Status")
        if st.session_state.analysis_results:
            successful = sum(1 for r in st.session_state.analysis_results if r['success'])
            st.write(f"Analyzed poems: {successful}/{len(st.session_state.analysis_results)}")
        
        if st.session_state.uploaded_file_name:
            st.write(f"File: {st.session_state.uploaded_file_name}")
        
        st.markdown("---")
        
        # Reset button
        if st.button("Reset Application", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content area based on selected tab
    if st.session_state.current_tab == 'upload':
        render_upload_tab()
    elif st.session_state.current_tab == 'results':
        render_results_tab()
    elif st.session_state.current_tab == 'corpus':
        render_corpus_tab()

if __name__ == "__main__":
    # Check dependencies
    if not ANALYZER_AVAILABLE:
        st.error("Error: Analyzer not available. Please ensure analyzer.py is in the same directory.")
        st.stop()
    
    if not PDF_HANDLER_AVAILABLE:
        st.error("Error: PDF handler not available. Please ensure pdf_handler.py is in the same directory.")
        st.stop()
    
    main()
