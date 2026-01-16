#!/usr/bin/env python3
"""
Tajik Poetry Analyzer - Fixed Streamlit UI
Fixed analysis display issues
"""

import streamlit as st
from pathlib import Path
import tempfile
import re
import json
from datetime import datetime
from typing import List, Dict, Any
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Import Handling with Better Error Messages
# -------------------------------------------------------------------
def safe_import():
    """Safely import required modules with clear error messages"""
    imports = {}
    errors = []
    
    # Try to import analyzer
    try:
        from analyzer import (
            TajikPoemAnalyzer,
            EnhancedTajikPoemAnalyzer,
            AnalysisConfig,
            MeterConfidence,
            QualityValidator,
            ExcelReporter,
        )
        imports['analyzer'] = {
            'TajikPoemAnalyzer': TajikPoemAnalyzer,
            'EnhancedTajikPoemAnalyzer': EnhancedTajikPoemAnalyzer,
            'AnalysisConfig': AnalysisConfig,
            'MeterConfidence': MeterConfidence,
            'QualityValidator': QualityValidator,
            'ExcelReporter': ExcelReporter,
        }
        logger.info("✓ Analyzer imported successfully")
    except ImportError as e:
        errors.append(f"Analyzer import error: {e}")
        logger.error(f"Analyzer import failed: {e}")
    
    # Try to import corpus managers
    try:
        from extended_corpus_manager import TajikLibraryManager
        imports['extended_corpus'] = TajikLibraryManager
        logger.info("✓ Extended corpus manager imported")
    except ImportError as e:
        logger.warning(f"Extended corpus manager not available: {e}")
    
    try:
        from corpus_manager import TajikCorpusManager
        imports['basic_corpus'] = TajikCorpusManager
        logger.info("✓ Basic corpus manager imported")
    except ImportError as e:
        logger.warning(f"Basic corpus manager not available: {e}")
    
    # Try to import PDF handler
    try:
        from pdf_handler import read_file_with_pdf_support
        imports['pdf_handler'] = read_file_with_pdf_support
        logger.info("✓ PDF handler imported")
    except ImportError as e:
        errors.append(f"PDF handler import error: {e}")
        logger.error(f"PDF handler import failed: {e}")
    
    return imports, errors

# Safe import
IMPORTS, IMPORT_ERRORS = safe_import()

# Check critical imports
ANALYZER_AVAILABLE = 'analyzer' in IMPORTS
PDF_HANDLER_AVAILABLE = 'pdf_handler' in IMPORTS
EXTENDED_CORPUS_AVAILABLE = 'extended_corpus' in IMPORTS
BASIC_CORPUS_AVAILABLE = 'basic_corpus' in IMPORTS

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
        'analysis_mode': 'enhanced',
        'corpus_type': 'both',
        'uploaded_file_name': '',
        'show_details': {},
        'corpus_metadata': {},
        'excel_data': None,
        'excel_filename': '',
        'analysis_in_progress': False,
        'last_error': None
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
    return f"Poem"

def validate_poem_content(poem_text: str) -> bool:
    """Validate poem content before analysis"""
    if not poem_text or len(poem_text.strip()) < 30:
        return False
    
    # Check for Tajik/Persian characters
    tajik_chars = set('АБВГҒДЕЁЖЗИӢЙКҚЛМНОПРСТУӮФХҲЧҶШЪЭЮЯабвгғдеёжзиӣйкқлмнопрстуӯфхҳчҷшъэюяӣӯ')
    arabic_persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
    
    text_set = set(poem_text)
    has_tajik = bool(text_set & tajik_chars)
    has_persian = bool(text_set & arabic_persian_chars)
    
    return has_tajik or has_persian

# -------------------------------------------------------------------
# Analysis Functions - SIMPLIFIED
# -------------------------------------------------------------------
def analyze_poem_simple(poem_text: str, poem_num: int, mode: str = 'enhanced') -> Dict[str, Any]:
    """Simplified poem analysis that handles errors gracefully"""
    try:
        if not ANALYZER_AVAILABLE:
            return {
                'poem_text': poem_text,
                'poem_num': poem_num,
                'title': extract_title_from_poem(poem_text),
                'error': 'Analyzer not available',
                'mode': mode,
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
        
        config = IMPORTS['analyzer']['AnalysisConfig'](lexicon_path='data/tajik_lexicon.json')
        
        if mode == 'enhanced':
            analyzer = IMPORTS['analyzer']['EnhancedTajikPoemAnalyzer'](config=config, enable_corpus=False)
        else:
            analyzer = IMPORTS['analyzer']['TajikPoemAnalyzer'](config=config)
        
        # Perform analysis
        analysis = analyzer.analyze_poem(poem_text)
        
        # Extract basic information
        result = {
            'poem_text': poem_text,
            'poem_num': poem_num,
            'title': extract_title_from_poem(poem_text),
            'mode': mode,
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'basic_info': {}
        }
        
        # Try to extract structural info
        try:
            if hasattr(analysis, 'structural'):
                structural = analysis.structural
                result['basic_info']['lines'] = structural.lines
                result['basic_info']['avg_syllables'] = getattr(structural, 'avg_syllables', 0)
                result['basic_info']['stanza_structure'] = getattr(structural, 'stanza_structure', 'unknown')
                result['basic_info']['rhyme_pattern'] = getattr(structural, 'rhyme_pattern', '')
                
                if hasattr(structural, 'aruz_analysis'):
                    result['basic_info']['meter'] = structural.aruz_analysis.identified_meter
                    result['basic_info']['meter_confidence'] = structural.aruz_analysis.confidence.value
                
                # Check for free verse
                if mode == 'enhanced' and hasattr(structural, 'is_free_verse'):
                    result['basic_info']['is_free_verse'] = structural.is_free_verse
        except Exception as e:
            logger.error(f"Error extracting structural info: {e}")
            result['basic_info']['structural_error'] = str(e)
        
        # Try to extract content info
        try:
            if hasattr(analysis, 'content'):
                content = analysis.content
                result['basic_info']['total_words'] = getattr(content, 'total_words', 0)
                result['basic_info']['unique_words'] = getattr(content, 'unique_words', 0)
                result['basic_info']['lexical_diversity'] = getattr(content, 'lexical_diversity', 0)
                result['basic_info']['stylistic_register'] = getattr(content, 'stylistic_register', 'unknown')
        except Exception as e:
            logger.error(f"Error extracting content info: {e}")
            result['basic_info']['content_error'] = str(e)
        
        # Try to extract quality metrics
        try:
            if hasattr(analysis, 'quality_metrics'):
                result['basic_info']['quality_score'] = analysis.quality_metrics.get('quality_score', 0)
                result['basic_info']['reliability'] = analysis.quality_metrics.get('reliability', 'unknown')
        except Exception as e:
            logger.error(f"Error extracting quality metrics: {e}")
        
        # Store the full analysis object for experts
        result['full_analysis'] = analysis
        
        return result
        
    except Exception as e:
        logger.error(f"Error in poem analysis {poem_num}: {e}")
        return {
            'poem_text': poem_text,
            'poem_num': poem_num,
            'title': extract_title_from_poem(poem_text),
            'error': str(e),
            'traceback': traceback.format_exc(),
            'mode': mode,
            'success': False,
            'timestamp': datetime.now().isoformat()
        }

# -------------------------------------------------------------------
# Display Functions - SIMPLIFIED
# -------------------------------------------------------------------
def display_poem_summary(result: Dict[str, Any]) -> None:
    """Display summary of poem analysis"""
    if not result['success']:
        with st.expander(f"❌ Poem {result['poem_num']} - Failed", expanded=False):
            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                with st.expander("Show technical details"):
                    st.code(result['traceback'], language='python')
            st.text_area("Poem Text", result['poem_text'][:500] + "..." 
                       if len(result['poem_text']) > 500 else result['poem_text'], 
                       height=150)
        return
    
    basic_info = result.get('basic_info', {})
    title = result['title']
    
    # Create columns for summary
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.write(f"**{title}** (Poem {result['poem_num']})")
        st.write(f"Lines: {basic_info.get('lines', 'N/A')}")
        st.write(f"Words: {basic_info.get('total_words', 'N/A')}")
    
    with col2:
        meter = basic_info.get('meter', 'unknown')
        st.write(f"Meter: {meter}")
        rhyme = basic_info.get('rhyme_pattern', '')
        if rhyme:
            st.write(f"Rhyme: {rhyme}")
    
    with col3:
        if basic_info.get('is_free_verse', False):
            st.markdown("**Free Verse**")
        else:
            st.markdown("**Classical**")
        
        quality = basic_info.get('quality_score', 0)
        if quality > 0.8:
            st.markdown("High Quality")
        elif quality > 0.6:
            st.markdown("Medium Quality")
        else:
            st.markdown("Low Quality")

def display_poem_details(result: Dict[str, Any]) -> None:
    """Display detailed analysis of a poem"""
    if not result['success']:
        return
    
    basic_info = result.get('basic_info', {})
    
    # Structural Analysis
    st.subheader("Structural Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Lines:** {basic_info.get('lines', 'N/A')}")
        st.write(f"**Average Syllables:** {basic_info.get('avg_syllables', 'N/A')}")
        st.write(f"**Stanza Form:** {basic_info.get('stanza_structure', 'N/A')}")
        
        # Check for Radīf
        rhyme_scheme = basic_info.get('rhyme_scheme', [])
        if rhyme_scheme:
            radif_values = [r.get('radif', '') for r in rhyme_scheme if r.get('radif')]
            if radif_values and len(set(radif_values)) == 1:
                st.write(f"**Radīf:** {radif_values[0]}")
    
    with col2:
        st.write(f"**Meter:** {basic_info.get('meter', 'N/A')}")
        st.write(f"**Confidence:** {basic_info.get('meter_confidence', 'N/A')}")
        st.write(f"**Rhyme Pattern:** {basic_info.get('rhyme_pattern', 'N/A')}")
    
    # Content Analysis
    if any(k in basic_info for k in ['total_words', 'unique_words', 'lexical_diversity']):
        st.subheader("Content Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'total_words' in basic_info:
                st.write(f"**Total Words:** {basic_info['total_words']}")
            if 'unique_words' in basic_info:
                st.write(f"**Unique Words:** {basic_info['unique_words']}")
        
        with col2:
            if 'lexical_diversity' in basic_info:
                st.write(f"**Lexical Diversity:** {basic_info['lexical_diversity']:.1%}")
            if 'stylistic_register' in basic_info:
                st.write(f"**Register:** {basic_info['stylistic_register']}")
        
        with col3:
            # Check if we have the full analysis for neologisms/archaisms
            if 'full_analysis' in result:
                try:
                    analysis = result['full_analysis']
                    if hasattr(analysis, 'content'):
                        if hasattr(analysis.content, 'neologisms') and analysis.content.neologisms:
                            st.write(f"**Neologisms:** {len(analysis.content.neologisms)}")
                        if hasattr(analysis.content, 'archaisms') and analysis.content.archaisms:
                            st.write(f"**Archaisms:** {len(analysis.content.archaisms)}")
                        
                        # Themes
                        if hasattr(analysis.content, 'theme_distribution'):
                            themes = analysis.content.theme_distribution
                            active_themes = [k for k, v in themes.items() if v > 0]
                            if active_themes:
                                st.write(f"**Themes:** {', '.join(active_themes[:3])}")
                except Exception as e:
                    logger.error(f"Error extracting detailed content: {e}")
    
    # Quality Metrics
    if 'quality_score' in basic_info or 'reliability' in basic_info:
        st.subheader("Quality Assessment")
        col1, col2 = st.columns(2)
        with col1:
            if 'quality_score' in basic_info:
                st.write(f"**Quality Score:** {basic_info['quality_score']:.0%}")
        with col2:
            if 'reliability' in basic_info:
                st.write(f"**Reliability:** {basic_info['reliability']}")
    
    # Show errors if any
    for key in ['structural_error', 'content_error']:
        if key in basic_info:
            st.warning(f"{key.replace('_', ' ').title()}: {basic_info[key]}")
    
    # Expert view (collapsed by default)
    if 'full_analysis' in result:
        with st.expander("Expert View (Raw Analysis)", expanded=False):
            try:
                analysis = result['full_analysis']
                st.json(json.loads(json.dumps(analysis, default=str)))
            except:
                st.write("Could not display raw analysis")

# -------------------------------------------------------------------
# Tab Functions
# -------------------------------------------------------------------
def render_upload_tab():
    """Render the upload and analysis tab"""
    st.header("Upload and Analyze")
    
    # Show import status
    if IMPORT_ERRORS:
        st.warning("Some imports failed. Basic functionality may be limited.")
        with st.expander("Import Errors"):
            for error in IMPORT_ERRORS:
                st.error(error)
    
    if not ANALYZER_AVAILABLE:
        st.error("Critical error: Analyzer module not available. Cannot perform analysis.")
        st.stop()
    
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
                    text = IMPORTS['pdf_handler'](tmp_path)
                    st.session_state.extracted_text = text
                    st.success(f"Text extracted: {len(text)} characters")
            else:
                # For TXT files
                if uploaded_file.name.endswith('.txt'):
                    text = uploaded_file.getvalue().decode('utf-8')
                    st.session_state.extracted_text = text
                    st.success(f"Text loaded: {len(text)} characters")
                else:
                    st.error("PDF handler not available. Cannot process PDF files.")
                    return
            
            # Show extracted text preview
            with st.expander("View extracted text", expanded=False):
                st.text_area("Content", text[:2000] + "..." if len(text) > 2000 else text, 
                           height=200, key="text_preview")
            
            # Analysis configuration
            st.subheader("Analysis Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_mode = st.radio(
                    "Analysis Mode",
                    ["Enhanced (with free verse detection)", "Classical (Arūḍ only)"],
                    key="analysis_mode_radio",
                    index=0
                )
                st.session_state.analysis_mode = 'enhanced' if 'Enhanced' in analysis_mode else 'classical'
            
            with col2:
                split_method = st.selectbox(
                    "Poem Splitting Method",
                    ["Automatic (by separators)", "Manual (by blank lines)"],
                    key="split_method",
                    index=0
                )
            
            # Split poems button
            if st.button("Preview Poem Split", key="split_button"):
                text = st.session_state.extracted_text
                
                if split_method == "Automatic (by separators)":
                    poems = split_poems_auto(text)
                else:
                    # Split by double blank lines
                    poems = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
                
                st.session_state.poems = poems
                
                if poems:
                    st.success(f"Found {len(poems)} potential poems")
                    
                    # Show preview
                    with st.expander("Poem Preview", expanded=True):
                        for i, poem in enumerate(poems[:10]):  # Show first 10
                            st.write(f"**Poem {i+1}** ({len(poem)} characters):")
                            st.text(poem[:200] + "..." if len(poem) > 200 else poem)
                            st.write("---")
                    
                    # Analysis button
                    if st.button("Start Analysis", type="primary", key="analyze_button"):
                        st.session_state.analysis_in_progress = True
                        st.session_state.last_error = None
                        
                        # Perform analysis
                        with st.spinner(f"Analyzing {len(poems)} poems..."):
                            results = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, poem in enumerate(poems):
                                # Update progress
                                progress = (i + 1) / len(poems)
                                progress_bar.progress(progress)
                                status_text.text(f"Analyzing poem {i+1} of {len(poems)}...")
                                
                                # Validate poem content
                                if not validate_poem_content(poem):
                                    results.append({
                                        'poem_text': poem,
                                        'poem_num': i + 1,
                                        'title': f"Poem {i + 1}",
                                        'error': "Invalid content (too short or no Tajik/Persian characters)",
                                        'success': False
                                    })
                                    continue
                                
                                # Analyze poem
                                result = analyze_poem_simple(poem, i + 1, st.session_state.analysis_mode)
                                results.append(result)
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.session_state.analysis_results = results
                            st.session_state.analysis_in_progress = False
                            
                            # Calculate success rate
                            successful = sum(1 for r in results if r['success'])
                            if successful > 0:
                                st.success(f"Analysis complete: {successful}/{len(results)} poems analyzed successfully")
                                
                                # Switch to results tab
                                st.session_state.current_tab = 'results'
                                st.rerun()
                            else:
                                st.error("No poems could be analyzed successfully. Check the error messages.")
                else:
                    st.warning("No poems found. Try a different splitting method.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"File processing error: {e}")
            st.session_state.last_error = str(e)
        
        finally:
            # Clean up temp file
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except:
                    pass
    
    else:
        # Show instructions when no file uploaded
        st.info("Please upload a PDF or TXT file to begin analysis.")
        
        # Sample analysis
        st.markdown("### Quick Test")
        if st.button("Test with Sample Poem", key="test_button"):
            sample_poem = """Булбул дар боғ мехонад суруди зебо,
Гул мешукуфад дар субҳи навбаҳор.
Дили ман бо ишқи ватан пур аст,
Табиати зебои Тоҷикистон."""
            
            st.session_state.poems = [sample_poem]
            st.session_state.analysis_mode = 'enhanced'
            
            with st.spinner("Testing analysis..."):
                result = analyze_poem_simple(sample_poem, 1, 'enhanced')
                st.session_state.analysis_results = [result]
                
                if result['success']:
                    st.success("Sample analysis successful!")
                    st.session_state.current_tab = 'results'
                    st.rerun()
                else:
                    st.error(f"Sample analysis failed: {result.get('error', 'Unknown error')}")

def render_results_tab():
    """Render the results tab"""
    st.header("Analysis Results")
    
    if not st.session_state.analysis_results:
        st.info("No analysis results available. Please upload and analyze a file first.")
        st.button("Go to Upload", key="go_to_upload", 
                 on_click=lambda: setattr(st.session_state, 'current_tab', 'upload'))
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
    
    # Export options at the top - SIMPLIFIED
    st.subheader("Export Options")
    
    if successful > 0:
        # Simple Excel export
        if st.button("Download Results as Excel", key="download_excel"):
            try:
                # Prepare data for Excel
                import pandas as pd
                from io import BytesIO
                
                data = []
                for result in results:
                    if result['success']:
                        row = {
                            'Poem Number': result['poem_num'],
                            'Title': result['title'],
                            'Lines': result['basic_info'].get('lines', ''),
                            'Meter': result['basic_info'].get('meter', ''),
                            'Rhyme Pattern': result['basic_info'].get('rhyme_pattern', ''),
                            'Total Words': result['basic_info'].get('total_words', ''),
                            'Lexical Diversity': result['basic_info'].get('lexical_diversity', ''),
                            'Quality Score': result['basic_info'].get('quality_score', ''),
                        }
                        data.append(row)
                
                if data:
                    df = pd.DataFrame(data)
                    
                    # Convert to Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Analysis Results', index=False)
                    
                    # Provide download button
                    st.download_button(
                        label="Click to download Excel file",
                        data=output.getvalue(),
                        file_name=f"tajik_poetry_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="excel_download"
                    )
                else:
                    st.warning("No successful analyses to export.")
                    
            except Exception as e:
                st.error(f"Error creating Excel file: {str(e)}")
        
        # Corpus addition
        if st.button("Add to Corpus", type="primary", key="add_to_corpus"):
            st.session_state.current_tab = 'corpus'
            st.rerun()
    
    # New analysis button
    if st.button("New Analysis", key="new_analysis"):
        st.session_state.analysis_results = []
        st.session_state.poems = []
        st.session_state.current_tab = 'upload'
        st.rerun()
    
    # Results display
    st.subheader("Poem Analysis Details")
    
    # Filter options
    show_all = st.checkbox("Show all poems", value=True, key="show_all")
    show_only_successful = st.checkbox("Show only successful analyses", value=False, key="show_only_successful")
    
    # Filter results
    filtered_results = results
    if show_only_successful:
        filtered_results = [r for r in results if r['success']]
    if not show_all and len(filtered_results) > 10:
        filtered_results = filtered_results[:10]
        st.info(f"Showing first 10 of {len(results)} poems. Check 'Show all' to see all.")
    
    # Display filtered results
    for i, result in enumerate(filtered_results):
        display_poem_summary(result)
        
        # Show details in expander
        if result['success']:
            with st.expander(f"View detailed analysis for {result['title']}", expanded=False):
                display_poem_details(result)
        
        st.write("---")

def render_corpus_tab():
    """Render the corpus management tab - SIMPLIFIED"""
    st.header("Corpus Management")
    
    if not st.session_state.analysis_results:
        st.warning("No analysis results available to add to corpus.")
        st.button("Go to Results", key="go_to_results_from_corpus",
                 on_click=lambda: setattr(st.session_state, 'current_tab', 'results'))
        return
    
    successful_results = [r for r in st.session_state.analysis_results if r['success']]
    
    if not successful_results:
        st.error("No successful analyses to add to corpus.")
        return
    
    # Simple corpus interface
    st.info(f"{len(successful_results)} successful analyses available for corpus addition.")
    
    # Basic metadata
    st.subheader("Basic Metadata")
    
    author_name = st.text_input("Author Name", key="corpus_author")
    publication_year = st.number_input("Publication Year", 
                                     min_value=1800, 
                                     max_value=datetime.now().year,
                                     value=2023,
                                     key="corpus_year")
    
    license_accepted = st.checkbox(
        "I accept the CC-BY-NC-SA 4.0 license for this contribution",
        value=True,
        key="corpus_license"
    )
    
    if st.button("Save to Local Corpus", type="primary", key="save_corpus"):
        if not license_accepted:
            st.error("You must accept the license to contribute.")
        elif not author_name:
            st.error("Author name is required.")
        else:
            try:
                # Simple file-based storage
                import json
                from datetime import datetime
                
                corpus_data = {
                    'metadata': {
                        'author': author_name,
                        'publication_year': publication_year,
                        'contributor': 'Streamlit App User',
                        'date': datetime.now().isoformat(),
                        'license': 'CC-BY-NC-SA-4.0'
                    },
                    'poems': []
                }
                
                for result in successful_results:
                    poem_data = {
                        'poem_number': result['poem_num'],
                        'title': result['title'],
                        'text': result['poem_text'],
                        'analysis': result['basic_info'],
                        'analysis_mode': result['mode'],
                        'timestamp': result['timestamp']
                    }
                    corpus_data['poems'].append(poem_data)
                
                # Save to file
                filename = f"corpus_{author_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = Path(filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(corpus_data, f, ensure_ascii=False, indent=2)
                
                # Provide download
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = f.read()
                
                st.download_button(
                    label="Download Corpus JSON",
                    data=json_data,
                    file_name=filename,
                    mime="application/json",
                    key="corpus_download"
                )
                
                st.success(f"Corpus saved to {filename}")
                
            except Exception as e:
                st.error(f"Error saving corpus: {str(e)}")
    
    # Back button
    st.button("Back to Results", key="back_to_results",
             on_click=lambda: setattr(st.session_state, 'current_tab', 'results'))

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
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("Tajik Poetry Analyzer")
    st.markdown("Scientific analysis of Tajik poetry with classical and modern approaches")
    
    # Sidebar for navigation and info
    with st.sidebar:
        st.header("Navigation")
        
        # Tab selection with icons
        tab_options = {
            'upload': "📤 Upload & Analyze",
            'results': "📊 View Results", 
            'corpus': "📚 Manage Corpus"
        }
        
        selected_tab = st.radio(
            "Select Tab:",
            list(tab_options.values()),
            key="sidebar_tabs"
        )
        
        # Map back to tab key
        for key, value in tab_options.items():
            if value == selected_tab:
                st.session_state.current_tab = key
                break
        
        st.markdown("---")
        
        # Analysis mode in sidebar
        st.header("Analysis Mode")
        mode_display = st.radio(
            "Mode:",
            ["Enhanced", "Classical"],
            key="sidebar_mode",
            index=0 if st.session_state.analysis_mode == 'enhanced' else 1
        )
        st.session_state.analysis_mode = mode_display.lower()
        
        st.markdown("---")
        
        # Status information
        st.header("Status")
        
        if st.session_state.uploaded_file_name:
            st.caption(f"File: {st.session_state.uploaded_file_name}")
        
        if st.session_state.analysis_results:
            successful = sum(1 for r in st.session_state.analysis_results if r['success'])
            total = len(st.session_state.analysis_results)
            st.caption(f"Poems: {successful}/{total} successful")
        
        if st.session_state.analysis_in_progress:
            st.warning("Analysis in progress...")
        
        if st.session_state.last_error:
            with st.expander("Last Error"):
                st.error(st.session_state.last_error)
        
        st.markdown("---")
        
        # Help and info
        st.header("Help")
        with st.expander("How to use"):
            st.markdown("""
            1. **Upload** a PDF or TXT file with Tajik poetry
            2. **Preview** the poem split
            3. **Analyze** using enhanced or classical mode
            4. **View** results and download Excel report
            5. **Add** to corpus for research
            """)
        
        # Reset button
        if st.button("Reset Application", type="secondary", key="sidebar_reset"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content area
    try:
        if st.session_state.current_tab == 'upload':
            render_upload_tab()
        elif st.session_state.current_tab == 'results':
            render_results_tab()
        elif st.session_state.current_tab == 'corpus':
            render_corpus_tab()
    except Exception as e:
        st.error(f"Error in application: {str(e)}")
        logger.error(f"Application error: {e}")
        st.info("Try resetting the application from the sidebar.")

if __name__ == "__main__":
    # Check critical dependencies
    if not ANALYZER_AVAILABLE:
        st.error("""
        Critical error: Analyzer module not available.
        
        Please ensure:
        1. analyzer.py is in the same directory
        2. All dependencies are installed: pip install -r requirements.txt
        3. The file contains valid Python code
        """)
        st.stop()
    
    main()
