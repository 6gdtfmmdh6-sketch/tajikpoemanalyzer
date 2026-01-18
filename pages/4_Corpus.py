#!/usr/bin/env python3
"""
Corpus Management Page
Export training data for NLP, manage raw corpus, Git integration
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Import managers
try:
    from corpus_manager import TajikCorpusManager
    CORPUS_AVAILABLE = True
except ImportError:
    CORPUS_AVAILABLE = False

try:
    from extended_corpus_manager import TajikLibraryManager
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False


# -------------------------------------------------------------------
# Session State
# -------------------------------------------------------------------
def init_corpus_state():
    """Initialize corpus session state"""
    defaults = {
        'corpus_stats': None,
        'corpus_export_path': None,
        'corpus_message': None,
        'corpus_message_type': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_message():
    """Display stored message"""
    if st.session_state.corpus_message:
        if st.session_state.corpus_message_type == 'success':
            st.success(st.session_state.corpus_message)
        elif st.session_state.corpus_message_type == 'error':
            st.error(st.session_state.corpus_message)
        else:
            st.info(st.session_state.corpus_message)
        st.session_state.corpus_message = None
        st.session_state.corpus_message_type = None


# -------------------------------------------------------------------
# Data Functions
# -------------------------------------------------------------------
def get_corpus_statistics() -> Dict:
    """Get statistics from corpus manager"""
    if not CORPUS_AVAILABLE:
        return {}
    
    try:
        manager = TajikCorpusManager()
        return manager.get_corpus_statistics()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {}


def get_library_statistics() -> Dict:
    """Get statistics from library manager"""
    if not LIBRARY_AVAILABLE:
        return {}
    
    try:
        manager = TajikLibraryManager()
        return manager.get_statistics()
    except Exception as e:
        logger.error(f"Error getting library stats: {e}")
        return {}


def export_for_git() -> str:
    """Export contributions for Git"""
    if not CORPUS_AVAILABLE:
        return ""
    
    try:
        manager = TajikCorpusManager()
        path = manager.export_contributions_for_git()
        return str(path)
    except Exception as e:
        logger.error(f"Export error: {e}")
        return ""


def export_plaintext_corpus(output_path: str) -> int:
    """Export corpus as plaintext for NLP training"""
    contrib_dir = Path("tajik_corpus/contributions")
    
    if not contrib_dir.exists():
        return 0
    
    texts = []
    count = 0
    
    for file in contrib_dir.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get('raw_text', data.get('normalized_text', ''))
                if text:
                    texts.append(text)
                    count += 1
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    
    # Write combined text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(texts))
    
    return count


def export_jsonl_corpus(output_path: str) -> int:
    """Export corpus as JSONL for ML training"""
    contrib_dir = Path("tajik_corpus/contributions")
    
    if not contrib_dir.exists():
        return 0
    
    count = 0
    
    with open(output_path, 'w', encoding='utf-8') as out:
        for file in contrib_dir.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Create training record
                record = {
                    'text': data.get('raw_text', ''),
                    'metadata': {
                        'author': data.get('metadata', {}).get('author', ''),
                        'year': data.get('metadata', {}).get('publication_year', ''),
                        'collection': data.get('metadata', {}).get('collection', ''),
                    },
                    'tags': data.get('tags', [])
                }
                
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
                
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
    
    return count


def generate_timeline_html() -> str:
    """Generate HTML timeline report"""
    if not LIBRARY_AVAILABLE:
        return ""
    
    try:
        manager = TajikLibraryManager()
        return manager.generate_timeline_report("html")
    except Exception as e:
        logger.error(f"Timeline error: {e}")
        return ""


# -------------------------------------------------------------------
# UI Components
# -------------------------------------------------------------------
def render_statistics():
    """Render corpus statistics"""
    st.header("Corpus Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simple Corpus")
        stats = get_corpus_statistics()
        
        if stats:
            st.metric("Total Poems", stats.get("total_poems", 0))
            st.metric("Total Lines", stats.get("total_lines", 0))
            st.metric("Total Words", stats.get("total_words", 0))
            st.metric("Unique Words", stats.get("unique_words", 0))
            
            if stats.get("total_words", 0) > 0 and stats.get("unique_words", 0) > 0:
                diversity = stats["unique_words"] / stats["total_words"] * 100
                st.metric("Lexical Diversity", f"{diversity:.1f}%")
        else:
            st.info("No corpus data available")
    
    with col2:
        st.subheader("Library")
        lib_stats = get_library_statistics()
        
        if lib_stats:
            st.metric("Registered Volumes", lib_stats.get("total_volumes", 0))
            st.metric("Authors", lib_stats.get("authors_count", 0))
            st.metric("Library Poems", lib_stats.get("total_poems", 0))
            
            year_range = lib_stats.get("publication_years", {})
            if year_range.get("min") and year_range.get("max"):
                st.metric("Year Range", f"{year_range['min']} - {year_range['max']}")
        else:
            st.info("No library data available")


def render_export_section():
    """Render export options"""
    st.header("Export Options")
    
    st.markdown("""
    Export your corpus in different formats:
    - **Plaintext**: Simple text file for basic NLP tasks
    - **JSONL**: JSON Lines format with metadata for ML training
    - **Git Export**: Structured export for version control
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Plaintext Export")
        st.write("Raw text without metadata")
        
        if st.button("Export as TXT", key="btn_export_txt"):
            output_dir = Path("tajik_corpus/exports")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"corpus_plaintext_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            count = export_plaintext_corpus(str(output_path))
            
            if count > 0:
                st.session_state.corpus_message = f"Exported {count} poems to {output_path}"
                st.session_state.corpus_message_type = 'success'
                st.session_state.corpus_export_path = str(output_path)
            else:
                st.session_state.corpus_message = "No poems to export"
                st.session_state.corpus_message_type = 'error'
            st.rerun()
    
    with col2:
        st.subheader("JSONL Export")
        st.write("With metadata for ML")
        
        if st.button("Export as JSONL", key="btn_export_jsonl"):
            output_dir = Path("tajik_corpus/exports")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"corpus_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            
            count = export_jsonl_corpus(str(output_path))
            
            if count > 0:
                st.session_state.corpus_message = f"Exported {count} poems to {output_path}"
                st.session_state.corpus_message_type = 'success'
                st.session_state.corpus_export_path = str(output_path)
            else:
                st.session_state.corpus_message = "No poems to export"
                st.session_state.corpus_message_type = 'error'
            st.rerun()
    
    with col3:
        st.subheader("Git Export")
        st.write("For version control")
        
        if st.button("Export for Git", key="btn_export_git"):
            path = export_for_git()
            
            if path:
                st.session_state.corpus_message = f"Exported to {path}"
                st.session_state.corpus_message_type = 'success'
                st.session_state.corpus_export_path = path
            else:
                st.session_state.corpus_message = "Export failed"
                st.session_state.corpus_message_type = 'error'
            st.rerun()
    
    # Show last export path
    if st.session_state.corpus_export_path:
        st.markdown("---")
        st.write(f"**Last export:** `{st.session_state.corpus_export_path}`")
        
        # Try to provide download
        export_path = Path(st.session_state.corpus_export_path)
        if export_path.exists():
            with open(export_path, 'rb') as f:
                st.download_button(
                    label="Download Last Export",
                    data=f.read(),
                    file_name=export_path.name,
                    key="btn_download_export"
                )


def render_timeline_section():
    """Render timeline report"""
    st.header("Timeline Report")
    
    if not LIBRARY_AVAILABLE:
        st.warning("Library manager required for timeline")
        return
    
    if st.button("Generate Timeline Report", key="btn_timeline"):
        html = generate_timeline_html()
        
        if html:
            # Save HTML file
            output_dir = Path("tajik_corpus/exports")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            st.success(f"Timeline saved to {output_path}")
            
            # Provide download
            st.download_button(
                label="Download Timeline HTML",
                data=html,
                file_name="tajik_poetry_timeline.html",
                mime="text/html",
                key="btn_download_timeline"
            )
            
            # Show preview
            with st.expander("Preview"):
                st.components.v1.html(html, height=600, scrolling=True)
        else:
            st.error("Could not generate timeline")


def render_existing_corpus_info():
    """Show info about existing large corpus"""
    st.header("Existing Corpus Data")
    
    corpus_file = Path("data/tajik_corpus.txt")
    lexicon_file = Path("data/tajik_lexicon.json")
    
    if corpus_file.exists():
        size_mb = corpus_file.stat().st_size / (1024 * 1024)
        st.write(f"**Main Corpus:** `{corpus_file}` ({size_mb:.1f} MB)")
        
        # Count lines
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            st.write(f"Lines: {line_count:,}")
        except:
            pass
    
    if lexicon_file.exists():
        try:
            with open(lexicon_file, 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
            st.write(f"**Lexicon:** {len(lexicon):,} words")
        except:
            pass


# -------------------------------------------------------------------
# Main Page
# -------------------------------------------------------------------
def main():
    init_corpus_state()
    
    st.title("Corpus Management")
    st.markdown("Manage training data and exports")
    
    show_message()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Statistics",
        "Export",
        "Timeline",
        "Raw Corpus"
    ])
    
    with tab1:
        render_statistics()
    
    with tab2:
        render_export_section()
    
    with tab3:
        render_timeline_section()
    
    with tab4:
        render_existing_corpus_info()


# Run
main()
