#!/usr/bin/env python3
"""
Library Management Page
Manage poetry volumes with metadata
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import library manager
try:
    from extended_corpus_manager import (
        TajikLibraryManager,
        VolumeMetadata,
        Genre,
        Period
    )
    LIBRARY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Library manager not available: {e}")
    LIBRARY_AVAILABLE = False

# Import simple corpus manager for existing contributions
try:
    from corpus_manager import TajikCorpusManager
    CORPUS_AVAILABLE = True
except ImportError:
    CORPUS_AVAILABLE = False


def init_library_state():
    """Initialize session state for library"""
    defaults = {
        'lib_selected_volume': None,
        'lib_edit_mode': False,
        'lib_metadata_saved': False,
        'lib_contributions_loaded': False,
        'lib_contributions': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_existing_contributions() -> List[Dict]:
    """Load existing contributions from simple corpus"""
    contributions = []
    corpus_dir = Path("./tajik_corpus/contributions")
    
    if corpus_dir.exists():
        for file in sorted(corpus_dir.glob("*.json")):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_filepath'] = str(file)
                    contributions.append(data)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
    
    return contributions


def save_contribution_metadata(filepath: str, metadata: Dict):
    """Update metadata in existing contribution file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update metadata fields
        if 'volume_metadata' not in data:
            data['volume_metadata'] = {}
        
        data['volume_metadata'].update(metadata)
        data['metadata']['last_modified'] = datetime.now().isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        return False


def display_metadata_form(contribution: Optional[Dict] = None) -> Dict:
    """Display metadata input form, return values"""
    
    # Get existing values if editing
    existing = {}
    if contribution and 'volume_metadata' in contribution:
        existing = contribution['volume_metadata']
    
    st.subheader("Volume Metadata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        author = st.text_input(
            "Author",
            value=existing.get('author', ''),
            key="meta_author",
            placeholder="e.g. Dilorom Soliboeva"
        )
        
        collection = st.text_input(
            "Collection / Volume Title",
            value=existing.get('collection', ''),
            key="meta_collection",
            placeholder="e.g. Tufonhoi sokit"
        )
    
    with col2:
        year = st.number_input(
            "Publication Year",
            min_value=1900,
            max_value=2030,
            value=existing.get('year', 2000),
            key="meta_year"
        )
        
        publisher = st.text_input(
            "Publisher",
            value=existing.get('publisher', ''),
            key="meta_publisher",
            placeholder="e.g. Adib"
        )
    
    return {
        'author': author,
        'collection': collection,
        'year': year,
        'publisher': publisher
    }


def display_contributions_table(contributions: List[Dict]):
    """Display table of existing contributions"""
    
    if not contributions:
        st.info("No contributions found. Use the Analyze page to add poems.")
        return
    
    st.subheader(f"Existing Contributions ({len(contributions)})")
    
    # Group by potential volumes (same metadata)
    groups = {}
    ungrouped = []
    
    for contrib in contributions:
        vol_meta = contrib.get('volume_metadata', {})
        if vol_meta.get('collection'):
            key = f"{vol_meta.get('author', 'Unknown')} - {vol_meta.get('collection')}"
            if key not in groups:
                groups[key] = []
            groups[key].append(contrib)
        else:
            ungrouped.append(contrib)
    
    # Display grouped
    if groups:
        st.markdown("### Volumes with Metadata")
        for group_name, items in groups.items():
            with st.expander(f"{group_name} ({len(items)} poems)"):
                vol_meta = items[0].get('volume_metadata', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Author:** {vol_meta.get('author', '-')}")
                with col2:
                    st.write(f"**Year:** {vol_meta.get('year', '-')}")
                with col3:
                    st.write(f"**Publisher:** {vol_meta.get('publisher', '-')}")
                with col4:
                    if st.button("Edit", key=f"edit_{group_name}"):
                        st.session_state.lib_selected_volume = group_name
                        st.session_state.lib_edit_mode = True
                        st.rerun()
    
    # Display ungrouped
    if ungrouped:
        st.markdown("### Poems Without Volume Metadata")
        st.warning(f"{len(ungrouped)} poems need metadata assignment")
        
        # Show table
        table_data = []
        for contrib in ungrouped[:20]:  # Limit display
            table_data.append({
                'ID': contrib.get('poem_id', '-'),
                'Title': contrib.get('title', '-')[:40],
                'Date': contrib.get('metadata', {}).get('submission_date', '-')[:10],
                'File': Path(contrib.get('_filepath', '')).name
            })
        
        st.dataframe(table_data, use_container_width=True)
        
        if len(ungrouped) > 20:
            st.caption(f"Showing 20 of {len(ungrouped)} poems")


def display_bulk_metadata_form(contributions: List[Dict]):
    """Form to apply metadata to multiple contributions at once"""
    
    st.subheader("Apply Metadata to Multiple Poems")
    st.info("This will apply the same metadata to all selected poems.")
    
    # Get ungrouped contributions
    ungrouped = [c for c in contributions if not c.get('volume_metadata', {}).get('collection')]
    
    if not ungrouped:
        st.success("All poems have metadata assigned.")
        return
    
    # Selection
    st.write(f"**{len(ungrouped)} poems without metadata**")
    
    apply_to_all = st.checkbox("Apply to all poems without metadata", value=True)
    
    if not apply_to_all:
        # Let user select specific poems
        selected_ids = st.multiselect(
            "Select poems",
            options=[c.get('poem_id', c.get('title', 'Unknown')) for c in ungrouped],
            default=[]
        )
    else:
        selected_ids = None  # Means all
    
    st.markdown("---")
    
    # Metadata form
    metadata = display_metadata_form()
    
    st.markdown("---")
    
    # Apply button
    if st.button("Apply Metadata", type="primary", key="btn_apply_metadata"):
        if not metadata['author'] or not metadata['collection']:
            st.error("Author and Collection are required.")
            return
        
        # Determine which contributions to update
        to_update = ungrouped if apply_to_all else [
            c for c in ungrouped 
            if c.get('poem_id', c.get('title')) in selected_ids
        ]
        
        # Update each contribution
        success_count = 0
        for contrib in to_update:
            filepath = contrib.get('_filepath')
            if filepath and save_contribution_metadata(filepath, metadata):
                success_count += 1
        
        if success_count > 0:
            st.session_state.lib_metadata_saved = True
            st.session_state.lib_contributions_loaded = False  # Force reload
            st.rerun()
        else:
            st.error("Failed to save metadata.")
    
    # Show success message
    if st.session_state.lib_metadata_saved:
        st.success("Metadata saved successfully!")
        st.session_state.lib_metadata_saved = False


def display_library_stats():
    """Display library statistics"""
    
    if not LIBRARY_AVAILABLE:
        return
    
    try:
        library = TajikLibraryManager()
        stats = library.get_statistics()
        
        if stats.get('total_volumes', 0) > 0:
            st.subheader("Library Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Volumes", stats.get('total_volumes', 0))
            with col2:
                st.metric("Poems", stats.get('total_poems', 0))
            with col3:
                st.metric("Authors", stats.get('authors_count', 0))
            with col4:
                year_range = f"{stats.get('publication_years', {}).get('min', '-')} - {stats.get('publication_years', {}).get('max', '-')}"
                st.metric("Year Range", year_range)
    except Exception as e:
        logger.error(f"Error loading library stats: {e}")


def main():
    init_library_state()
    
    st.title("Library Management")
    st.markdown("Manage poetry volumes and metadata")
    st.markdown("---")
    
    if not CORPUS_AVAILABLE:
        st.error("Corpus manager not available.")
        st.stop()
    
    # Load contributions
    if not st.session_state.lib_contributions_loaded:
        st.session_state.lib_contributions = load_existing_contributions()
        st.session_state.lib_contributions_loaded = True
    
    contributions = st.session_state.lib_contributions
    
    # Display stats
    display_library_stats()
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Browse", "Add Metadata", "Register Volume"])
    
    with tab1:
        # Reload button
        if st.button("Reload", key="btn_reload"):
            st.session_state.lib_contributions_loaded = False
            st.rerun()
        
        display_contributions_table(contributions)
    
    with tab2:
        display_bulk_metadata_form(contributions)
    
    with tab3:
        st.subheader("Register Complete Volume")
        st.info("This creates a formal library entry with full bibliographic data.")
        
        if not LIBRARY_AVAILABLE:
            st.warning("Extended library manager not available.")
        else:
            # Full volume registration form
            st.markdown("### Bibliographic Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                vol_author = st.text_input("Author Name", key="vol_author")
                vol_title = st.text_input("Volume Title", key="vol_title")
                vol_year = st.number_input("Publication Year", 1900, 2030, 2000, key="vol_year")
                vol_publisher = st.text_input("Publisher", key="vol_publisher")
            
            with col2:
                vol_city = st.text_input("City", key="vol_city")
                vol_pages = st.number_input("Total Pages", 0, 1000, 0, key="vol_pages")
                vol_genre = st.multiselect(
                    "Genres",
                    options=[g.value for g in Genre],
                    key="vol_genre"
                )
                vol_notes = st.text_area("Notes", key="vol_notes", height=100)
            
            st.markdown("---")
            
            # Select poems to include
            ungrouped = [c for c in contributions if not c.get('volume_metadata', {}).get('collection')]
            
            if ungrouped:
                st.markdown("### Select Poems for This Volume")
                
                include_all = st.checkbox("Include all unassigned poems", value=True, key="vol_include_all")
                
                if not include_all:
                    selected_poems = st.multiselect(
                        "Select poems",
                        options=[f"{c.get('poem_id', '-')}: {c.get('title', '-')[:30]}" for c in ungrouped],
                        key="vol_selected_poems"
                    )
                
                st.markdown("---")
                
                if st.button("Register Volume", type="primary", key="btn_register_volume"):
                    if not vol_author or not vol_title:
                        st.error("Author and Title are required.")
                    else:
                        try:
                            library = TajikLibraryManager()
                            
                            # Create metadata
                            metadata = VolumeMetadata(
                                author_name=vol_author,
                                volume_title=vol_title,
                                publication_year=vol_year,
                                publisher=vol_publisher if vol_publisher else None,
                                city=vol_city if vol_city else None,
                                pages=vol_pages if vol_pages > 0 else None,
                                notes=vol_notes if vol_notes else None
                            )
                            
                            # Get poems data
                            poems_to_register = ungrouped if include_all else [
                                c for c in ungrouped
                                if f"{c.get('poem_id', '-')}: {c.get('title', '-')[:30]}" in selected_poems
                            ]
                            
                            # Prepare poems data
                            poems_data = []
                            for contrib in poems_to_register:
                                poems_data.append({
                                    'content': contrib.get('raw_text', ''),
                                    'analysis': contrib.get('analysis', {})
                                })
                            
                            # Register volume
                            volume_id = library.register_volume(metadata, poems_data)
                            
                            # Also update simple corpus metadata
                            simple_metadata = {
                                'author': vol_author,
                                'collection': vol_title,
                                'year': vol_year,
                                'publisher': vol_publisher
                            }
                            
                            for contrib in poems_to_register:
                                filepath = contrib.get('_filepath')
                                if filepath:
                                    save_contribution_metadata(filepath, simple_metadata)
                            
                            st.success(f"Volume registered: {volume_id}")
                            st.session_state.lib_contributions_loaded = False
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error registering volume: {e}")
                            logger.exception("Volume registration failed")
            else:
                st.info("No unassigned poems available. Analyze poems first.")


# Run
main()
