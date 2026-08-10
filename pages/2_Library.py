#!/usr/bin/env python3
"""
Library Management Page
Manage poetry volumes with metadata - simplified batch workflow
"""

import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from extended_corpus_manager import TajikLibraryManager, VolumeMetadata, Genre
    LIBRARY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Library manager not available: {e}")
    LIBRARY_AVAILABLE = False

try:
    from corpus_manager import TajikCorpusManager
    CORPUS_AVAILABLE = True
except ImportError:
    CORPUS_AVAILABLE = False


def init_library_state():
    defaults = {
        'lib_contributions_loaded': False,
        'lib_contributions': [],
        'lib_batches': {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_existing_contributions() -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Load poems from the unified corpus, grouped by work.

    Batches used to mean "one upload file"; grouping by work is the
    meaningful unit now that every poem carries a source.
    """
    try:
        from corpus_core import Corpus
        corpus = Corpus()
        contributions = corpus.as_contributions()
    except Exception as e:
        logger.error(f"Could not load corpus: {e}")
        return [], {}

    batches: Dict[str, Dict] = {}
    for data in contributions:
        wid = data.get('upload_batch_id') or '_ungrouped_'
        batch = batches.setdefault(wid, {
            'poems': [],
            'source_filename': data.get('source_filename', 'Unknown'),
            'batch_id': wid,
        })
        batch['poems'].append(data)
    return contributions, batches

def save_contribution_metadata(filepath: str, metadata: Dict) -> bool:
    """Update metadata in existing contribution file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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


def delete_batch(batch_id: str, batches: Dict) -> Tuple[int, List[str]]:
    """Delete all poems in a batch"""
    if batch_id not in batches:
        return 0, [f"Batch '{batch_id}' not found"]
    
    batch = batches[batch_id]
    deleted_count = 0
    errors = []
    
    for poem in batch['poems']:
        filepath = poem.get('_filepath')
        if filepath and Path(filepath).exists():
            try:
                os.remove(filepath)
                deleted_count += 1
                logger.info(f"Deleted: {filepath}")
            except Exception as e:
                errors.append(f"Failed to delete {filepath}: {e}")
    
    # Clean up master corpus if exists
    master_corpus = Path("./tajik_corpus/corpus/master.json")
    if master_corpus.exists():
        try:
            with open(master_corpus, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            
            poems = corpus_data.get('poems', [])
            corpus_data['poems'] = [
                p for p in poems 
                if p.get('upload_batch_id') != batch_id
            ]
            
            if 'statistics' in corpus_data:
                corpus_data['statistics']['total_poems'] = len(corpus_data['poems'])
            
            with open(master_corpus, 'w', encoding='utf-8') as f:
                json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            errors.append(f"Master corpus update error: {e}")
    
    return deleted_count, errors


def display_batch_selector(batches: Dict, key_prefix: str = "batch", exclude_ungrouped: bool = False) -> Optional[str]:
    """Display batch selection dropdown"""
    if not batches:
        st.warning("No batches found.")
        return None
    
    options = []
    for batch_id, batch_data in batches.items():
        if exclude_ungrouped and batch_id == '_ungrouped_':
            continue
        if batch_id == '_ungrouped_':
            label = f"Individual poems ({len(batch_data['poems'])} poems)"
        else:
            label = f"{batch_data['source_filename']} ({len(batch_data['poems'])} poems)"
        options.append((batch_id, label))
    
    if not options:
        st.info("No batches available.")
        return None
    
    options.sort(key=lambda x: (x[0] == '_ungrouped_', x[1]))
    
    selected = st.selectbox(
        "Select Batch",
        options=[opt[0] for opt in options],
        format_func=lambda x: next(opt[1] for opt in options if opt[0] == x),
        key=f"{key_prefix}_selector"
    )
    
    return selected


def display_batch_info(batch_data: Dict):
    """Display information about a batch"""
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Source:** `{batch_data['source_filename']}`")
        st.write(f"**Poems:** {len(batch_data['poems'])}")
    with col2:
        # Show existing metadata if any
        first_poem = batch_data['poems'][0] if batch_data['poems'] else {}
        vol_meta = first_poem.get('volume_metadata', {})
        if vol_meta.get('author') or vol_meta.get('collection'):
            st.write(f"**Author:** {vol_meta.get('author', '-')}")
            st.write(f"**Collection:** {vol_meta.get('collection', '-')}")
    
    with st.expander("Preview poems"):
        for poem in batch_data['poems'][:10]:
            st.write(f"- {poem.get('title', 'Untitled')}")
        if len(batch_data['poems']) > 10:
            st.caption(f"... and {len(batch_data['poems']) - 10} more")


def display_browse_tab(contributions: List[Dict], batches: Dict):
    """Browse tab - overview of all batches"""
    if not contributions:
        st.info("No poems in library yet. Use the Analyze page to add poems.")
        return
    
    st.subheader(f"Library Overview ({len(contributions)} poems)")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        real_batches = len([b for b in batches if b != '_ungrouped_'])
        st.metric("Batches", real_batches)
    with col2:
        with_meta = sum(1 for c in contributions if c.get('volume_metadata', {}).get('author'))
        st.metric("With Metadata", with_meta)
    with col3:
        ungrouped = len(batches.get('_ungrouped_', {}).get('poems', []))
        st.metric("Ungrouped", ungrouped)
    
    st.markdown("---")
    
    # List batches
    for batch_id, batch_data in sorted(batches.items(), key=lambda x: x[0] == '_ungrouped_'):
        if batch_id == '_ungrouped_':
            title = "Individual Poems (no batch)"
        else:
            title = batch_data['source_filename']
        
        first_poem = batch_data['poems'][0] if batch_data['poems'] else {}
        vol_meta = first_poem.get('volume_metadata', {})
        has_metadata = vol_meta.get('author') or vol_meta.get('collection')
        status = "[complete]" if has_metadata else "[needs metadata]"
        
        with st.expander(f"{title} ({len(batch_data['poems'])} poems) {status}"):
            if batch_id != '_ungrouped_':
                st.caption(f"Batch ID: `{batch_id}`")
            
            if has_metadata:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Author:** {vol_meta.get('author', '-')}")
                with col2:
                    st.write(f"**Collection:** {vol_meta.get('collection', '-')}")
                with col3:
                    st.write(f"**Year:** {vol_meta.get('year', '-')}")
                with col4:
                    st.write(f"**Publisher:** {vol_meta.get('publisher', '-')}")
            
            st.markdown("**Poems:**")
            for poem in batch_data['poems'][:15]:
                st.write(f"- {poem.get('title', 'Untitled')}")
            if len(batch_data['poems']) > 15:
                st.caption(f"... and {len(batch_data['poems']) - 15} more")


def display_edit_metadata_tab(batches: Dict):
    """Edit metadata for a batch"""
    st.subheader("Edit Batch Metadata")
    st.info("Update or add metadata for an existing batch.")
    
    selected_batch_id = display_batch_selector(batches, key_prefix="edit")
    
    if not selected_batch_id:
        return
    
    batch_data = batches[selected_batch_id]
    
    st.markdown("---")
    display_batch_info(batch_data)
    st.markdown("---")
    
    # Get existing metadata
    first_poem = batch_data['poems'][0] if batch_data['poems'] else {}
    existing = first_poem.get('volume_metadata', {})
    
    st.markdown("### Metadata")
    
    col1, col2 = st.columns(2)
    with col1:
        author = st.text_input("Author", value=existing.get('author', '') or '', key="edit_author")
        collection = st.text_input("Collection/Volume", value=existing.get('collection', '') or '', key="edit_collection")
    with col2:
        year = st.number_input("Year", 1900, 2030, value=existing.get('year') or 2000, key="edit_year")
        publisher = st.text_input("Publisher", value=existing.get('publisher', '') or '', key="edit_publisher")
    
    st.markdown("---")
    
    if st.button("Save Metadata", type="primary", key="btn_save_metadata"):
        metadata = {
            'author': author if author else None,
            'collection': collection if collection else None,
            'year': year,
            'publisher': publisher if publisher else None,
        }
        
        success_count = 0
        for poem in batch_data['poems']:
            filepath = poem.get('_filepath')
            if filepath and save_contribution_metadata(filepath, metadata):
                success_count += 1
        
        if success_count > 0:
            st.success(f"Metadata saved for {success_count} poems!")
            st.session_state.lib_contributions_loaded = False
            st.rerun()
        else:
            st.error("Failed to save metadata.")


def delete_single_poem(filepath: str) -> bool:
    """Delete a single poem file"""
    try:
        if filepath and Path(filepath).exists():
            os.remove(filepath)
            logger.info(f"Deleted single poem: {filepath}")
            return True
    except Exception as e:
        logger.error(f"Error deleting {filepath}: {e}")
    return False


def display_delete_tab(batches: Dict):
    """Delete batch tab - now with ungrouped poem support"""
    st.subheader("Delete Poems")
    st.warning("Deleting permanently removes poems from the library.")
    
    # Handle ungrouped poems first
    ungrouped = batches.get('_ungrouped_', {}).get('poems', [])
    if ungrouped:
        st.markdown("### Ungrouped Poems")
        st.info(f"{len(ungrouped)} poem(s) without batch assignment can be deleted individually.")
        
        with st.expander("Manage ungrouped poems", expanded=True):
            for i, poem in enumerate(ungrouped):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i+1}. {poem.get('title', 'Untitled')[:50]}")
                with col2:
                    if st.button("Delete", key=f"del_ungrouped_{i}"):
                        if delete_single_poem(poem.get('_filepath', '')):
                            st.success(f"Deleted: {poem.get('title', 'Untitled')[:30]}")
                            st.session_state.lib_contributions_loaded = False
                            st.rerun()
                        else:
                            st.error("Failed to delete")
            
            st.markdown("---")
            confirm_all = st.checkbox("I confirm deletion of ALL ungrouped poems", key="confirm_all_ungrouped")
            if st.button("Delete ALL Ungrouped", key="btn_del_all_ungrouped", disabled=not confirm_all):
                deleted = sum(1 for p in ungrouped if delete_single_poem(p.get('_filepath', '')))
                if deleted > 0:
                    st.success(f"Deleted {deleted} ungrouped poems")
                    st.session_state.lib_contributions_loaded = False
                    st.rerun()
        
        st.markdown("---")
    
    # Regular batches
    st.markdown("### Batches")
    deletable = {k: v for k, v in batches.items() if k != '_ungrouped_'}
    
    if not deletable:
        st.info("No batches available to delete.")
        return
    
    selected_batch_id = display_batch_selector(deletable, key_prefix="delete", exclude_ungrouped=True)
    
    if not selected_batch_id:
        return
    
    batch_data = deletable[selected_batch_id]
    
    st.markdown("---")
    display_batch_info(batch_data)
    st.markdown("---")
    
    st.error(f"This will permanently delete **{len(batch_data['poems'])} poems**!")
    
    confirm = st.checkbox(
        f"I confirm I want to delete '{batch_data['source_filename']}'",
        key="delete_confirm"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Delete Batch", type="primary", disabled=not confirm, key="btn_delete"):
            with st.spinner("Deleting..."):
                deleted, errors = delete_batch(selected_batch_id, batches)
            
            if errors:
                for err in errors:
                    st.error(err)
            
            if deleted > 0:
                st.success(f"Deleted {deleted} poems")
                st.session_state.lib_contributions_loaded = False
                st.rerun()
    
    with col2:
        if st.button("Cancel", key="btn_cancel"):
            st.rerun()


def main():
    init_library_state()
    
    st.title("Library")
    st.markdown("Manage your poetry collection")
    st.markdown("---")
    
    if not CORPUS_AVAILABLE:
        st.error("Corpus manager not available.")
        st.stop()
    
    # Load data
    if not st.session_state.lib_contributions_loaded:
        contributions, batches = load_existing_contributions()
        st.session_state.lib_contributions = contributions
        st.session_state.lib_batches = batches
        st.session_state.lib_contributions_loaded = True
    
    contributions = st.session_state.lib_contributions
    batches = st.session_state.lib_batches
    
    # Reload button
    if st.button("Reload", key="btn_reload"):
        st.session_state.lib_contributions_loaded = False
        st.rerun()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Browse", "Edit Metadata", "Delete Batch"])
    
    with tab1:
        display_browse_tab(contributions, batches)
    
    with tab2:
        display_edit_metadata_tab(batches)
    
    with tab3:
        display_delete_tab(batches)


main()
