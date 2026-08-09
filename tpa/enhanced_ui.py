#!/usr/bin/env python3
"""
Enhanced Streamlit UI for Tajik Poetry Analyzer
Production-ready interface with batch processing support
"""

import streamlit as st
import asyncio
import pandas as pd
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil
from datetime import datetime
import logging

# Import core modules
from secure_config import get_config, AppConfig
from improved_analyzer import ProductionAnalyzer, BatchConfig, BatchProgress
from analyzer import PoemData, validate_poem_data

logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="Tajik Poetry Analyzer v2.0",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class UIManager:
    """Manages UI state and operations"""
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize session state
        if 'analyzer' not in st.session_state:
            batch_config = BatchConfig(
                batch_size=self.config.processing.default_batch_size,
                max_concurrent=min(4, self.config.security.max_concurrent_analyses),
                memory_limit_mb=self.config.security.max_memory_usage_mb,
                enable_streaming=self.config.processing.enable_streaming
            )
            st.session_state.analyzer = ProductionAnalyzer(batch_config)
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = 'idle'
    
    def validate_files(self, uploaded_files) -> Dict[str, Any]:
        """Validate uploaded files"""
        valid_files = []
        invalid_files = []
        total_size = 0
        
        for uploaded_file in uploaded_files:
            file_size = len(uploaded_file.getvalue())
            total_size += file_size
            
            # Check individual file
            allowed, message = self.config.is_file_allowed(uploaded_file.name, file_size)
            
            if allowed:
                valid_files.append(uploaded_file)
            else:
                invalid_files.append({'name': uploaded_file.name, 'error': message})
        
        # Check batch limits
        batch_allowed, batch_message = self.config.is_batch_allowed(len(valid_files), total_size)
        
        return {
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'total_size': total_size,
            'batch_allowed': batch_allowed,
            'batch_message': batch_message
        }
    
    def save_uploaded_files(self, uploaded_files: List) -> List[Path]:
        """Save uploaded files to temporary directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix="tajik_analyzer_"))
        saved_paths = []

        try:
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name

                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                saved_paths.append(file_path)

            return saved_paths
        except Exception as e:
            # Cleanup on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def cleanup_temp_files(self, file_paths: List[Path]) -> None:
        """Clean up temporary files and directories"""
        for file_path in file_paths:
            try:
                # Get the parent temp directory
                temp_dir = file_path.parent
                if temp_dir.exists() and 'tajik_analyzer_' in str(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                    break  # Only need to delete once per batch
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")

def render_header():
    """Render application header"""
    st.title("📚 Tajik Poetry Analyzer v2.0")
    st.markdown("""
    **Advanced Scientific Analysis of Tajik Poetry with Batch Processing**
    
    This production-ready system provides comprehensive analysis including:
    • Prosodic and metrical analysis (ʿArūḍ system)
    • Semantic field mapping and isotopy detection  
    • Translation theoretical analysis (Ette/Bachmann-Medick)
    • Semiotic analysis (Lotman's cultural semiotics)
    • Multi-perspective literary assessment
    """)

def render_sidebar():
    """Render sidebar with configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        config = get_config()
        
        # Environment info
        st.info(f"""
        **Environment:** {config.environment.value}  
        **Debug Mode:** {'On' if config.debug_mode else 'Off'}
        """)
        
        # Processing settings
        st.subheader("Processing Settings")
        
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=config.processing.max_batch_size,
            value=config.processing.default_batch_size,
            help="Number of poems to process in each batch"
        )
        
        max_concurrent = st.slider(
            "Concurrent Analyses",
            min_value=1,
            max_value=config.security.max_concurrent_analyses,
            value=min(4, config.security.max_concurrent_analyses),
            help="Maximum number of simultaneous analyses"
        )
        
        enable_streaming = st.checkbox(
            "Enable Streaming",
            value=config.processing.enable_streaming,
            help="Use streaming processing for large files"
        )
        
        # Update analyzer config
        if 'analyzer' in st.session_state:
            st.session_state.analyzer.config.batch_size = batch_size
            st.session_state.analyzer.config.max_concurrent = max_concurrent
            st.session_state.analyzer.config.enable_streaming = enable_streaming
        
        st.divider()
        
        # System limits
        st.subheader("System Limits")
        st.text(f"Max file size: {config.security.max_file_size_mb}MB")
        st.text(f"Max batch size: {config.security.max_batch_size_mb}MB")
        st.text(f"Max files per batch: {config.security.max_files_per_batch}")
        
        st.divider()
        
        # Actions
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            if 'analysis_results' in st.session_state:
                st.session_state.analysis_results = []
            st.success("Cache cleared!")
        
        if st.button("📊 View System Stats"):
            render_system_stats()

def render_system_stats():
    """Render system statistics"""
    import psutil
    
    with st.expander("System Statistics", expanded=True):
        # Memory usage
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent:.1f}%")
        st.metric("Available Memory", f"{memory.available / 1024**3:.1f} GB")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        st.metric("Disk Usage", f"{disk.percent:.1f}%")

def render_file_upload():
    """Render file upload interface"""
    st.header("1. 📁 File Upload")
    
    ui_manager = UIManager()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload poetry files",
        type=['txt', 'docx', 'pdf', 'rtf'],
        accept_multiple_files=True,
        help="You can upload multiple files for batch processing"
    )
    
    if uploaded_files:
        # Validate files
        validation_result = ui_manager.validate_files(uploaded_files)
        
        # Display validation results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Valid Files", len(validation_result['valid_files']))
        
        with col2:
            st.metric("Invalid Files", len(validation_result['invalid_files']))
        
        with col3:
            st.metric("Total Size", f"{validation_result['total_size'] / 1024**2:.1f} MB")
        
        # Show invalid files if any
        if validation_result['invalid_files']:
            st.error("❌ Some files are invalid:")
            for invalid_file in validation_result['invalid_files']:
                st.write(f"• **{invalid_file['name']}**: {invalid_file['error']}")
        
        # Check batch limits
        if not validation_result['batch_allowed']:
            st.error(f"❌ Batch validation failed: {validation_result['batch_message']}")
            return None
        
        # Show valid files
        if validation_result['valid_files']:
            st.success(f"✅ {len(validation_result['valid_files'])} files ready for processing")
            
            # File details
            with st.expander("File Details"):
                file_data = []
                for file in validation_result['valid_files']:
                    file_data.append({
                        'Name': file.name,
                        'Size (KB)': f"{len(file.getvalue()) / 1024:.1f}",
                        'Type': file.type or 'text/plain'
                    })
                
                st.dataframe(pd.DataFrame(file_data), hide_index=True)
            
            return validation_result['valid_files']
    
    return None

def render_processing_interface(valid_files: List):
    """Render processing interface"""
    st.header("2. 🚀 Analysis Processing")
    
    if not valid_files:
        st.info("Upload files to start processing")
        return
    
    # Processing options
    col1, col2 = st.columns(2)
    
    with col1:
        process_individually = st.checkbox(
            "Process files individually",
            value=False,
            help="Process each file separately vs. as a single batch"
        )
    
    with col2:
        save_intermediate = st.checkbox(
            "Save intermediate results",
            value=True,
            help="Save results after each file/batch"
        )
    
    # Start processing
    if st.button("🚀 Start Analysis", type="primary"):
        st.session_state.processing_status = 'running'
        
        # Save files to temporary location
        ui_manager = UIManager()
        temp_paths = ui_manager.save_uploaded_files(valid_files)
        
        try:
            # Process files
            results = asyncio.run(process_files_async(temp_paths, process_individually))

            # Store results
            st.session_state.analysis_results = results
            st.session_state.processing_status = 'completed'

            st.success("✅ Processing completed!")

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            st.session_state.processing_status = 'failed'
            logger.error(f"Processing error: {e}")

        finally:
            # Always cleanup temp files
            try:
                ui_manager.cleanup_temp_files(temp_paths)
                logger.debug("Temp files cleaned up successfully")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {cleanup_error}")

        if st.session_state.processing_status == 'completed':
            st.rerun()

async def process_files_async(file_paths: List[Path], individually: bool = False) -> List[Dict]:
    """Process files asynchronously with progress tracking"""
    analyzer = st.session_state.analyzer
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress: BatchProgress):
        progress_bar.progress(progress.completion_rate)
        status_text.text(
            f"Processing: {progress.processed_poems}/{progress.total_poems} poems "
            f"(Success rate: {progress.success_rate:.1%})"
        )
    
    try:
        if individually:
            # Process each file separately
            all_results = []
            for i, file_path in enumerate(file_paths):
                status_text.text(f"Processing file {i+1}/{len(file_paths)}: {file_path.name}")
                
                file_results = await analyzer.analyze_large_file(file_path, update_progress)
                all_results.extend(file_results['results'])
            
            return all_results
        else:
            # Process all files as batch
            batch_results = await analyzer.analyze_batch_files(file_paths, update_progress)
            return batch_results['results']
    
    finally:
        progress_bar.empty()
        status_text.empty()

def render_results():
    """Render analysis results"""
    if not st.session_state.analysis_results:
        return
    
    st.header("3. 📊 Analysis Results")
    
    results = st.session_state.analysis_results
    successful_results = [r for r in results if r.get('success', False)]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Poems", len(results))
    
    with col2:
        st.metric("Successful", len(successful_results))
    
    with col3:
        success_rate = len(successful_results) / len(results) if results else 0
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col4:
        if successful_results:
            avg_quality = sum(
                r['analysis'].quality_metrics.get('quality_score', 0)
                for r in successful_results
                if 'analysis' in r and hasattr(r['analysis'], 'quality_metrics')
            ) / len(successful_results)
            st.metric("Avg Quality", f"{avg_quality:.2f}")
    
    # Results table
    if successful_results:
        st.subheader("Analysis Summary")
        
        # Create summary dataframe
        summary_data = []
        for result in successful_results:
            analysis = result.get('analysis')
            poem = result.get('poem')

            if analysis and poem:
                # quality_metrics is a dict
                quality_score = analysis.quality_metrics.get('quality_score', 0) if isinstance(analysis.quality_metrics, dict) else 0

                summary_data.append({
                    'Title': poem.title[:50],
                    'Author': poem.author or 'Unknown',
                    'Lines': analysis.structural.lines,
                    'Meter': analysis.structural.aruz_analysis.identified_meter,
                    'Rhyme Pattern': analysis.structural.rhyme_pattern,
                    'Lexical Diversity': f"{analysis.content.lexical_diversity:.3f}",
                    'Quality Score': f"{quality_score:.2f}"
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    # Detailed results
    render_detailed_results(successful_results)

def render_detailed_results(results: List[Dict]):
    """Render detailed analysis results"""
    if not results:
        return
    
    st.subheader("Detailed Analysis")
    
    # Select poem for detailed view
    poem_titles = [f"{i+1}. {r['poem'].title[:50]}" for i, r in enumerate(results)]
    
    selected_idx = st.selectbox(
        "Select poem for detailed analysis:",
        range(len(poem_titles)),
        format_func=lambda x: poem_titles[x]
    )
    
    if selected_idx is not None:
        result = results[selected_idx]
        analysis = result['analysis']
        poem = result['poem']
        
        # Poem metadata
        with st.expander("📋 Poem Metadata", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Title:** {poem.title}")
                st.write(f"**Author:** {poem.author or 'Unknown'}")
            
            with col2:
                st.write(f"**Year:** {poem.publication_year or 'N/A'}")
                st.write(f"**Collection:** {poem.collection or 'N/A'}")
            
            with col3:
                st.write(f"**Language:** {poem.language_variant}")
                quality = analysis.quality_metrics.get('reliability', 'unknown') if isinstance(analysis.quality_metrics, dict) else 'unknown'
                st.write(f"**Quality:** {quality}")
        
        # Structural analysis
        with st.expander("🏗️ Structural Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Meter:** {analysis.structural.aruz_analysis.identified_meter}")
                st.write(f"**Confidence:** {analysis.structural.meter_confidence.value}")
                st.write(f"**Lines:** {analysis.structural.lines}")
            
            with col2:
                st.write(f"**Rhyme Pattern:** {analysis.structural.rhyme_pattern}")
                st.write(f"**Form:** {analysis.structural.stanza_structure}")
                st.write(f"**Avg Syllables:** {analysis.structural.avg_syllables:.1f}")
        
        # Content analysis
        with st.expander("📝 Content Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Lexical Diversity:** {analysis.content.lexical_diversity:.3f}")
                st.write(f"**Style:** {analysis.content.stylistic_register}")
                st.write(f"**Neologisms:** {len(analysis.content.neologisms)}")
            
            with col2:
                # Theme distribution chart
                if analysis.content.theme_distribution:
                    theme_data = {k: v for k, v in analysis.content.theme_distribution.items() if v > 0}
                    if theme_data:
                        st.write("**Theme Distribution:**")
                        st.bar_chart(pd.Series(theme_data))
        
        # Literary assessment
        with st.expander("🎭 Literary Assessment"):
            assessment = analysis.literary
            
            assessment_data = {
                'German Perspective': assessment.german_perspective,
                'Persian Tradition': assessment.persian_tradition,
                'Tajik Elements': assessment.tajik_elements,
                'Modernist Features': assessment.modernist_features,
                'Postcolonial Markers': assessment.postcolonial_markers
            }
            
            st.bar_chart(pd.Series(assessment_data))

def render_export_options():
    """Render export options"""
    if not st.session_state.analysis_results:
        return
    
    st.header("4. 📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to Excel"):
            export_to_excel()
    
    with col2:
        if st.button("📄 Export to JSON"):
            export_to_json()
    
    with col3:
        if st.button("📋 Export to CSV"):
            export_to_csv()

def export_to_excel():
    """Export results to Excel"""
    try:
        results = st.session_state.analysis_results
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            st.error("No successful results to export")
            return
        
        # Prepare data for Excel export
        excel_data = []
        for result in successful_results:
            excel_data.append({
                'poem_id': result.get('poem_id', ''),
                'title': result['poem'].title,
                'content': result['poem'].content,
                'analysis': result['analysis'],
                'validation': result['analysis'].quality_metrics
            })
        
        # Generate Excel file
        analyzer = st.session_state.analyzer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tajik_poetry_analysis_{timestamp}.xlsx"
        
        analyzer._save_as_excel(excel_data, Path(filename))
        
        # Download button
        with open(filename, 'rb') as f:
            st.download_button(
                "⬇️ Download Excel Report",
                f.read(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Cleanup
        Path(filename).unlink(missing_ok=True)
        st.success("Excel export completed!")
        
    except Exception as e:
        st.error(f"Export failed: {e}")

def export_to_json():
    """Export results to JSON"""
    try:
        results = st.session_state.analysis_results
        analyzer = st.session_state.analyzer
        
        # Serialize results
        serialized_results = []
        for result in results:
            if result.get('success', False):
                serialized_results.append(analyzer._serialize_result(result))
        
        # Create JSON
        json_data = json.dumps(serialized_results, indent=2, ensure_ascii=False, default=str)
        
        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tajik_poetry_analysis_{timestamp}.json"
        
        st.download_button(
            "⬇️ Download JSON Results",
            json_data,
            file_name=filename,
            mime="application/json"
        )
        
        st.success("JSON export completed!")
        
    except Exception as e:
        st.error(f"Export failed: {e}")

def export_to_csv():
    """Export results to CSV"""
    try:
        results = st.session_state.analysis_results
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            st.error("No successful results to export")
            return
        
        # Create CSV data
        csv_data = []
        for result in successful_results:
            analysis = result['analysis']
            poem = result['poem']
            
            csv_data.append({
                'poem_id': result.get('poem_id', ''),
                'title': poem.title,
                'author': poem.author or '',
                'lines': analysis.structural.lines,
                'meter': analysis.structural.aruz_analysis.identified_meter,
                'rhyme_pattern': analysis.structural.rhyme_pattern,
                'lexical_diversity': analysis.content.lexical_diversity,
                'quality_score': analysis.quality_metrics.get('quality_score', 0) if isinstance(analysis.quality_metrics, dict) else 0
            })
        
        # Convert to DataFrame and CSV
        df = pd.DataFrame(csv_data)
        csv_string = df.to_csv(index=False)
        
        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tajik_poetry_analysis_{timestamp}.csv"
        
        st.download_button(
            "⬇️ Download CSV Results",
            csv_string,
            file_name=filename,
            mime="text/csv"
        )
        
        st.success("CSV export completed!")
        
    except Exception as e:
        st.error(f"Export failed: {e}")

def main():
    """Main UI function"""
    try:
        # Initialize UI manager
        ui_manager = UIManager()
        
        # Render UI components
        render_header()
        render_sidebar()
        
        # Main content
        valid_files = render_file_upload()
        
        if valid_files:
            render_processing_interface(valid_files)
        
        # Show results if available
        render_results()
        render_export_options()
        
        # Footer
        st.divider()
        st.markdown("""
        **Tajik Poetry Analyzer v2.0** - Advanced Scientific Analysis  
        Developed with ❤️ for Tajik literature research
        """)
        
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"UI error: {e}")

if __name__ == "__main__":
    main()
