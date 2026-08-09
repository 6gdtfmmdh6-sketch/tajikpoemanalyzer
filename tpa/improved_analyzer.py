#!/usr/bin/env python3
"""
Production-Ready Tajik Poetry Analyzer with Batch Processing
Enhanced with async processing, memory management, and security features
"""

import asyncio
import aiofiles
import logging
import threading
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
import json
from collections import defaultdict
import sqlite3
from contextlib import asynccontextmanager
import concurrent.futures
import tempfile
import hashlib

# Import the original analyzer components
from analyzer import (
    TajikPoemAnalyzer, PoemData, AnalysisConfig, ComprehensiveAnalysis,
    validate_poem_data, QualityValidator, ExcelReporter
)
from secure_config import AppConfig, get_config

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_text_length: int = 1_000_000
    batch_size: int = 50
    max_concurrent: int = 4
    memory_limit_mb: int = 2048
    timeout_per_poem: int = 30
    enable_streaming: bool = True
    chunk_size: int = 8192
    auto_save_interval: int = 100
    enable_checkpoints: bool = True
    
    def __post_init__(self):
        """Validate batch configuration"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if self.memory_limit_mb < 256:
            raise ValueError("memory_limit_mb must be at least 256")

@dataclass
class BatchProgress:
    """Tracks batch processing progress"""
    total_poems: int = 0
    processed_poems: int = 0
    successful_poems: int = 0
    failed_poems: int = 0
    start_time: float = 0
    current_time: float = 0
    
    @property
    def completion_rate(self) -> float:
        """Get completion percentage"""
        if self.total_poems == 0:
            return 0.0
        return self.processed_poems / self.total_poems
    
    @property
    def success_rate(self) -> float:
        """Get success percentage"""
        if self.processed_poems == 0:
            return 0.0
        return self.successful_poems / self.processed_poems
    
    @property
    def estimated_time_remaining(self) -> float:
        """Estimate remaining time in seconds"""
        if self.processed_poems == 0:
            return 0.0
        
        elapsed = self.current_time - self.start_time
        rate = self.processed_poems / elapsed
        
        if rate <= 0:
            return 0.0
        
        remaining_poems = self.total_poems - self.processed_poems
        return remaining_poems / rate

class MemoryManager:
    """Manages memory usage during batch processing"""
    
    def __init__(self, max_memory_mb: int):
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.logger = logging.getLogger(f"{__name__}.MemoryManager")
    
    def check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss
            
            if current_memory > self.max_memory_bytes:
                self.logger.warning(
                    f"Memory usage {current_memory / 1024 / 1024:.1f}MB "
                    f"exceeds limit of {self.max_memory_mb}MB"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking memory: {e}")
            return True  # Assume OK if we can't check
    
    def force_cleanup(self) -> None:
        """Force garbage collection"""
        gc.collect()
        self.logger.debug("Forced garbage collection")

class StreamingTextProcessor:
    """Processes large text files in streaming fashion"""

    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(f"{__name__}.StreamingTextProcessor")

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet"""
        import chardet

        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']

                self.logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")

                # Fallback to utf-8 if confidence is too low or encoding is None
                if not encoding or confidence < 0.7:
                    self.logger.warning(f"Low confidence encoding detection ({confidence:.2f}), using UTF-8")
                    return 'utf-8'

                # Normalize encoding name
                encoding = encoding.lower()
                if 'utf-8' in encoding or 'utf8' in encoding:
                    return 'utf-8'
                elif 'windows-1251' in encoding or 'cp1251' in encoding:
                    return 'windows-1251'
                elif 'iso-8859' in encoding:
                    return 'iso-8859-1'

                return encoding

        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using UTF-8")
            return 'utf-8'

    async def read_large_file(self, file_path: Path) -> AsyncGenerator[str, None]:
        """Read files of various formats including PDF"""
        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            async for chunk in self._read_pdf_file(file_path):
                yield chunk
        elif suffix == '.docx':
            async for chunk in self._read_docx_file(file_path):
                yield chunk
        else:
            async for chunk in self._read_text_file(file_path):
                yield chunk

    async def _read_text_file(self, file_path: Path) -> AsyncGenerator[str, None]:
        """Read text file in chunks with automatic encoding detection"""
        try:
            # Detect encoding first
            encoding = self._detect_encoding(file_path)
            self.logger.info(f"Reading {file_path.name} with encoding: {encoding}")

            # Try to open with detected encoding
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding, errors='strict') as f:
                    while True:
                        chunk = await f.read(self.chunk_size)
                        if not chunk:
                            break
                        yield chunk
            except UnicodeDecodeError as ude:
                # If strict fails, try with error handling
                self.logger.warning(f"Encoding error with {encoding}, trying with error replacement")
                async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    while True:
                        chunk = await f.read(self.chunk_size)
                        if not chunk:
                            break
                        yield chunk

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise

    async def _read_pdf_file(self, file_path: Path) -> AsyncGenerator[str, None]:
        """Extract text from PDF using PyPDF2, with OCR fallback for scanned PDFs"""
        try:
            import PyPDF2

            text_content = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)

                self.logger.info(f"Reading PDF with {len(pdf_reader.pages)} pages")

                # Try extracting text from first page to check if it's a scanned PDF
                first_page_text = pdf_reader.pages[0].extract_text() if len(pdf_reader.pages) > 0 else ""

                # If first page has very little text, it might be a scanned PDF
                if len(first_page_text.strip()) < 50 and len(pdf_reader.pages) > 0:
                    self.logger.info("PDF appears to be scanned (low text content), attempting OCR")
                    try:
                        ocr_text = await self._try_ocr_extraction(file_path)
                        if ocr_text:
                            yield ocr_text
                            return
                        else:
                            self.logger.warning("OCR extraction returned no text, falling back to PyPDF2")
                    except Exception as ocr_error:
                        self.logger.warning(f"OCR extraction failed: {ocr_error}, falling back to PyPDF2")

                # Regular PyPDF2 extraction
                for page_num, page in enumerate(pdf_reader.pages):
                    # Extract text with layout preservation
                    text = page.extract_text()

                    # Clean PDF artifacts
                    text = self._clean_pdf_text(text)

                    if text.strip():
                        text_content.append(text)

                    # Yield in chunks for streaming (every 5 pages)
                    if len(text_content) >= 5:
                        yield '\n\n'.join(text_content)
                        text_content = []

                # Yield remaining content
                if text_content:
                    yield '\n\n'.join(text_content)

        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            raise

    async def _read_docx_file(self, file_path: Path) -> AsyncGenerator[str, None]:
        """Extract text from DOCX using python-docx"""
        try:
            from docx import Document

            doc = Document(file_path)
            text_content = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    text_content.append(text)

                    # Yield in chunks
                    if len(text_content) >= 20:
                        yield '\n'.join(text_content)
                        text_content = []

            # Yield remaining content
            if text_content:
                yield '\n'.join(text_content)

        except Exception as e:
            self.logger.error(f"DOCX extraction failed: {e}")
            raise

    async def _try_ocr_extraction(self, file_path: Path) -> Optional[str]:
        """Try to extract text using OCR for scanned PDFs"""
        try:
            from ocr_processor import TajikOCREngine

            self.logger.info("Attempting OCR extraction for scanned PDF")
            ocr_engine = TajikOCREngine(max_workers=2)
            text = await ocr_engine.extract_text_from_scanned_pdf(file_path)
            return text

        except ImportError as e:
            self.logger.warning(f"OCR not available: {e}")
            self.logger.warning("Install OCR dependencies: pip install pytesseract pdf2image")
            return None
        except Exception as e:
            self.logger.error(f"OCR extraction error: {e}")
            return None

    def _clean_pdf_text(self, text: str) -> str:
        """Clean PDF-extracted text"""
        # Remove ligatures and correct line breaks
        replacements = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            '-\n': '',  # Remove soft hyphens
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Normalize Arabic/Persian ligatures
        arabic_ligatures = {
            'ﻻ': 'لا',
            'ﷲ': 'الله',
            'ﻵ': 'لآ',
            'ﻷ': 'لأ',
            'ﻹ': 'لإ',
        }

        for ligature, expanded in arabic_ligatures.items():
            text = text.replace(ligature, expanded)

        return text
    
    async def split_text_streaming(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """Split text stream into poems with improved detection"""
        buffer = ""

        async for chunk in text_stream:
            buffer += chunk

        # Log for debugging
        self.logger.info(f"Processing text buffer: {len(buffer)} characters")

        import re

        # Strategy 1: Try separator patterns (more specific)
        separator_patterns = [
            r'\n\*{3,}\n',      # *** or more
            r'\n-{3,}\n',       # --- or more
            r'\n={3,}\n',       # === or more
            r'\n#{2,}\s*\n',    # ## or more
        ]

        poems_found = []

        for pattern in separator_patterns:
            potential_poems = re.split(pattern, buffer)
            if len(potential_poems) > 1:
                self.logger.debug(f"Pattern {pattern}: found {len(potential_poems)} sections")
                for poem_text in potential_poems:
                    poem_text = poem_text.strip()
                    if len(poem_text) > 50 and poem_text not in poems_found:
                        poems_found.append(poem_text)
                        yield poem_text
                # If found multiple poems, we're done
                if len(poems_found) > 1:
                    self.logger.info(f"Split into {len(poems_found)} poems using separators")
                    return

        # Strategy 2: Try 3+ newlines
        if not poems_found:
            self.logger.debug("Trying 3+ newline split")
            potential_poems = re.split(r'\n\s*\n\s*\n+', buffer)
            for poem_text in potential_poems:
                poem_text = poem_text.strip()
                if len(poem_text) > 50 and poem_text not in poems_found:
                    poems_found.append(poem_text)
                    yield poem_text

        # Strategy 3: If still nothing, try 2 newlines (more aggressive)
        if not poems_found:
            self.logger.debug("Trying 2 newline split")
            potential_poems = re.split(r'\n\s*\n', buffer)
            for poem_text in potential_poems:
                poem_text = poem_text.strip()
                # Be more strict on length for 2-newline split
                if len(poem_text) > 100:
                    poems_found.append(poem_text)
                    yield poem_text

        # If still nothing, yield entire buffer
        if not poems_found and len(buffer.strip()) > 50:
            self.logger.warning("No splitting patterns found, treating as single poem")
            yield buffer.strip()
        else:
            self.logger.info(f"Total poems found: {len(poems_found)}")

class EnhancedCorpusManager:
    """Enhanced corpus manager with batch operations"""
    
    def __init__(self, db_path: str, config: BatchConfig):
        self.db_path = db_path
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EnhancedCorpusManager")
        self._lock = threading.RLock()
        self._create_enhanced_tables()
    
    def _create_enhanced_tables(self):
        """Create enhanced database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables with additional indexes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS authors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    birth_year INTEGER,
                    death_year INTEGER,
                    country TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS works (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT UNIQUE,
                    publication_year INTEGER,
                    publisher TEXT,
                    collection TEXT,
                    language_variant TEXT DEFAULT 'tajik',
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(author_id) REFERENCES authors(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    work_id INTEGER NOT NULL,
                    analysis_data TEXT NOT NULL,
                    quality_score REAL,
                    processing_time REAL,
                    analyzer_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(work_id) REFERENCES works(id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_works_hash ON works(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_works_author ON works(author_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_work ON analysis_results(work_id)")
    
    def _get_connection(self):
        """Get database connection with enhanced settings"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
            check_same_thread=False
        )
        
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        
        return conn
    
    def add_poems_batch(self, poems_data: List[tuple]) -> Dict[str, int]:
        """Add multiple poems with analysis results in batch"""
        stats = {'added': 0, 'updated': 0, 'skipped': 0, 'errors': 0}
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for poem_data, analysis_result in poems_data:
                    try:
                        # Calculate content hash for deduplication
                        content_hash = hashlib.sha256(
                            poem_data.content.encode('utf-8')
                        ).hexdigest()
                        
                        # Check if content already exists
                        cursor.execute(
                            "SELECT id FROM works WHERE content_hash = ?",
                            (content_hash,)
                        )
                        existing = cursor.fetchone()
                        
                        if existing:
                            stats['skipped'] += 1
                            continue
                        
                        # Add or get author
                        cursor.execute(
                            "INSERT OR IGNORE INTO authors (name) VALUES (?)",
                            (poem_data.author or "Unknown",)
                        )
                        cursor.execute(
                            "SELECT id FROM authors WHERE name = ?",
                            (poem_data.author or "Unknown",)
                        )
                        author_id = cursor.fetchone()[0]
                        
                        # Add work
                        cursor.execute("""
                            INSERT INTO works (
                                author_id, title, content, content_hash,
                                publication_year, publisher, collection,
                                language_variant
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            author_id,
                            poem_data.title,
                            poem_data.content,
                            content_hash,
                            poem_data.publication_year,
                            poem_data.publisher,
                            poem_data.collection,
                            poem_data.language_variant
                        ))
                        
                        work_id = cursor.lastrowid
                        
                        # Add analysis result if available
                        if analysis_result:
                            quality_score = analysis_result.quality_metrics.get('quality_score', 0.0)
                            
                            cursor.execute("""
                                INSERT INTO analysis_results (
                                    work_id, analysis_data, quality_score,
                                    analyzer_version
                                ) VALUES (?, ?, ?, ?)
                            """, (
                                work_id,
                                json.dumps(asdict(analysis_result), default=str),
                                quality_score,
                                "2.0.0"
                            ))
                        
                        stats['added'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error adding poem {poem_data.title}: {e}")
                        stats['errors'] += 1
                        continue
                
                conn.commit()
        
        return stats
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM works")
            total_works = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM authors")
            total_authors = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM analysis_results")
            analyzed_works = cursor.fetchone()[0]
            
            # Content size
            cursor.execute("SELECT SUM(LENGTH(content)) FROM works")
            total_content_bytes = cursor.fetchone()[0] or 0
            
            return {
                'total_works': total_works,
                'total_authors': total_authors,
                'analyzed_works': analyzed_works,
                'total_content_mb': total_content_bytes / 1024 / 1024,
                'analysis_coverage': analyzed_works / max(total_works, 1)
            }

class BatchAnalysisEngine:
    """Core batch analysis engine with async processing"""
    
    def __init__(self, analyzer: TajikPoemAnalyzer, config: BatchConfig):
        self.analyzer = analyzer
        self.config = config
        self.memory_manager = MemoryManager(config.memory_limit_mb)
        self.logger = logging.getLogger(f"{__name__}.BatchAnalysisEngine")
        
        # Create thread pool for CPU-bound tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_concurrent
        )
    
    async def process_batch_async(
        self,
        poems: List[PoemData],
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> Dict[str, Any]:
        """Process batch of poems asynchronously"""
        progress = BatchProgress(
            total_poems=len(poems),
            start_time=time.time()
        )
        
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def analyze_poem_with_semaphore(poem: PoemData) -> Dict[str, Any]:
            async with semaphore:
                return await self._analyze_single_poem_async(poem)
        
        # Create tasks for all poems
        tasks = [analyze_poem_with_semaphore(poem) for poem in poems]
        
        # Process tasks with progress updates
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                
                # Update progress
                progress.processed_poems += 1
                progress.current_time = time.time()
                
                if result.get('success', False):
                    progress.successful_poems += 1
                else:
                    progress.failed_poems += 1
                
                # Call progress callback
                if progress_callback:
                    progress_callback(progress)
                
                # Check memory usage
                if not self.memory_manager.check_memory():
                    self.memory_manager.force_cleanup()
                
            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                progress.failed_poems += 1
        
        return {
            'results': results,
            'progress': progress,
            'summary': {
                'total_poems': progress.total_poems,
                'successful': progress.successful_poems,
                'failed': progress.failed_poems,
                'success_rate': progress.success_rate,
                'processing_time': progress.current_time - progress.start_time
            }
        }
    
    async def _analyze_single_poem_async(self, poem: PoemData) -> Dict[str, Any]:
        """Analyze single poem asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run analysis in thread pool
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._analyze_single_poem_safe,
                poem
            )
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing poem {poem.poem_id}: {e}")
            return {
                'success': False,
                'poem_id': poem.poem_id,
                'error': str(e),
                'processing_time': 0
            }
    
    def _analyze_single_poem_safe(self, poem: PoemData) -> Dict[str, Any]:
        """Safely analyze single poem with timeout and error handling"""
        start_time = time.time()
        
        try:
            # Validate poem
            errors = validate_poem_data(poem)
            if errors:
                return {
                    'success': False,
                    'poem_id': poem.poem_id,
                    'errors': errors,
                    'processing_time': time.time() - start_time
                }
            
            # Analyze poem
            analysis = self.analyzer.analyze_poem(poem)
            
            return {
                'success': True,
                'poem_id': poem.poem_id,
                'poem': poem,
                'analysis': analysis,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'poem_id': poem.poem_id,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

class ProductionAnalyzer:
    """Production-ready analyzer with batch processing capabilities"""

    def __init__(self, config: BatchConfig, enable_intertextuality: bool = True):
        self.config = config
        self.core_analyzer = TajikPoemAnalyzer()
        self.batch_engine = BatchAnalysisEngine(self.core_analyzer, config)
        self.text_processor = StreamingTextProcessor(config.chunk_size)
        self.logger = logging.getLogger(f"{__name__}.ProductionAnalyzer")

        # Initialize intertextuality analyzer if enabled
        self.enable_intertextuality = enable_intertextuality
        self.intertextuality_analyzer = None
        if enable_intertextuality:
            try:
                from intertextuality_analyzer import ClassicalCorpus
                self.intertextuality_analyzer = ClassicalCorpus()
                corpus_stats = self.intertextuality_analyzer.get_corpus_statistics()
                self.logger.info(f"Intertextuality analysis enabled: {corpus_stats}")
            except Exception as e:
                self.logger.warning(f"Could not initialize intertextuality analyzer: {e}")
                self.enable_intertextuality = False
    
    async def analyze_large_file(
        self,
        file_path: Path,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> Dict[str, Any]:
        """Analyze large file with streaming processing"""
        self.logger.info(f"Starting analysis of large file: {file_path}")
        
        try:
            # Stream and split text
            text_stream = self.text_processor.read_large_file(file_path)
            poem_texts = self.text_processor.split_text_streaming(text_stream)
            
            # Convert to PoemData objects
            poems = []
            i = 0
            async for poem_text in poem_texts:
                poem = PoemData(
                    title=f"Poem {i+1} from {file_path.name}",
                    content=poem_text,
                    poem_id=f"{file_path.stem}_{i+1:03d}"
                )
                poems.append(poem)
                i += 1
            
            if not poems:
                raise ValueError("No poems found in file")

            # Process batch with optional intertextuality analysis
            results = await self.batch_engine.process_batch_async(poems, progress_callback)

            # Add intertextual analysis if enabled
            if self.enable_intertextuality and self.intertextuality_analyzer:
                results = await self._add_intertextual_analysis(results)

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            raise
    
    async def analyze_batch_files(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> Dict[str, Any]:
        """Analyze multiple files as a batch"""
        self.logger.info(f"Starting batch analysis of {len(file_paths)} files")
        
        all_results = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            self.logger.info(f"Processing file {i+1}/{total_files}: {file_path.name}")
            
            try:
                file_results = await self.analyze_large_file(file_path, progress_callback)
                all_results.extend(file_results['results'])
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        # Compile final results
        successful = len([r for r in all_results if r.get('success', False)])
        failed = len(all_results) - successful
        
        return {
            'results': all_results,
            'files_processed': total_files,
            'progress': BatchProgress(
                total_poems=len(all_results),
                processed_poems=len(all_results),
                successful_poems=successful,
                failed_poems=failed
            ),
            'summary': {
                'total_files': total_files,
                'total_poems': len(all_results),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / max(len(all_results), 1)
            }
        }
    
    async def _add_intertextual_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add intertextual analysis to batch results"""
        self.logger.info("Adding intertextual analysis to results")

        try:
            for result in results.get('results', []):
                if not result.get('success', False):
                    continue

                poem_data = result.get('poem')
                analysis = result.get('analysis')

                if not poem_data or not analysis:
                    continue

                # Find intertextual references
                try:
                    references = self.intertextuality_analyzer.find_intertextual_references(
                        poem_data.content,
                        min_similarity=0.7
                    )

                    # Add to analysis result
                    if hasattr(analysis, 'content') and hasattr(analysis.content, 'intertextual_references'):
                        # Convert ClassicalReference objects to dicts for JSON serialization
                        ref_dicts = [
                            {
                                'source_author': ref.source_author,
                                'source_work': ref.source_work,
                                'source_text': ref.source_text,
                                'matched_text': ref.matched_text,
                                'similarity_score': ref.similarity_score,
                                'reference_type': ref.reference_type,
                                'confidence': ref.confidence,
                                'contextual_notes': ref.contextual_notes
                            }
                            for ref in references
                        ]
                        analysis.content.intertextual_references = ref_dicts

                        # Calculate intertextuality score
                        if references:
                            intertext_score = sum(r.confidence for r in references) / len(references)
                            # Store as additional metric
                            if not hasattr(analysis.content, 'classical_themes'):
                                analysis.content.classical_themes = {}
                            analysis.content.classical_themes['intertextuality_score'] = intertext_score

                        self.logger.debug(f"Found {len(references)} intertextual references for poem: {poem_data.title}")

                except Exception as e:
                    self.logger.warning(f"Error finding intertextual references for {poem_data.title}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error adding intertextual analysis: {e}")

        return results

    def save_batch_results(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        format_type: str = 'json'
    ) -> None:
        """Save batch results in specified format"""
        self.logger.info(f"Saving {len(results)} results to {output_path}")
        
        try:
            if format_type == 'json':
                self._save_as_json(results, output_path)
            elif format_type == 'excel':
                self._save_as_excel(results, output_path)
            elif format_type == 'csv':
                self._save_as_csv(results, output_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            self.logger.info(f"Results saved successfully to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def _save_as_json(self, results: List[Dict[str, Any]], output_path: Path):
        """Save results as JSON"""
        serialized_results = []
        
        for result in results:
            if result.get('success', False):
                serialized_results.append(self._serialize_result(result))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_results, f, indent=2, ensure_ascii=False, default=str)
    
    def _serialize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize analysis result for JSON output"""
        if not result.get('success', False):
            return {
                'success': False,
                'poem_id': result.get('poem_id'),
                'error': result.get('error', 'Unknown error')
            }
        
        analysis = result.get('analysis')
        poem = result.get('poem')
        
        return {
            'success': True,
            'poem_id': result.get('poem_id'),
            'poem_metadata': {
                'title': poem.title,
                'author': poem.author,
                'publication_year': poem.publication_year
            },
            'analysis_summary': {
                'meter': analysis.structural.aruz_analysis.identified_meter,
                'meter_confidence': analysis.structural.meter_confidence.value,
                'lines': analysis.structural.lines,
                'rhyme_pattern': analysis.structural.rhyme_pattern,
                'lexical_diversity': analysis.content.lexical_diversity,
                'quality_score': analysis.quality_metrics.get('quality_score', 0.0)
            },
            'processing_time': result.get('processing_time', 0)
        }
    
    def _save_as_excel(self, results: List[Dict[str, Any]], output_path: Path):
        """Save results as Excel using existing reporter"""
        excel_results = []
        
        for result in results:
            if result.get('success', False):
                excel_results.append({
                    'poem_id': result.get('poem_id'),
                    'title': result['poem'].title,
                    'content': result['poem'].content,
                    'analysis': result['analysis'],
                    'validation': result['analysis'].quality_metrics
                })
        
        if excel_results:
            reporter = ExcelReporter()
            reporter.create_comprehensive_report(excel_results, str(output_path))
    
    def _save_as_csv(self, results: List[Dict[str, Any]], output_path: Path):
        """Save results as CSV"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'poem_id', 'title', 'author', 'meter', 'confidence',
                'lines', 'rhyme_pattern', 'lexical_diversity', 'quality_score'
            ])
            
            # Write data
            for result in results:
                if result.get('success', False):
                    analysis = result['analysis']
                    poem = result['poem']
                    
                    writer.writerow([
                        result.get('poem_id'),
                        poem.title,
                        poem.author,
                        analysis.structural.aruz_analysis.identified_meter,
                        analysis.structural.meter_confidence.value,
                        analysis.structural.lines,
                        analysis.structural.rhyme_pattern,
                        analysis.content.lexical_diversity,
                        analysis.quality_metrics.get('quality_score', 0.0)
                    ])

# Compatibility exports
__all__ = [
    'ProductionAnalyzer', 'BatchConfig', 'BatchProgress', 'MemoryManager',
    'StreamingTextProcessor', 'EnhancedCorpusManager', 'BatchAnalysisEngine',
    'PoemData'  # Re-export from original
]
