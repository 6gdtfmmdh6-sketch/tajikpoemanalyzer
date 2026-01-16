#!/usr/bin/env python3
"""
PDF Handler for Tajik Poetry Analyzer
Integrates PDF and OCR functionality into the original analyzer
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Handles PDF text extraction with OCR fallback"""

    def __init__(self):
        self.ocr_available = False
        try:
            from ocr_processor import TajikOCREngine
            self.ocr_engine = TajikOCREngine(max_workers=2)
            self.ocr_available = True
            logger.info("OCR engine initialized successfully")
        except ImportError as e:
            logger.warning(f"OCR not available: {e}")
            logger.info("Install OCR dependencies: pip install pytesseract pdf2image")

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content

        Raises:
            ImportError: If PyPDF2 is not installed
            Exception: If extraction fails
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 required for PDF support. Install: pip install PyPDF2")

        try:
            text_content = []

            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                logger.info(f"Reading PDF with {len(pdf_reader.pages)} pages")

                # Try extracting text from first page to check if it's scanned
                first_page_text = ""
                if len(pdf_reader.pages) > 0:
                    first_page_text = pdf_reader.pages[0].extract_text()

                # If very little text, likely scanned - try OCR
                if len(first_page_text.strip()) < 50 and self.ocr_available:
                    logger.info("PDF appears to be scanned, attempting OCR extraction")
                    return self._extract_with_ocr(pdf_path)

                # Normal text extraction
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()

                    # Clean PDF text
                    text = self._clean_pdf_text(text)

                    if text.strip():
                        text_content.append(text)

            result = '\n\n'.join(text_content)
            logger.info(f"Extracted {len(result)} characters from PDF")
            return result

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

    def _extract_with_ocr(self, pdf_path: Path) -> str:
        """Extract text using OCR (synchronous wrapper)"""
        if not self.ocr_available:
            logger.warning("OCR not available, returning empty text")
            return ""

        try:
            import asyncio
            # Run OCR in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            text = loop.run_until_complete(
                self.ocr_engine.extract_text_from_scanned_pdf(pdf_path)
            )
            return text
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def _clean_pdf_text(self, text: str) -> str:
        """Clean PDF-extracted text"""
        # Remove ligatures
        replacements = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
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


def read_file_with_pdf_support(file_path: Path) -> str:
    """
    Read text file or PDF

    Args:
        file_path: Path to input file

    Returns:
        File content as string
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Handle PDF
    if file_path.suffix.lower() == '.pdf':
        extractor = PDFTextExtractor()
        return extractor.extract_text_from_pdf(file_path)

    # Handle text files
    else:
        # Try different encodings
        encodings = ['utf-8', 'windows-1251', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Fallback: read with errors='replace'
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
