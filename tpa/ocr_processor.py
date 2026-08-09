#!/usr/bin/env python3
"""
OCR Processor für gescannte tadschikische/persische PDFs
"""

import logging
from pathlib import Path
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

logger = logging.getLogger(__name__)


class TajikOCREngine:
    """OCR-Engine für tadschikische/persische Texte in gescannten PDFs"""

    def __init__(self, tessdata_path: Optional[str] = None, max_workers: int = 4):
        if not HAS_OCR:
            raise ImportError("pytesseract and PIL are required for OCR functionality")

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Tesseract-Sprachpakete für persische/tadschikische Sprachen
        self.languages = {
            'tajik_cyrillic': 'rus',  # Tadschikisch nutzt russisches Tesseract-Modell
            'persian': 'fas',          # Persisch (Arabische Schrift)
            'arabic': 'ara',           # Arabisch
            'russian': 'rus',          # Russisch (für Lehnwörter)
            'english': 'eng'           # Englisch (für gemischte Texte)
        }

        # Tesseract-Konfiguration
        self.tesseract_config = {
            'tajik_cyrillic': '-c preserve_interword_spaces=1 --psm 3',
            'persian': '-c preserve_interword_spaces=1 --psm 3',
            'arabic': '-c preserve_interword_spaces=1 --psm 3',
            'mixed': '-c preserve_interword_spaces=1 --psm 3'
        }

        if tessdata_path:
            pytesseract.pytesseract.tesseract_cmd = tessdata_path

    async def extract_text_from_scanned_pdf(self, pdf_path: Path) -> str:
        """Extrahiere Text aus gescanntem PDF mit OCR"""
        try:
            logger.info(f"Starting OCR processing for: {pdf_path.name}")

            # PDF in Bilder konvertieren
            images = await self._pdf_to_images(pdf_path)

            if not images:
                logger.warning(f"No images found in PDF: {pdf_path.name}")
                return ""

            # OCR auf allen Seiten parallel ausführen
            tasks = [self._ocr_image_async(image, page_num)
                    for page_num, image in enumerate(images)]

            page_texts = await asyncio.gather(*tasks)

            # Text bereinigen und zusammenfügen
            cleaned_text = self._clean_ocr_text("\n\n".join(page_texts))

            logger.info(f"OCR completed for {pdf_path.name}: {len(cleaned_text)} characters")
            return cleaned_text

        except Exception as e:
            logger.error(f"OCR processing failed for {pdf_path.name}: {e}")
            raise

    async def _pdf_to_images(self, pdf_path: Path) -> List:
        """Konvertiere PDF-Seiten zu PIL Images"""
        loop = asyncio.get_event_loop()

        def convert_pdf():
            # Versuche pdf2image für vollständige Konvertierung
            try:
                from pdf2image import convert_from_path
                logger.info(f"Converting PDF to images using pdf2image")
                return convert_from_path(str(pdf_path), dpi=300)
            except ImportError:
                logger.warning("pdf2image not installed - OCR requires pdf2image package")
                logger.warning("Install with: pip install pdf2image")
                logger.warning("Also requires poppler-utils on system")
                return []
            except Exception as e:
                logger.error(f"Failed to convert PDF to images: {e}")
                return []

        return await loop.run_in_executor(self.executor, convert_pdf)

    async def _ocr_image_async(self, image, page_num: int) -> str:
        """Führe OCR auf einem Bild asynchron aus"""
        loop = asyncio.get_event_loop()

        def perform_ocr():
            try:
                # Auto-detect language
                detected_lang = self._detect_language_from_image(image)

                # OCR mit spezifischer Konfiguration
                config = self.tesseract_config.get(detected_lang, '-c preserve_interword_spaces=1 --psm 3')

                # Bestimme Tesseract-Sprachcode
                lang_code = self._get_tesseract_lang_code(detected_lang)

                text = pytesseract.image_to_string(
                    image,
                    lang=lang_code,
                    config=config
                )

                # Persisch/Arabisch-Bidi-Probleme beheben
                if detected_lang in ['persian', 'arabic']:
                    text = self._fix_bidi_text(text)

                logger.debug(f"OCR page {page_num}: {len(text)} characters extracted")
                return text

            except Exception as e:
                logger.error(f"OCR failed for page {page_num}: {e}")
                return ""

        return await loop.run_in_executor(self.executor, perform_ocr)

    def _get_tesseract_lang_code(self, detected_lang: str) -> str:
        """Konvertiere erkannte Sprache zu Tesseract-Sprachcode"""
        lang_mapping = {
            'tajik_cyrillic': 'rus',  # Russisch für kyrillisch
            'persian': 'fas',
            'arabic': 'ara',
            'mixed': 'fas+rus+eng'
        }
        return lang_mapping.get(detected_lang, 'eng')

    def _detect_language_from_image(self, image) -> str:
        """Erkennen der Hauptsprache im Bild"""
        try:
            # Versuche zuerst mit schneller Spracherkennung
            # Für Performance: Nur obere 20% des Bildes analysieren
            width, height = image.size
            cropped = image.crop((0, 0, width, int(height * 0.2)))

            # Versuche zuerst persisch/arabisch
            text_sample = pytesseract.image_to_string(cropped, lang='fas', config='--psm 1')

            # Zähle arabische/persische Zeichen
            arabic_persian_chars = set('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيپچژگ')
            arabic_count = sum(1 for c in text_sample if c in arabic_persian_chars)

            if arabic_count > len(text_sample) * 0.2:  # Mindestens 20% arabische/persische Zeichen
                return 'persian'

            # Versuche tadschikisch (kyrillisch)
            text_sample = pytesseract.image_to_string(cropped, lang='rus', config='--psm 1')
            cyrillic_chars = set('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяӣӯҷқғҳ')
            cyrillic_count = sum(1 for c in text_sample if c in cyrillic_chars)

            if cyrillic_count > len(text_sample) * 0.2:
                return 'tajik_cyrillic'

            return 'mixed'

        except Exception as e:
            logger.warning(f"Language detection failed: {e}, using mixed mode")
            return 'mixed'

    def _fix_bidi_text(self, text: str) -> str:
        """Behebe Bidirektionalitätsprobleme in arabisch/persischen Texten"""
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display

            # Text für arabische/persische Schrift umformen
            reshaped_text = arabic_reshaper.reshape(text)
            fixed_text = get_display(reshaped_text)

            return fixed_text
        except ImportError:
            logger.debug("arabic_reshaper or python-bidi not installed, skipping bidi fix")
            return text
        except Exception as e:
            logger.warning(f"Bidi text fix failed: {e}")
            return text

    def _clean_ocr_text(self, text: str) -> str:
        """Bereinige OCR-Ergebnisse"""
        # Normalisiere Leerzeichen und Zeilenumbrüche
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:  # Leere Zeilen entfernen
                # Multiple Spaces zu einem Space
                line = ' '.join(line.split())
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def __del__(self):
        """Cleanup executor on deletion"""
        try:
            self.executor.shutdown(wait=False)
        except:
            pass
