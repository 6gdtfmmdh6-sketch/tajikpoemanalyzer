#!/usr/bin/env python3
"""
Poem Splitter and Title Extractor for Tajik Poetry Analyzer
===========================================================

This module handles the separation of poems from text files and
proper extraction of titles, ensuring that title metadata is
correctly communicated to the analysis pipeline.

The Problem This Solves:
------------------------
In the old implementation, `_split_poems()` would extract titles but
then pass the content to `analyze()` without communicating whether
the title was already removed. This caused titles to be counted
in syllable statistics.

Solution:
---------
1. PoemData dataclass now includes `title_extracted: bool` flag
2. StructuralAnalyzer.analyze() checks this flag before processing
3. Clear separation between "raw content" and "analyzed content"

Design Principles:
------------------
- Explicit metadata: Never assume, always communicate state
- Defensive parsing: Handle edge cases gracefully
- Linguistic awareness: Use Tajik/Persian title conventions
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoemMetadata:
    """
    Metadata about a poem's structure and provenance.
    
    This is the key innovation: we explicitly track what processing
    has been done so downstream analyzers don't double-process.
    """
    title_extracted: bool = False      # True if title was removed from content
    title_line_number: int = -1        # Original line number of title (-1 if no title)
    source_file: Optional[str] = None  # Source file path
    source_line_start: int = 0         # Starting line in source file
    source_line_end: int = 0           # Ending line in source file
    encoding: str = 'utf-8'            # Source encoding
    separator_type: Optional[str] = None  # What separated this poem from others
    raw_content: str = ""              # Original unprocessed content


@dataclass
class PoemData:
    """
    Data structure for a single poem with full metadata.
    
    Key fields:
    - title: The extracted title (may be auto-generated if not found)
    - content: The poem TEXT (title already removed if title_extracted=True)
    - metadata: Processing metadata including title_extracted flag
    """
    title: str
    content: str  # This is the BODY only (title removed if metadata.title_extracted)
    poem_id: Optional[str] = None
    metadata: PoemMetadata = field(default_factory=PoemMetadata)
    
    def get_full_text(self) -> str:
        """Get the complete poem including title (for display)."""
        if self.metadata.title_extracted and self.title:
            return f"{self.title}\n\n{self.content}"
        return self.content
    
    def get_analysis_text(self) -> str:
        """
        Get the text to analyze (title EXCLUDED).
        
        This is what should be passed to StructuralAnalyzer.
        """
        return self.content  # Title is already excluded


@dataclass
class SplitterConfig:
    """Configuration for poem splitting and title extraction."""
    
    # Title detection
    min_title_length: int = 2
    max_title_length: int = 80
    title_must_be_uppercase: bool = False  # Tajik titles often ALL CAPS
    title_forbidden_endings: Tuple[str, ...] = ('.', '!', '?', ':', ',', ';')
    
    # Poem separation
    min_poem_length: int = 10  # Minimum characters
    min_poem_lines: int = 2    # Minimum lines
    
    # Separator patterns (regex)
    separator_patterns: Tuple[str, ...] = (
        r'\*{5,}',           # *****
        r'-{5,}',            # -----
        r'={5,}',            # =====
        r'_{5,}',            # _____
        r'#{2,}\s*\d+\s*#{2,}',  # ## 1 ##
        r'\n\s*\n\s*\n+',    # Multiple blank lines
    )


class TajikTitleExtractor:
    """
    Extracts titles from Tajik poems using linguistic heuristics.
    
    Tajik poetry title conventions:
    - Often in ALL CAPS (ВАҚТ, МОДАР, ВАТАН)
    - Usually short (1-5 words)
    - No ending punctuation
    - May be numbered (1. ВАҚТ)
    - May have dedication (Ба модарам)
    """
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        self.config = config or SplitterConfig()
        
        # Patterns that indicate a title line
        self.title_patterns = [
            # Numbered title: "1. ВАҚТ" or "I. МОДАР"
            re.compile(r'^\s*[\dIVXLCDM]+[\.\)]\s*(.+)$'),
            # ALL CAPS title (common in Tajik)
            re.compile(r'^[А-ЯҒӢҚӮҲҶЁA-Z\s]+$'),
            # Short line without punctuation
            re.compile(r'^[А-Яа-яҒғӢӣҚқӮӯҲҳҶҷЁёA-Za-z\s]{2,50}$'),
        ]
        
        # Patterns that indicate NOT a title
        self.non_title_patterns = [
            # Lines with verse-like features
            re.compile(r'[,;:—–-]\s*$'),  # Ends with punctuation typical in verse
            re.compile(r'^[а-яғӣқӯҳҷё]'),  # Starts with lowercase
        ]
    
    def extract_title(self, lines: List[str]) -> Tuple[Optional[str], List[str], int]:
        """
        Extract title from poem lines.
        
        Args:
            lines: List of poem lines
            
        Returns:
            Tuple of (title, remaining_lines, title_line_index)
            If no title found: (None, original_lines, -1)
        """
        if not lines:
            return None, lines, -1
        
        # Check first few lines for title
        for i, line in enumerate(lines[:3]):  # Check first 3 lines max
            line = line.strip()
            
            if not line:
                continue
            
            # Check length constraints
            if not (self.config.min_title_length <= len(line) <= self.config.max_title_length):
                continue
            
            # Check forbidden endings
            if any(line.endswith(end) for end in self.config.title_forbidden_endings):
                continue
            
            # Check for title patterns
            is_title = False
            
            # ALL CAPS is strong indicator in Tajik
            if line.isupper() or self._is_mixed_caps_title(line):
                is_title = True
            
            # Numbered title
            for pattern in self.title_patterns[:1]:  # Check numbered pattern
                if pattern.match(line):
                    # Extract title without number
                    match = pattern.match(line)
                    if match:
                        line = match.group(1).strip()
                    is_title = True
                    break
            
            # Short line without verse features
            if not is_title and len(line) < 40:
                # Check it's not verse
                has_verse_features = any(p.search(line) for p in self.non_title_patterns)
                if not has_verse_features and line[0].isupper():
                    is_title = True
            
            if is_title:
                remaining = lines[i+1:]
                # Skip any blank lines after title
                while remaining and not remaining[0].strip():
                    remaining = remaining[1:]
                
                logger.debug(f"Extracted title: '{line}' from line {i}")
                return line, remaining, i
        
        return None, lines, -1
    
    def _is_mixed_caps_title(self, line: str) -> bool:
        """Check if line is a title with mixed caps (like 'ВАҚТ' or 'Модар')."""
        words = line.split()
        if not words:
            return False
        
        # First word capitalized, rest can vary
        first_word = words[0]
        return first_word[0].isupper() and len(line) < 50


class EnhancedPoemSplitter:
    """
    Splits a text file into individual poems with proper metadata.
    
    Key improvement: Explicitly tracks whether title was extracted
    and communicates this to downstream analyzers.
    """
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        self.config = config or SplitterConfig()
        self.title_extractor = TajikTitleExtractor(self.config)
        
        # Compile separator pattern
        self.separator_pattern = re.compile(
            '|'.join(self.config.separator_patterns)
        )
        
        logger.info("EnhancedPoemSplitter initialized")
    
    def split_text(self, text: str, source_file: Optional[str] = None) -> List[PoemData]:
        """
        Split text into individual poems.
        
        Args:
            text: Full text containing one or more poems
            source_file: Optional source file path for metadata
            
        Returns:
            List of PoemData with proper metadata
        """
        # Normalize text
        text = unicodedata.normalize('NFC', text)
        
        # Split by separators
        blocks = self.separator_pattern.split(text)
        
        poems = []
        line_offset = 0
        
        for i, block in enumerate(blocks, 1):
            block = block.strip()
            
            # Skip too-short blocks
            if len(block) < self.config.min_poem_length:
                line_offset += block.count('\n') + 1
                continue
            
            lines = block.split('\n')
            non_empty_lines = [l for l in lines if l.strip()]
            
            # Skip blocks with too few lines
            if len(non_empty_lines) < self.config.min_poem_lines:
                line_offset += len(lines)
                continue
            
            # Extract title
            title, content_lines, title_line_idx = self.title_extractor.extract_title(lines)
            
            # Build metadata
            metadata = PoemMetadata(
                title_extracted=(title is not None),
                title_line_number=title_line_idx,
                source_file=source_file,
                source_line_start=line_offset,
                source_line_end=line_offset + len(lines),
                raw_content=block
            )
            
            # Create poem data
            poem = PoemData(
                title=title if title else f"Poem {i}",
                content='\n'.join(content_lines) if title else block,
                poem_id=f"poem_{i:03d}",
                metadata=metadata
            )
            
            poems.append(poem)
            line_offset += len(lines)
        
        logger.info(f"Split text into {len(poems)} poems")
        return poems
    
    def split_file(self, filepath: str) -> List[PoemData]:
        """
        Split a file into individual poems.
        
        Args:
            filepath: Path to text file
            
        Returns:
            List of PoemData
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Try different encodings
        for encoding in ['utf-8', 'utf-8-sig', 'cp1251', 'iso-8859-1']:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode file: {filepath}")
        
        return self.split_text(text, source_file=str(path))


# === INTEGRATION HELPER ===

def analyze_poem_safely(analyzer, poem: PoemData) -> Any:
    """
    Safely analyze a poem, respecting metadata flags.
    
    This is the integration point that ensures titles aren't
    double-counted in syllable statistics.
    
    Args:
        analyzer: A TajikPoemAnalyzer or EnhancedTajikPoemAnalyzer instance
        poem: PoemData with metadata
        
    Returns:
        Analysis result
    """
    # Use the analysis text (title excluded)
    text_to_analyze = poem.get_analysis_text()
    
    # Log what we're doing
    if poem.metadata.title_extracted:
        logger.debug(f"Analyzing '{poem.title}' (title excluded from content)")
    else:
        logger.debug(f"Analyzing '{poem.title}' (no title extracted)")
    
    # Call the analyzer
    return analyzer.analyze_poem(text_to_analyze)


# === DEMONSTRATION ===

if __name__ == "__main__":
    # Test text with titles
    test_text = """
ВАҚТ

Лаҳзаи ширин гузашт, ин аст вақт
Дил ба ёди ёр месӯзад ҳар вақт
Зиндагӣ мисли шароре аст вақт
Бояд онро қадр донист ҳар вақт

*****

МОДАР

Эй модар, эй меҳрубон,
Дар оғӯши ту ёфтам ҷон.
Кӯҳҳои ту сари фалак расида,
Дарёҳои ту ҷовидон.

*****

Дар кӯҳсори ватан гулҳо мешукуфанд,
Дили ошиқ аз муҳаббат меларзад.
Баҳори нав ба замин таҷдид меорад,
Навиди хушҳолии мардум мерасад.
"""
    
    print("=== Poem Splitter Demo ===\n")
    
    splitter = EnhancedPoemSplitter()
    poems = splitter.split_text(test_text)
    
    for poem in poems:
        print(f"Poem ID: {poem.poem_id}")
        print(f"Title: '{poem.title}'")
        print(f"Title extracted: {poem.metadata.title_extracted}")
        print(f"Title line number: {poem.metadata.title_line_number}")
        print(f"Content lines: {len(poem.content.split(chr(10)))}")
        print(f"Content preview: {poem.content[:50]}...")
        print()
        
        # Show the difference
        print("Full text (for display):")
        print(poem.get_full_text()[:100] + "...")
        print()
        
        print("Analysis text (title excluded):")
        print(poem.get_analysis_text()[:100] + "...")
        print()
        print("-" * 50)
        print()
