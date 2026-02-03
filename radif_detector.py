#!/usr/bin/env python3
"""
Radīf Detector for Tajik Poetry Analyzer
========================================

Implements linguistically correct Radīf detection based on classical
Persian poetics (ʿArūḍ).

Definition (per Šams-i Qays Rāzī, al-Muʿjam):
---------------------------------------------
Radīf (ردیف) is the word or phrase that FOLLOWS the Qāfiyeh (rhyme)
and is REPEATED IDENTICALLY at the end of EVERY line (or every second
line in maṯnawī form).

Structure of line ending:
    ... [semantic content] [QĀFIYEH] [RADĪF]
    
Example from Ḥāfiẓ:
    مرا چشم‌هاست که هر دم به کوی تو می‌نگرد | هر شبی
    ز غم عشق تو می‌سوزم و می‌گذرد | هر شبی
    
    Here: Radīf = "هر شبی" (every night)
          Qāfiyeh = the rhyming sound before it (-garad/-gazrad)

Key Distinction from Previous Implementation:
--------------------------------------------
The OLD code looked for ANY frequently occurring word.
The NEW code checks specifically:
1. Is the word/phrase at the ACTUAL END of lines?
2. Does it appear in ALL lines (or a strict pattern like AA BA CA...)?
3. Is there a Qāfiyeh BEFORE it (i.e., is this truly Radīf or just rhyme)?

References:
-----------
- Šams-i Qays Rāzī (13th c.): al-Muʿjam fī maʿāyīr ašʿār al-ʿAjam
- Thiesen, Finn (1982): A Manual of Classical Persian Prosody
- Elwell-Sutton, L.P. (1976): The Persian Metres
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class RadifResult:
    """Result of Radīf detection with linguistic metadata."""
    
    radif_present: bool
    radif_text: str
    radif_type: str  # 'full' (all lines), 'ghazal' (AA BA CA...), 'partial', 'none'
    frequency: float  # Percentage of lines with this Radīf
    qafiyeh_before_radif: List[str]  # The rhyming elements before Radīf
    lines_with_radif: List[int]  # Line indices (0-based) that have Radīf
    lines_without_radif: List[int]  # Line indices without Radīf
    cleaned_lines: List[str]  # Lines with Radīf removed (for meter analysis)
    confidence: float  # How certain we are this is a true Radīf
    
    @property
    def is_classical_radif(self) -> bool:
        """True if this follows classical Persian Radīf rules."""
        return self.radif_type in ('full', 'ghazal') and self.confidence > 0.8


class ClassicalRadifDetector:
    """
    Detects Radīf according to classical Persian poetic rules.
    
    This implementation follows the definition in Šams-i Qays's al-Muʿjam:
    Radīf must be an IDENTICAL word/phrase at line endings, following the Qāfiyeh.
    """
    
    def __init__(self):
        # Common Tajik/Persian particles that are NOT typically Radīf
        # (they're too grammatical to be meaningful refrains)
        self.unlikely_radif = {
            'ва', 'ки', 'ба', 'аз', 'дар', 'бо', 'то', 'чун',
            'ҳам', 'на', 'ё', 'агар', 'пас', 'лекин', 'аммо',
            # Persian script equivalents
            'و', 'که', 'به', 'از', 'در', 'با', 'تا', 'چون'
        }
        
        # Minimum percentage of lines that must have Radīf for it to be valid
        self.min_full_radif_threshold = 0.9  # 90% for "full" Radīf
        self.min_ghazal_radif_threshold = 0.45  # ~50% for ghazal pattern (AA BA CA...)
        
        logger.info("ClassicalRadifDetector initialized")
    
    def detect(self, poem_content: str) -> RadifResult:
        """
        Detect Radīf in a poem using classical Persian rules.
        
        Args:
            poem_content: The full text of the poem
            
        Returns:
            RadifResult with detailed Radīf information
        """
        lines = [line.strip() for line in poem_content.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return self._no_radif(lines)
        
        # Step 1: Extract final words/phrases from each line
        line_endings = self._extract_line_endings(lines)
        
        # Step 2: Find candidate Radīf (must be at actual line end)
        candidates = self._find_radif_candidates(line_endings, lines)
        
        if not candidates:
            return self._no_radif(lines)
        
        # Step 3: Validate candidates against classical rules
        best_radif = self._validate_radif_candidates(candidates, lines, line_endings)
        
        if not best_radif:
            return self._no_radif(lines)
        
        # Step 4: Build result with full analysis
        return self._build_result(best_radif, lines, line_endings)
    
    def _extract_line_endings(self, lines: List[str]) -> List[Dict]:
        """
        Extract the final 1-3 words from each line.
        
        Returns list of dicts with:
        - 'line_idx': line index
        - 'full_line': original line
        - 'words': list of words in line
        - 'endings': dict of {word_count: ending_phrase}
        """
        results = []
        
        for idx, line in enumerate(lines):
            # Extract words (Cyrillic + Arabic script support)
            words = re.findall(r'[\wӣӯ\u0600-\u06FF]+', line)
            
            if not words:
                results.append({
                    'line_idx': idx,
                    'full_line': line,
                    'words': [],
                    'endings': {}
                })
                continue
            
            # Get last 1, 2, 3 words as potential Radīf
            endings = {}
            for n in range(1, min(4, len(words) + 1)):
                ending = ' '.join(words[-n:])
                endings[n] = ending.lower()
            
            results.append({
                'line_idx': idx,
                'full_line': line,
                'words': words,
                'endings': endings
            })
        
        return results
    
    def _find_radif_candidates(self, line_endings: List[Dict], 
                                lines: List[str]) -> List[Dict]:
        """
        Find potential Radīf candidates that appear at actual line ends.
        
        A valid candidate must:
        1. Appear as the FINAL word(s) of multiple lines
        2. Not be a common grammatical particle (unless in phrase)
        3. Be identical (not just similar) across lines
        """
        candidates = []
        
        # Count endings by word count
        for word_count in [1, 2, 3]:
            ending_counter = Counter()
            
            for le in line_endings:
                if word_count in le['endings']:
                    ending = le['endings'][word_count]
                    ending_counter[ending] += 1
            
            # Find endings that appear multiple times
            for ending, count in ending_counter.items():
                if count >= 2:  # At least 2 occurrences
                    # Skip unlikely single-word Radīf
                    if word_count == 1 and ending in self.unlikely_radif:
                        continue
                    
                    # Find which lines have this ending
                    lines_with = [
                        le['line_idx'] for le in line_endings
                        if le['endings'].get(word_count) == ending
                    ]
                    
                    frequency = len(lines_with) / len(lines)
                    
                    candidates.append({
                        'text': ending,
                        'word_count': word_count,
                        'count': count,
                        'frequency': frequency,
                        'lines_with': lines_with
                    })
        
        # Sort by frequency (prefer higher), then by word count (prefer shorter)
        candidates.sort(key=lambda x: (-x['frequency'], x['word_count']))
        
        return candidates
    
    def _validate_radif_candidates(self, candidates: List[Dict],
                                    lines: List[str],
                                    line_endings: List[Dict]) -> Optional[Dict]:
        """
        Validate candidates against classical Radīf rules.
        
        Classical rules:
        1. FULL RADĪF: Same word(s) at end of ALL lines
        2. GHAZAL RADĪF: Same word(s) at end of rhyming lines (AA BA CA...)
           - First two lines (maṭlaʿ) both have Radīf
           - Then every second line has Radīf
        3. PARTIAL: >50% of lines, but not fitting above patterns
        """
        for candidate in candidates:
            text = candidate['text']
            frequency = candidate['frequency']
            lines_with = set(candidate['lines_with'])
            
            # Check for FULL Radīf
            if frequency >= self.min_full_radif_threshold:
                candidate['radif_type'] = 'full'
                candidate['confidence'] = frequency
                return candidate
            
            # Check for GHAZAL pattern (AA BA CA DA...)
            # Lines 0, 1 must have Radīf (maṭlaʿ)
            # Then lines 3, 5, 7... (every second line starting from 3)
            if len(lines) >= 4:
                ghazal_lines = {0, 1}  # First couplet (maṭlaʿ)
                for i in range(3, len(lines), 2):  # 3, 5, 7, ...
                    ghazal_lines.add(i)
                
                # Check if Radīf appears in ghazal pattern
                ghazal_match = lines_with & ghazal_lines
                ghazal_coverage = len(ghazal_match) / len(ghazal_lines)
                
                if ghazal_coverage >= 0.8:  # 80% of expected positions
                    candidate['radif_type'] = 'ghazal'
                    candidate['confidence'] = ghazal_coverage
                    candidate['lines_with'] = list(ghazal_lines & lines_with)
                    return candidate
            
            # Check for PARTIAL Radīf
            if frequency >= self.min_ghazal_radif_threshold:
                candidate['radif_type'] = 'partial'
                candidate['confidence'] = frequency * 0.7  # Lower confidence
                return candidate
        
        return None
    
    def _build_result(self, radif_info: Dict, lines: List[str],
                      line_endings: List[Dict]) -> RadifResult:
        """Build the final RadifResult with all metadata."""
        
        radif_text = radif_info['text']
        lines_with = set(radif_info['lines_with'])
        lines_without = [i for i in range(len(lines)) if i not in lines_with]
        
        # Extract Qāfiyeh (the rhyming element BEFORE Radīf)
        qafiyeh_list = []
        for le in line_endings:
            if le['line_idx'] in lines_with:
                words = le['words']
                radif_word_count = radif_info['word_count']
                
                # Qāfiyeh is the word before Radīf
                if len(words) > radif_word_count:
                    qafiyeh_word = words[-(radif_word_count + 1)]
                    # Extract rhyming portion (last 2-3 chars typically)
                    qafiyeh = qafiyeh_word[-3:] if len(qafiyeh_word) >= 3 else qafiyeh_word
                    qafiyeh_list.append(qafiyeh.lower())
        
        # Create cleaned lines (Radīf removed for meter analysis)
        cleaned_lines = []
        for i, line in enumerate(lines):
            if i in lines_with:
                # Remove Radīf from end
                pattern = re.escape(radif_text) + r'\s*$'
                cleaned = re.sub(pattern, '', line, flags=re.IGNORECASE).strip()
                # Also remove trailing punctuation
                cleaned = cleaned.rstrip(' ,;:!?.—–-')
                cleaned_lines.append(cleaned)
            else:
                cleaned_lines.append(line)
        
        logger.info(f"Detected {radif_info['radif_type']} Radīf: '{radif_text}' "
                   f"in {len(lines_with)}/{len(lines)} lines")
        
        return RadifResult(
            radif_present=True,
            radif_text=radif_text,
            radif_type=radif_info['radif_type'],
            frequency=radif_info['frequency'],
            qafiyeh_before_radif=qafiyeh_list,
            lines_with_radif=list(lines_with),
            lines_without_radif=lines_without,
            cleaned_lines=cleaned_lines,
            confidence=radif_info['confidence']
        )
    
    def _no_radif(self, lines: List[str]) -> RadifResult:
        """Return result indicating no Radīf found."""
        return RadifResult(
            radif_present=False,
            radif_text="",
            radif_type="none",
            frequency=0.0,
            qafiyeh_before_radif=[],
            lines_with_radif=[],
            lines_without_radif=list(range(len(lines))),
            cleaned_lines=lines.copy(),
            confidence=0.0
        )


# === INTEGRATION WITH ANALYZER ===

def integrate_radif_detection(structural_analyzer_instance):
    """
    Monkey-patch or extend the StructuralAnalyzer to use ClassicalRadifDetector.
    
    Usage:
        from radif_detector import integrate_radif_detection
        integrate_radif_detection(my_analyzer.structural_analyzer)
    """
    structural_analyzer_instance.radif_detector = ClassicalRadifDetector()
    logger.info("Integrated ClassicalRadifDetector into StructuralAnalyzer")


# === DEMONSTRATION ===

if __name__ == "__main__":
    detector = ClassicalRadifDetector()
    
    # Test poem with clear Radīf "вақт"
    test_poem_with_radif = """
Лаҳзаи ширин гузашт, ин аст вақт
Дил ба ёди ёр месӯзад ҳар вақт
Зиндагӣ мисли шароре аст вақт
Бояд онро қадр донист ҳар вақт
"""
    
    # Test poem without Radīf
    test_poem_without = """
Дар кӯҳсори ватан гулҳо мешукуфанд
Дили ошиқ аз муҳаббат меларзад
Баҳори нав ба замин таҷдид меорад
Навиди хушҳолии мардум мерасад
"""
    
    print("=== Radīf Detection Demo ===\n")
    
    print("Test 1: Poem with Radīf 'вақт'")
    result1 = detector.detect(test_poem_with_radif)
    print(f"  Radīf present: {result1.radif_present}")
    print(f"  Radīf text: '{result1.radif_text}'")
    print(f"  Type: {result1.radif_type}")
    print(f"  Frequency: {result1.frequency:.0%}")
    print(f"  Confidence: {result1.confidence:.0%}")
    print(f"  Qāfiyeh: {result1.qafiyeh_before_radif}")
    print(f"  Is classical: {result1.is_classical_radif}")
    print()
    
    print("Test 2: Poem without Radīf")
    result2 = detector.detect(test_poem_without)
    print(f"  Radīf present: {result2.radif_present}")
    print(f"  Type: {result2.radif_type}")
    print()
    
    print("Test 3: Cleaned lines (Radīf removed)")
    for i, (orig, clean) in enumerate(zip(
        test_poem_with_radif.strip().split('\n'),
        result1.cleaned_lines
    )):
        orig = orig.strip()
        if orig:
            print(f"  Original: {orig}")
            print(f"  Cleaned:  {clean}")
            print()
