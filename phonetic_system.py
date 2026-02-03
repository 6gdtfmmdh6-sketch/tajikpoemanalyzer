#!/usr/bin/env python3
"""
Phonetic System for Tajik Poetry Analyzer
=========================================

This module provides a simplified phonological abstraction for ʿArūḍ analysis,
plus optional scientific transliteration (IJMES/DMG) for academic outputs.

Design Philosophy:
------------------
For prosodic analysis, we don't need full IPA detail. We need:
1. Vowel length distinction (short vs. long)
2. Syllable structure (open CV vs. closed CVC)
3. Diphthong recognition

For scientific citation in a Masterarbeit, we need:
- IJMES (International Journal of Middle East Studies) transliteration
- DMG (Deutsche Morgenländische Gesellschaft) for German academic contexts

References:
-----------
- Šams-i Qays Rāzī, al-Muʿjam fī maʿāyīr ašʿār al-ʿAjam (13th c.)
- Thiesen, Finn (1982): A Manual of Classical Persian Prosody
- Perry, John R. (2005): A Tajik Persian Reference Grammar
"""

import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict
import re


class VowelLength(Enum):
    """Phonological vowel length for prosody"""
    SHORT = "short"  # a, e, i, o, u
    LONG = "long"    # ā, ī, ū (Tajik: ӣ, ӯ, historically also о)


class PhonemeClass(Enum):
    """Abstract phoneme classes for prosodic analysis"""
    VOWEL_SHORT = "V"
    VOWEL_LONG = "V̄"
    CONSONANT = "C"
    SEMIVOWEL = "J"  # й, в when glide


@dataclass
class ProsodicPhoneme:
    """A phoneme with prosodic weight information"""
    grapheme: str           # Original Cyrillic/Arabic character
    phoneme_class: PhonemeClass
    ijmes: str              # IJMES transliteration
    dmg: str                # DMG transliteration
    
    def is_vowel(self) -> bool:
        return self.phoneme_class in (PhonemeClass.VOWEL_SHORT, PhonemeClass.VOWEL_LONG)
    
    def is_long(self) -> bool:
        return self.phoneme_class == PhonemeClass.VOWEL_LONG


class TajikPhonology:
    """
    Tajik phonological system for prosodic analysis.
    
    This replaces the IPA-based system with a simpler abstraction
    that captures exactly what's needed for ʿArūḍ meter analysis.
    """
    
    def __init__(self):
        # === CYRILLIC MAPPINGS ===
        
        # Short vowels (Tajik Cyrillic)
        self.short_vowels_cyr = {
            'а': ('a', 'a'),      # IJMES, DMG
            'е': ('e', 'e'),
            'и': ('i', 'i'),
            'о': ('o', 'o'),
            'у': ('u', 'u'),
            'э': ('e', 'e'),
        }
        
        # Long vowels (Tajik Cyrillic) - marked with macron in IJMES/DMG
        self.long_vowels_cyr = {
            'ӣ': ('ī', 'ī'),      # Long i
            'ӯ': ('ū', 'ū'),      # Long u
            # Note: Historical ā is written as 'о' in modern Tajik
            # but synchronically 'о' is short in most contexts
        }
        
        # Consonants (Tajik Cyrillic → IJMES, DMG)
        self.consonants_cyr = {
            'б': ('b', 'b'),
            'в': ('v', 'w'),      # Can be semivowel
            'г': ('g', 'g'),
            'ғ': ('gh', 'ġ'),     # Voiced uvular fricative
            'д': ('d', 'd'),
            'ж': ('zh', 'ž'),
            'з': ('z', 'z'),
            'й': ('y', 'y'),      # Semivowel
            'к': ('k', 'k'),
            'қ': ('q', 'q'),      # Voiceless uvular stop
            'л': ('l', 'l'),
            'м': ('m', 'm'),
            'н': ('n', 'n'),
            'п': ('p', 'p'),
            'р': ('r', 'r'),
            'с': ('s', 's'),
            'т': ('t', 't'),
            'ф': ('f', 'f'),
            'х': ('kh', 'ḫ'),     # Voiceless uvular fricative
            'ҳ': ('h', 'h'),      # Voiceless glottal fricative
            'ч': ('ch', 'č'),
            'ҷ': ('j', 'ǧ'),      # Voiced palato-alveolar affricate
            'ш': ('sh', 'š'),
            'ъ': ('ʿ', 'ʿ'),      # ʿAyn (usually silent in Tajik)
        }
        
        # Diphthongs (counted as long for prosody)
        self.diphthongs_cyr = {
            'ай': ('ay', 'ay'),
            'ой': ('oy', 'oy'),
            'уй': ('uy', 'uy'),
            'ей': ('ey', 'ey'),
            'ав': ('aw', 'aw'),
            'ов': ('ow', 'ow'),
        }
        
        # Compound letters (Cyrillic)
        self.compounds_cyr = {
            'я': ('ya', 'ya'),    # й + а
            'ю': ('yu', 'yu'),    # й + у
            'ё': ('yo', 'yo'),    # й + о
        }
        
        # === ARABIC-PERSIAN SCRIPT MAPPINGS ===
        # For historical texts or Persian-script Tajik
        
        self.arabic_consonants = {
            'ب': ('b', 'b'),
            'پ': ('p', 'p'),
            'ت': ('t', 't'),
            'ث': ('s̱', 'ṯ'),      # Historically th, now s
            'ج': ('j', 'ǧ'),
            'چ': ('ch', 'č'),
            'ح': ('ḥ', 'ḥ'),      # Voiceless pharyngeal
            'خ': ('kh', 'ḫ'),
            'د': ('d', 'd'),
            'ذ': ('ẕ', 'ḏ'),      # Historically dh, now z
            'ر': ('r', 'r'),
            'ز': ('z', 'z'),
            'ژ': ('zh', 'ž'),
            'س': ('s', 's'),
            'ش': ('sh', 'š'),
            'ص': ('ṣ', 'ṣ'),      # Emphatic s
            'ض': ('ż', 'ḍ'),      # Emphatic d/z
            'ط': ('ṭ', 'ṭ'),      # Emphatic t
            'ظ': ('ẓ', 'ẓ'),      # Emphatic z
            'ع': ('ʿ', 'ʿ'),      # ʿAyn
            'غ': ('gh', 'ġ'),
            'ف': ('f', 'f'),
            'ق': ('q', 'q'),
            'ک': ('k', 'k'),
            'گ': ('g', 'g'),
            'ل': ('l', 'l'),
            'م': ('m', 'm'),
            'ن': ('n', 'n'),
            'و': ('v/w/ū', 'w/ū'),  # Context-dependent
            'ه': ('h', 'h'),
            'ی': ('y/ī', 'y/ī'),    # Context-dependent
        }
        
        # Build lookup sets
        self._vowels = set(self.short_vowels_cyr.keys()) | set(self.long_vowels_cyr.keys())
        self._consonants = set(self.consonants_cyr.keys())
        self._semivowels = {'й', 'в', 'و', 'ی'}
        
    def analyze_prosodic(self, text: str) -> List[ProsodicPhoneme]:
        """
        Analyze text for prosodic phonemes.
        
        Returns a list of ProsodicPhoneme objects with class information
        needed for ʿArūḍ analysis.
        """
        text = unicodedata.normalize('NFC', text.lower())
        phonemes = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Skip whitespace and punctuation
            if char.isspace() or not char.isalnum():
                i += 1
                continue
            
            # Check for diphthongs (2-char sequences)
            if i + 1 < len(text):
                digraph = text[i:i+2]
                if digraph in self.diphthongs_cyr:
                    ijmes, dmg = self.diphthongs_cyr[digraph]
                    phonemes.append(ProsodicPhoneme(
                        grapheme=digraph,
                        phoneme_class=PhonemeClass.VOWEL_LONG,  # Diphthongs = long
                        ijmes=ijmes,
                        dmg=dmg
                    ))
                    i += 2
                    continue
            
            # Check for compound letters
            if char in self.compounds_cyr:
                ijmes, dmg = self.compounds_cyr[char]
                # я, ю, ё = semivowel + short vowel
                phonemes.append(ProsodicPhoneme(
                    grapheme=char,
                    phoneme_class=PhonemeClass.VOWEL_SHORT,
                    ijmes=ijmes,
                    dmg=dmg
                ))
                i += 1
                continue
            
            # Long vowels
            if char in self.long_vowels_cyr:
                ijmes, dmg = self.long_vowels_cyr[char]
                phonemes.append(ProsodicPhoneme(
                    grapheme=char,
                    phoneme_class=PhonemeClass.VOWEL_LONG,
                    ijmes=ijmes,
                    dmg=dmg
                ))
                i += 1
                continue
            
            # Short vowels
            if char in self.short_vowels_cyr:
                ijmes, dmg = self.short_vowels_cyr[char]
                phonemes.append(ProsodicPhoneme(
                    grapheme=char,
                    phoneme_class=PhonemeClass.VOWEL_SHORT,
                    ijmes=ijmes,
                    dmg=dmg
                ))
                i += 1
                continue
            
            # Consonants
            if char in self.consonants_cyr:
                ijmes, dmg = self.consonants_cyr[char]
                # Check if semivowel in glide position
                if char in self._semivowels:
                    # After vowel = part of diphthong (handled above)
                    # Before vowel = semivowel/glide
                    phoneme_class = PhonemeClass.SEMIVOWEL
                else:
                    phoneme_class = PhonemeClass.CONSONANT
                phonemes.append(ProsodicPhoneme(
                    grapheme=char,
                    phoneme_class=phoneme_class,
                    ijmes=ijmes,
                    dmg=dmg
                ))
                i += 1
                continue
            
            # Arabic script consonants
            if char in self.arabic_consonants:
                ijmes, dmg = self.arabic_consonants[char]
                phonemes.append(ProsodicPhoneme(
                    grapheme=char,
                    phoneme_class=PhonemeClass.CONSONANT,
                    ijmes=ijmes,
                    dmg=dmg
                ))
                i += 1
                continue
            
            # Unknown character - skip
            i += 1
        
        return phonemes
    
    def to_ijmes(self, text: str) -> str:
        """
        Convert text to IJMES transliteration.
        
        Use this for scientific citations in English-language publications.
        """
        phonemes = self.analyze_prosodic(text)
        return ''.join(p.ijmes for p in phonemes)
    
    def to_dmg(self, text: str) -> str:
        """
        Convert text to DMG transliteration.
        
        Use this for scientific citations in German-language publications
        (standard for Deutsche Morgenländische Gesellschaft).
        """
        phonemes = self.analyze_prosodic(text)
        return ''.join(p.dmg for p in phonemes)
    
    def syllabify_prosodic(self, text: str) -> List[Tuple[str, bool]]:
        """
        Syllabify text and return (syllable_text, is_heavy) tuples.
        
        A syllable is HEAVY (—) if:
        1. It contains a long vowel (ӣ, ӯ)
        2. It contains a diphthong (ай, ой, etc.)
        3. It is closed (CVC) - ends with consonant
        
        A syllable is LIGHT (U) if:
        1. It is open (CV) with short vowel
        
        This is the core logic needed for ʿArūḍ meter analysis.
        """
        text = unicodedata.normalize('NFC', text.lower())
        syllables = []
        
        # Remove punctuation but keep spaces
        clean_text = re.sub(r'[^\w\s]', '', text)
        words = clean_text.split()
        
        for word in words:
            word_syllables = self._syllabify_word(word)
            syllables.extend(word_syllables)
        
        return syllables
    
    def _syllabify_word(self, word: str) -> List[Tuple[str, bool]]:
        """Syllabify a single word."""
        syllables = []
        i = 0
        
        while i < len(word):
            syl_start = i
            syl_has_long_vowel = False
            syl_has_coda = False
            
            # 1. Onset: collect initial consonants
            while i < len(word) and word[i] in self._consonants:
                i += 1
            
            # 2. Nucleus: must have vowel
            if i < len(word) and word[i] in self._vowels:
                # Check for diphthong
                if i + 1 < len(word) and word[i:i+2] in self.diphthongs_cyr:
                    syl_has_long_vowel = True
                    i += 2
                elif word[i] in self.long_vowels_cyr:
                    syl_has_long_vowel = True
                    i += 1
                else:
                    i += 1
                
                # 3. Coda: check for closing consonants
                # In Tajik, a consonant belongs to the coda if:
                # - It's word-final, OR
                # - It's followed by another consonant
                while i < len(word) and word[i] in self._consonants:
                    # Look ahead: if next char is vowel, this C starts next syllable
                    if i + 1 < len(word) and word[i+1] in self._vowels:
                        break
                    syl_has_coda = True
                    i += 1
                
                # Determine weight
                is_heavy = syl_has_long_vowel or syl_has_coda
                syllables.append((word[syl_start:i], is_heavy))
            else:
                # No vowel found - skip this segment
                i += 1
        
        return syllables
    
    def get_prosodic_pattern(self, text: str) -> str:
        """
        Get the prosodic pattern (—/U sequence) for a line of poetry.
        
        This is what we use for ʿArūḍ meter matching.
        """
        syllables = self.syllabify_prosodic(text)
        return ''.join('—' if is_heavy else 'U' for _, is_heavy in syllables)
    
    def count_syllables(self, text: str) -> int:
        """Count the number of syllables in text."""
        return len(self.syllabify_prosodic(text))


# === TRANSLITERATION UTILITIES ===

class ScientificTransliterator:
    """
    Utility class for generating scientific transliterations.
    
    Useful for:
    - Citing Tajik poetry in academic papers
    - Creating standardized romanizations for indexes
    - Exporting data for cross-linguistic comparison
    """
    
    def __init__(self):
        self.phonology = TajikPhonology()
    
    def transliterate(self, text: str, system: str = 'ijmes') -> str:
        """
        Transliterate text to specified system.
        
        Args:
            text: Tajik Cyrillic or Arabic-script text
            system: 'ijmes' or 'dmg'
            
        Returns:
            Romanized text in specified system
        """
        if system.lower() == 'ijmes':
            return self.phonology.to_ijmes(text)
        elif system.lower() == 'dmg':
            return self.phonology.to_dmg(text)
        else:
            raise ValueError(f"Unknown transliteration system: {system}")
    
    def format_citation(self, title: str, author: str = None, 
                        system: str = 'ijmes') -> str:
        """
        Format a poem title/author for academic citation.
        
        Example:
            format_citation("ВАҚТ", "Дилором Солибоева")
            → "Waqt" (Dilorom Soliboeva)
        """
        title_trans = self.transliterate(title, system)
        # Capitalize first letter
        title_trans = title_trans[0].upper() + title_trans[1:] if title_trans else ""
        
        if author:
            author_trans = self.transliterate(author, system)
            # Capitalize name parts
            author_parts = author_trans.split()
            author_trans = ' '.join(p.capitalize() for p in author_parts)
            return f'"{title_trans}" ({author_trans})'
        
        return f'"{title_trans}"'


# === DEMONSTRATION ===

if __name__ == "__main__":
    phonology = TajikPhonology()
    transliterator = ScientificTransliterator()
    
    # Test text
    test_line = "Дар кӯҳсори ватан гулҳо мешукуфанд"
    
    print("=== Tajik Phonology System Demo ===\n")
    print(f"Input: {test_line}")
    print(f"IJMES: {phonology.to_ijmes(test_line)}")
    print(f"DMG:   {phonology.to_dmg(test_line)}")
    print(f"Prosodic pattern: {phonology.get_prosodic_pattern(test_line)}")
    print(f"Syllable count: {phonology.count_syllables(test_line)}")
    
    print("\n=== Syllable Analysis ===")
    for syl, is_heavy in phonology.syllabify_prosodic(test_line):
        weight = "—" if is_heavy else "U"
        print(f"  {syl:10} → {weight}")
    
    print("\n=== Citation Formatting ===")
    print(transliterator.format_citation("ВАҚТ", "Дилором Солибоева", "ijmes"))
    print(transliterator.format_citation("ВАҚТ", "Дилором Солибоева", "dmg"))
