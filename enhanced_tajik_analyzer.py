#!/usr/bin/env python3
"""
Enhanced Tajik Poetry Analyzer - Phase 1 Implementation
Scientific Research Grade with Proper ʿArūḍ Analysis

This implementation provides:
1. Proper ʿArūḍ (Classical Arabic-Persian prosody) analysis
2. Phonetic-based rhyme detection
3. Accurate syllable weight calculation
4. Scientific error handling and validation

Limitations explicitly acknowledged:
- Phonetic transcription requires manual phoneme mapping
- Some meters may need corpus validation
- Complex Persian morphophonology not fully implemented
"""

import re
import json
import logging
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import warnings

# Configure logging for scientific use
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('poetry_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SyllableWeight(Enum):
    """Prosodic weight classification"""
    HEAVY = "—"  # Long syllable
    LIGHT = "U"  # Short syllable
    UNKNOWN = "?"  # Uncertain weight


class MeterConfidence(Enum):
    """Confidence levels for meter identification"""
    HIGH = "high"  # >90% pattern match
    MEDIUM = "medium"  # 70-90% pattern match  
    LOW = "low"  # 50-70% pattern match
    NONE = "none"  # <50% pattern match


@dataclass
class ProsodicSyllable:
    """Represents a syllable with prosodic information"""
    text: str
    weight: SyllableWeight
    phonetic: Optional[str] = None
    position: int = 0
    confidence: float = 1.0


@dataclass
class AruzPattern:
    """Represents a classical ʿArūḍ meter pattern"""
    name: str
    pattern: str
    description: str
    variations: List[str] = field(default_factory=list)
    frequency_weight: float = 1.0  # How common this meter is


@dataclass
class PhoneticAnalysis:
    """Results of phonetic analysis"""
    phonetic_transcription: str
    syllable_boundaries: List[int]
    stress_pattern: List[int]
    confidence: float


@dataclass
class RhymeAnalysis:
    """Advanced rhyme analysis results"""
    qafiyeh: str  # The actual rhyming sound
    radif: str  # Repeated refrain after rhyme
    phonetic_rhyme: str  # Phonetic representation
    rhyme_type: str  # perfect, imperfect, eye-rhyme, etc.
    confidence: float


@dataclass
class AruzAnalysis:
    """Results of ʿArūḍ meter analysis"""
    identified_meter: str
    pattern_match: str
    confidence: MeterConfidence
    pattern_accuracy: float
    variations_detected: List[str]
    line_scansion: List[ProsodicSyllable]


class PersianPhonetics:
    """
    Persian/Tajik phonetic analysis system

    WARNING: This is a simplified implementation. For production research,
    you would need a full Persian phonological analyzer with:
    - Complete phoneme inventory
    - Morphophonological rules
    - Dialect variation handling
    """

    def __init__(self):
        # Basic Persian/Tajik phoneme mapping (simplified)
        # NOTE: This is incomplete - full implementation would need linguistic expertise
        self.phoneme_map = {
            # Consonants
            'ب': 'b', 'پ': 'p', 'ت': 't', 'ث': 's', 'ج': 'ʤ', 'چ': 'ʧ',
            'ح': 'ħ', 'خ': 'x', 'د': 'd', 'ذ': 'z', 'ر': 'r', 'ز': 'z',
            'ژ': 'ʒ', 'س': 's', 'ش': 'ʃ', 'ص': 's', 'ض': 'z', 'ط': 't',
            'ظ': 'z', 'ع': 'ʔ', 'غ': 'ɣ', 'ف': 'f', 'ق': 'q', 'ک': 'k',
            'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'و': 'w', 'ه': 'h',
            'ی': 'j',

            # Tajik Cyrillic additions
            'қ': 'q', 'ғ': 'ɣ', 'ҳ': 'h', 'ҷ': 'ʤ', 'ӣ': 'iː', 'ӯ': 'uː',

            # Vowels
            'а': 'a', 'о': 'o', 'у': 'u', 'е': 'e', 'и': 'i', 'э': 'e',
            'я': 'ja', 'ю': 'ju', 'ё': 'jo'
        }

        # Vowel categories for syllable analysis
        self.short_vowels = {'a', 'e', 'i', 'o', 'u'}
        self.long_vowels = {'aː', 'eː', 'iː', 'oː', 'uː'}
        self.diphthongs = {'aj', 'aw', 'oj', 'ej'}

        logger.info("PersianPhonetics initialized with basic phoneme mapping")
        logger.warning(
            "This is a simplified phonetic system - full research requires comprehensive phonological analysis")

    def to_phonetic(self, text: str) -> PhoneticAnalysis:
        """
        Convert text to phonetic representation

        LIMITATION: This is a basic character-to-phoneme mapping.
        Real implementation would need:
        - Morphophonological rules
        - Context-dependent pronunciations
        - Dialect variations
        """
        try:
            # Normalize text
            text = unicodedata.normalize('NFC', text.strip())

            # Basic phonetic conversion
            phonetic = ""
            for char in text:
                if char in self.phoneme_map:
                    phonetic += self.phoneme_map[char]
                elif char.isspace():
                    phonetic += " "
                else:
                    phonetic += char  # Keep unknown characters

            # Find syllable boundaries (simplified)
            syllable_boundaries = self._find_syllable_boundaries(phonetic)

            # Basic stress pattern (simplified - stress typically on last syllable in Persian)
            stress_pattern = [0] * len(syllable_boundaries)
            if syllable_boundaries:
                stress_pattern[-1] = 1  # Primary stress on last syllable

            return PhoneticAnalysis(
                phonetic_transcription=phonetic,
                syllable_boundaries=syllable_boundaries,
                stress_pattern=stress_pattern,
                confidence=0.7  # Medium confidence due to simplification
            )

        except Exception as e:
            logger.error(f"Phonetic analysis failed: {e}")
            return PhoneticAnalysis("", [], [], 0.0)

    def _find_syllable_boundaries(self, phonetic: str) -> List[int]:
        """
        Find syllable boundaries in phonetic string

        SIMPLIFIED: Real implementation needs full phonotactic constraints
        """
        boundaries = [0]
        vowel_positions = []

        # Find vowel positions
        i = 0
        while i < len(phonetic):
            if phonetic[i] in self.short_vowels or phonetic[i:i + 2] in self.long_vowels:
                vowel_positions.append(i)
                if phonetic[i:i + 2] in self.long_vowels:
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        # Simple syllable boundary detection
        for i in range(1, len(vowel_positions)):
            # Place boundary between vowels (simplified)
            boundary = (vowel_positions[i - 1] + vowel_positions[i]) // 2
            boundaries.append(boundary)

        if phonetic:
            boundaries.append(len(phonetic))

        return boundaries


class AruzMeterAnalyzer:
    """
    Classical ʿArūḍ (Arabic-Persian prosody) analyzer

    Implements the 16 classical Arabic meters adapted for Persian/Tajik poetry
    """

    def __init__(self):
        # Classical ʿArūḍ meters with their patterns
        # Pattern notation: — = heavy syllable, U = light syllable
        self.aruz_meters = {
            # The 16 classical meters
            "ṭawīl": AruzPattern(
                name="ṭawīl",
                pattern="U—UU—U—UU—",
                description="فعولن مفاعيلن فعولن مفاعيلن",
                variations=["U—UU—U—UU", "U—UU—U—U—"],
                frequency_weight=1.5  # Very common in classical poetry
            ),
            "basīṭ": AruzPattern(
                name="basīṭ",
                pattern="UU—U—UU—U—",
                description="مستفعلن فاعلن مستفعلن فاعلن",
                variations=["UU—U—UU—U", "UU—UUU—U—"],
                frequency_weight=1.2
            ),
            "wāfir": AruzPattern(
                name="wāfir",
                pattern="U—UU—UU—U",
                description="مفاعلتن مفاعلتن مفاعلتن",
                variations=["U—UU—UU—", "U—UUU—U"],
                frequency_weight=1.0
            ),
            "kāmil": AruzPattern(
                name="kāmil",
                pattern="UU—UUU—UU—",
                description="متفاعلن متفاعلن متفاعلن",
                variations=["UU—UU—UU—", "UU—UUU—U"],
                frequency_weight=1.0
            ),
            "mutaqārib": AruzPattern(
                name="mutaqārib",
                pattern="U—U—U—U—",
                description="فعولن فعولن فعولن فعولن",
                variations=["U—U—U—U", "UU—U—U—"],
                frequency_weight=0.8
            ),
            "hazaj": AruzPattern(
                name="hazaj",
                pattern="—U—U—U—U",
                description="مفعولن مفعولن مفعولن مفعولن",
                variations=["—U—U—U—", "U—U—U—U"],
                frequency_weight=0.9
            ),
            "rajaz": AruzPattern(
                name="rajaz",
                pattern="UU—UU—UU—",
                description="مستفعلن مستفعلن مستفعلن",
                variations=["UU—UU—U—", "UUU—UU—"],
                frequency_weight=0.7
            ),
            "ramal": AruzPattern(
                name="ramal",
                pattern="U—U—U—U—",
                description="فاعلاتن فاعلاتن فاعلاتن",
                variations=["U—UU—U—", "UU—U—U—"],
                frequency_weight=0.8
            ),
            "sarīʿ": AruzPattern(
                name="sarīʿ",
                pattern="UUU—U—UU—",
                description="مستفعلن مستفعلن مفعولات",
                variations=["UUU—UU—", "UU—U—UU—"],
                frequency_weight=0.6
            ),
            "munasarih": AruzPattern(
                name="munasarih",
                pattern="UU—U—UU—U",
                description="مستفعلن مفعولات مستفعلن",
                variations=["UU—U—UU—", "UUU—UU—U"],
                frequency_weight=0.5
            ),
            "khafīf": AruzPattern(
                name="khafīf",
                pattern="U—UU—U—U",
                description="فاعلاتن مستفعلن فاعلن",
                variations=["U—UU—U—", "UU—UU—U"],
                frequency_weight=0.6
            ),
            "muḍāriʿ": AruzPattern(
                name="muḍāriʿ",
                pattern="—U—U—U—",
                description="مفعولات مفعولات مفعولات",
                variations=["—U—U—U", "U—U—U—"],
                frequency_weight=0.4
            ),
            "muqtaḍab": AruzPattern(
                name="muqtaḍab",
                pattern="U—U—U—",
                description="مفعولات مفعولات",
                variations=["U—U—U", "UU—U—"],
                frequency_weight=0.3
            ),
            "mujtath": AruzPattern(
                name="mujtath",
                pattern="UU—U—",
                description="مستفعلن مستفعلن",
                variations=["UU—U", "UUU—"],
                frequency_weight=0.3
            ),
            "mutadārik": AruzPattern(
                name="mutadārik",
                pattern="U—U—U—U—",
                description="فاعلن فاعلن فاعلن فاعلن",
                variations=["U—U—U—", "UU—U—U"],
                frequency_weight=0.5
            )
        }

        self.phonetics = PersianPhonetics()
        logger.info(f"AruzMeterAnalyzer initialized with {len(self.aruz_meters)} classical meters")

    def analyze_meter(self, line: str) -> AruzAnalysis:
        """
        Analyze a line of poetry for ʿArūḍ meter
        """
        try:
            # Get phonetic analysis
            phonetic_analysis = self.phonetics.to_phonetic(line)

            # Calculate syllable weights
            syllables = self._extract_syllables(line, phonetic_analysis)

            if not syllables:
                logger.warning(f"No syllables found in line: {line[:50]}...")
                return self._create_empty_analysis()

            # Generate prosodic pattern
            pattern = "".join([syl.weight.value for syl in syllables])

            # Match against ʿArūḍ meters
            best_match = self._find_best_meter_match(pattern)

            return AruzAnalysis(
                identified_meter=best_match["meter"],
                pattern_match=best_match["pattern"],
                confidence=best_match["confidence"],
                pattern_accuracy=best_match["accuracy"],
                variations_detected=best_match["variations"],
                line_scansion=syllables
            )

        except Exception as e:
            logger.error(f"Meter analysis failed for line '{line[:50]}...': {e}")
            return self._create_empty_analysis()

    def _extract_syllables(self, line: str, phonetic: PhoneticAnalysis) -> List[ProsodicSyllable]:
        """
        Extract syllables with prosodic weights
        """
        syllables = []

        if not phonetic.syllable_boundaries or len(phonetic.syllable_boundaries) < 2:
            # Fallback to simple syllable detection
            return self._simple_syllable_extraction(line)

        # Extract syllables based on phonetic boundaries
        for i in range(len(phonetic.syllable_boundaries) - 1):
            start = phonetic.syllable_boundaries[i]
            end = phonetic.syllable_boundaries[i + 1]

            # Get syllable text (approximate mapping back to original)
            syl_text = self._map_phonetic_to_text(line, start, end, len(phonetic.syllable_boundaries) - 1)

            # Calculate prosodic weight
            weight = self._calculate_syllable_weight(
                phonetic.phonetic_transcription[start:end],
                i == len(phonetic.syllable_boundaries) - 2  # is_final
            )

            syllables.append(ProsodicSyllable(
                text=syl_text,
                weight=weight,
                phonetic=phonetic.phonetic_transcription[start:end],
                position=i,
                confidence=phonetic.confidence
            ))

        return syllables

    def _simple_syllable_extraction(self, line: str) -> List[ProsodicSyllable]:
        """
        Fallback simple syllable extraction when phonetic analysis fails
        """
        # Remove diacritics and normalize
        line = unicodedata.normalize('NFC', line)
        line = re.sub(r'[\u064B-\u065F]', '', line)

        # Basic vowel-based syllable detection
        vowels = "аоуеиӣэяюёāūī"
        diphthongs = ["ай", "уй", "ой", "ӯй", "аӣ", "оӣ", "уӣ"]

        syllables = []
        i = 0
        syl_count = 0

        while i < len(line):
            if line[i].isalpha():
                # Find syllable boundary
                syl_start = i
                nucleus_found = False

                # Look for vowel nucleus
                while i < len(line) and line[i].isalpha():
                    # Check for diphthongs first
                    if i < len(line) - 1:
                        diphthong = line[i:i + 2]
                        if diphthong in diphthongs:
                            nucleus_found = True
                            i += 2
                            break

                    if line[i] in vowels:
                        nucleus_found = True
                        i += 1
                        break
                    i += 1

                # Continue to syllable end
                while i < len(line) and line[i].isalpha() and line[i] not in vowels:
                    i += 1

                if nucleus_found:
                    syl_text = line[syl_start:i]
                    weight = self._estimate_syllable_weight_simple(syl_text)

                    syllables.append(ProsodicSyllable(
                        text=syl_text,
                        weight=weight,
                        position=syl_count,
                        confidence=0.6  # Lower confidence for simple method
                    ))
                    syl_count += 1
            else:
                i += 1

        return syllables

    def _calculate_syllable_weight(self, phonetic_syl: str, is_final: bool) -> SyllableWeight:
        """
        Calculate syllable weight based on Persian prosodic rules

        Persian prosodic rules (simplified):
        - Heavy: CVV (long vowel), CVC at word end, CVC before consonant cluster
        - Light: CV, CVC in other positions
        """
        if not phonetic_syl:
            return SyllableWeight.UNKNOWN

        # Count vowels and consonants
        vowel_count = sum(1 for char in phonetic_syl if char in self.phonetics.short_vowels)
        long_vowel_count = sum(1 for i in range(len(phonetic_syl) - 1)
                               if phonetic_syl[i:i + 2] in self.phonetics.long_vowels)

        # Check for long vowels or diphthongs -> Heavy
        if long_vowel_count > 0 or any(diphthong in phonetic_syl for diphthong in self.phonetics.diphthongs):
            return SyllableWeight.HEAVY

        # Check syllable structure
        consonant_cluster = re.search(r'[bcdfghjklmnpqrstvwxyz]{2,}', phonetic_syl.lower())
        ends_with_consonant = phonetic_syl and phonetic_syl[-1] not in self.phonetics.short_vowels

        # Final syllable ending in consonant -> Heavy
        if is_final and ends_with_consonant:
            return SyllableWeight.HEAVY

        # CVC before consonant cluster -> Heavy
        if consonant_cluster and ends_with_consonant:
            return SyllableWeight.HEAVY

        # Default to light
        return SyllableWeight.LIGHT

    def _estimate_syllable_weight_simple(self, syllable: str) -> SyllableWeight:
        """
        Simple syllable weight estimation for fallback
        """
        # Long vowel markers
        if any(marker in syllable for marker in ['ā', 'ī', 'ū', 'ӣ', 'ӯ']):
            return SyllableWeight.HEAVY

        # Diphthongs
        if any(diphthong in syllable for diphthong in ['ай', 'уй', 'ой']):
            return SyllableWeight.HEAVY

        # Simple heuristic: more than 2 characters often indicates complexity
        if len(syllable) > 2:
            return SyllableWeight.HEAVY

        return SyllableWeight.LIGHT

    def _map_phonetic_to_text(self, original: str, start: int, end: int, total_syllables: int) -> str:
        """
        Approximate mapping from phonetic positions back to original text
        This is simplified - real implementation would need alignment algorithm
        """
        # Simple approximation
        chars_per_syllable = len(original) // max(1, total_syllables)
        text_start = start * chars_per_syllable // max(1, len(original))
        text_end = end * chars_per_syllable // max(1, len(original))

        return original[text_start:text_end].strip()

    def _find_best_meter_match(self, pattern: str) -> Dict[str, Any]:
        """
        Find the best matching ʿArūḍ meter for the given pattern
        """
        best_match = {
            "meter": "unknown",
            "pattern": pattern,
            "confidence": MeterConfidence.NONE,
            "accuracy": 0.0,
            "variations": []
        }

        best_score = 0.0

        for meter_name, meter_info in self.aruz_meters.items():
            # Test main pattern
            score = self._pattern_similarity(pattern, meter_info.pattern)

            # Test variations
            variation_scores = []
            for variation in meter_info.variations:
                var_score = self._pattern_similarity(pattern, variation)
                variation_scores.append((variation, var_score))

            # Take best score (main pattern or variation)
            max_variation_score = max([score] + [s for _, s in variation_scores])

            # Weight by frequency
            weighted_score = max_variation_score * meter_info.frequency_weight

            if weighted_score > best_score:
                best_score = weighted_score
                best_match.update({
                    "meter": meter_name,
                    "pattern": meter_info.pattern,
                    "accuracy": max_variation_score,
                    "variations": [var for var, s in variation_scores if s > 0.7]
                })

        # Determine confidence level
        if best_score >= 0.9:
            best_match["confidence"] = MeterConfidence.HIGH
        elif best_score >= 0.7:
            best_match["confidence"] = MeterConfidence.MEDIUM
        elif best_score >= 0.5:
            best_match["confidence"] = MeterConfidence.LOW
        else:
            best_match["confidence"] = MeterConfidence.NONE

        return best_match

    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """
        Calculate similarity between two prosodic patterns
        Uses Levenshtein-like algorithm adapted for prosodic patterns
        """
        if not pattern1 or not pattern2:
            return 0.0

        # Normalize lengths for comparison
        len1, len2 = len(pattern1), len(pattern2)
        max_len = max(len1, len2)

        if max_len == 0:
            return 1.0

        return matches / max_len

    def _create_empty_rhyme_analysis(self) -> RhymeAnalysis:
        """Create empty rhyme analysis for error cases"""
        return RhymeAnalysis(
            qafiyeh="",
            radif="",
            phonetic_rhyme="",
            rhyme_type="none",
            confidence=0.0
        )


class EnhancedSyllableAnalyzer:
    """
    Enhanced syllable analysis with proper weight calculation
    """

    def __init__(self):
        self.phonetics = PersianPhonetics()
        logger.info("EnhancedSyllableAnalyzer initialized")

    def analyze_syllable_structure(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive syllable analysis
        """
        try:
            phonetic_analysis = self.phonetics.to_phonetic(text)

            if not phonetic_analysis.syllable_boundaries:
                logger.warning(f"No syllable boundaries found for: {text[:50]}...")
                return self._create_empty_syllable_analysis()

            syllables = []
            total_weight = 0

            for i in range(len(phonetic_analysis.syllable_boundaries) - 1):
                start = phonetic_analysis.syllable_boundaries[i]
                end = phonetic_analysis.syllable_boundaries[i + 1]

                syl_phonetic = phonetic_analysis.phonetic_transcription[start:end]
                is_final = (i == len(phonetic_analysis.syllable_boundaries) - 2)

                # Calculate weight
                weight = self._calculate_detailed_weight(syl_phonetic, is_final)
                syllables.append({
                    'text': syl_phonetic,
                    'weight': weight.value,
                    'position': i,
                    'is_final': is_final
                })

                if weight == SyllableWeight.HEAVY:
                    total_weight += 2
                else:
                    total_weight += 1

            return {
                'syllables': syllables,
                'total_syllables': len(syllables),
                'heavy_syllables': sum(1 for s in syllables if s['weight'] == '—'),
                'light_syllables': sum(1 for s in syllables if s['weight'] == 'U'),
                'total_prosodic_weight': total_weight,
                'phonetic_confidence': phonetic_analysis.confidence,
                'stress_pattern': phonetic_analysis.stress_pattern
            }

        except Exception as e:
            logger.error(f"Syllable analysis failed: {e}")
            return self._create_empty_syllable_analysis()

    def _calculate_detailed_weight(self, phonetic_syl: str, is_final: bool) -> SyllableWeight:
        """
        Detailed syllable weight calculation with Persian phonotactics
        """
        if not phonetic_syl:
            return SyllableWeight.UNKNOWN

        # Check for long vowels
        if any(lv in phonetic_syl for lv in self.phonetics.long_vowels):
            return SyllableWeight.HEAVY

        # Check for diphthongs
        if any(diphthong in phonetic_syl for diphthong in self.phonetics.diphthongs):
            return SyllableWeight.HEAVY

        # Consonant cluster analysis
        consonants = re.findall(r'[bcdfghjklmnpqrstvwxyzħʔɣʤʧʒʃ]+', phonetic_syl)

        # CVC at word end -> Heavy
        if is_final and consonants and phonetic_syl[-1] not in self.phonetics.short_vowels:
            return SyllableWeight.HEAVY

        # CVC before consonant cluster -> Heavy
        if len(consonants) > 1:
            return SyllableWeight.HEAVY

        return SyllableWeight.LIGHT

    def _create_empty_syllable_analysis(self) -> Dict[str, Any]:
        """Create empty analysis for error cases"""
        return {
            'syllables': [],
            'total_syllables': 0,
            'heavy_syllables': 0,
            'light_syllables': 0,
            'total_prosodic_weight': 0,
            'phonetic_confidence': 0.0,
            'stress_pattern': []
        }


@dataclass
class EnhancedStructuralAnalysis:
    """Enhanced structural analysis results"""
    lines: int
    syllable_analysis: Dict[str, Any]
    aruz_analysis: AruzAnalysis
    rhyme_scheme: List[RhymeAnalysis]
    rhyme_pattern: str
    avg_syllables: float
    prosodic_consistency: float
    meter_confidence: MeterConfidence


class EnhancedStructuralAnalyzer:
    """
    Enhanced structural analyzer integrating all Phase 1 improvements
    """

    def __init__(self, config):
        self.config = config
        self.aruz_analyzer = AruzMeterAnalyzer()
        self.rhyme_analyzer = AdvancedRhymeAnalyzer()
        self.syllable_analyzer = EnhancedSyllableAnalyzer()

        logger.info("EnhancedStructuralAnalyzer initialized with all Phase 1 components")

    def analyze(self, poem_content: str) -> EnhancedStructuralAnalysis:
        """
        Comprehensive structural analysis
        """
        try:
            lines = [line.strip() for line in poem_content.split('\n') if line.strip()]

            if not lines:
                raise ValueError("No valid lines found in poem")

            # Analyze each line
            line_analyses = []
            syllable_counts = []

            for line in lines:
                # Aruz analysis
                aruz = self.aruz_analyzer.analyze_meter(line)

                # Rhyme analysis
                rhyme = self.rhyme_analyzer.analyze_rhyme(line)

                # Syllable analysis
                syllables = self.syllable_analyzer.analyze_syllable_structure(line)

                line_analyses.append({
                    'line': line,
                    'aruz': aruz,
                    'rhyme': rhyme,
                    'syllables': syllables
                })

                syllable_counts.append(syllables['total_syllables'])

            # Generate overall rhyme scheme
            rhyme_pattern = self._generate_rhyme_pattern([la['rhyme'] for la in line_analyses])

            # Calculate prosodic consistency
            prosodic_consistency = self._calculate_prosodic_consistency(line_analyses)

            # Determine overall meter confidence
            meter_confidences = [la['aruz'].confidence for la in line_analyses]
            overall_meter_confidence = self._determine_overall_confidence(meter_confidences)

            # Aggregate syllable analysis
            total_syllables = sum(syllable_counts)
            avg_syllables = total_syllables / len(lines) if lines else 0

            aggregate_syllable_analysis = {
                'total_syllables': total_syllables,
                'avg_syllables_per_line': avg_syllables,
                'syllable_distribution': syllable_counts,
                'line_analyses': [la['syllables'] for la in line_analyses]
            }

            return EnhancedStructuralAnalysis(
                lines=len(lines),
                syllable_analysis=aggregate_syllable_analysis,
                aruz_analysis=line_analyses[0]['aruz'] if line_analyses else AruzAnalysis("unknown", "",
                                                                                          MeterConfidence.NONE, 0.0, [],
                                                                                          []),
                rhyme_scheme=[la['rhyme'] for la in line_analyses],
                rhyme_pattern=rhyme_pattern,
                avg_syllables=round(avg_syllables, 2),
                prosodic_consistency=prosodic_consistency,
                meter_confidence=overall_meter_confidence
            )

        except Exception as e:
            logger.error(f"Enhanced structural analysis failed: {e}")
            raise

    def _generate_rhyme_pattern(self, rhyme_analyses: List[RhymeAnalysis]) -> str:
        """
        Generate rhyme scheme pattern based on phonetic similarity
        """
        if not rhyme_analyses:
            return ""

        pattern = []
        rhyme_groups = {}
        next_label = 'A'

        for rhyme in rhyme_analyses:
            # Find matching group
            matched_group = None

            for existing_rhyme, label in rhyme_groups.items():
                similarity = self.rhyme_analyzer.calculate_rhyme_similarity(rhyme, existing_rhyme)
                if similarity > 0.7:  # Threshold for rhyme match
                    matched_group = label
                    break

            if matched_group:
                pattern.append(matched_group)
            else:
                pattern.append(next_label)
                rhyme_groups[rhyme] = next_label
                next_label = chr(ord(next_label) + 1)

        return ''.join(pattern)

    def _calculate_prosodic_consistency(self, line_analyses: List[Dict]) -> float:
        """
        Calculate how consistent the prosodic pattern is across lines
        """
        if not line_analyses:
            return 0.0

        # Check meter consistency
        meters = [la['aruz'].identified_meter for la in line_analyses]
        most_common_meter = max(set(meters), key=meters.count) if meters else "unknown"
        meter_consistency = meters.count(most_common_meter) / len(meters)

        # Check syllable count consistency
        syllable_counts = [la['syllables']['total_syllables'] for la in line_analyses]
        if syllable_counts:
            avg_syllables = sum(syllable_counts) / len(syllable_counts)
            syllable_variance = sum((c - avg_syllables) ** 2 for c in syllable_counts) / len(syllable_counts)
            syllable_consistency = max(0, 1 - (syllable_variance / avg_syllables) if avg_syllables > 0 else 0)
        else:
            syllable_consistency = 0

        # Combine metrics
        return (meter_consistency + syllable_consistency) / 2

    def _determine_overall_confidence(self, confidences: List[MeterConfidence]) -> MeterConfidence:
        """
        Determine overall meter confidence from individual line confidences
        """
        if not confidences:
            return MeterConfidence.NONE

        # Count confidence levels
        confidence_counts = {conf: confidences.count(conf) for conf in MeterConfidence}

        # Determine based on majority
        total_lines = len(confidences)

        if confidence_counts[MeterConfidence.HIGH] / total_lines >= 0.7:
            return MeterConfidence.HIGH
        elif confidence_counts[MeterConfidence.MEDIUM] / total_lines >= 0.5:
            return MeterConfidence.MEDIUM
        elif confidence_counts[MeterConfidence.LOW] / total_lines >= 0.3:
            return MeterConfidence.LOW
        else:
            return MeterConfidence.NONE


class ScientificValidator:
    """
    Validation and quality control for scientific research
    """

    @staticmethod
    def validate_analysis_quality(analysis: EnhancedStructuralAnalysis) -> Dict[str, Any]:
        """
        Validate analysis quality and provide warnings
        """
        warnings_list = []
        quality_score = 1.0

        # Check meter confidence
        if analysis.meter_confidence == MeterConfidence.NONE:
            warnings_list.append("No reliable meter detected - results may be unreliable")
            quality_score *= 0.5
        elif analysis.meter_confidence == MeterConfidence.LOW:
            warnings_list.append("Low meter confidence - manual verification recommended")
            quality_score *= 0.7

        # Check prosodic consistency
        if analysis.prosodic_consistency < 0.5:
            warnings_list.append("Low prosodic consistency - poem may be free verse or damaged")
            quality_score *= 0.6

        # Check syllable analysis
        if not analysis.syllable_analysis['line_analyses']:
            warnings_list.append("Syllable analysis failed - phonetic transcription may be unreliable")
            quality_score *= 0.4

        # Check line count
        if analysis.lines < 2:
            warnings_list.append("Very short poem - statistical analysis not reliable")
            quality_score *= 0.3

        return {
            'quality_score': quality_score,
            'warnings': warnings_list,
            'reliability_level': ScientificValidator._determine_reliability(quality_score),
            'recommended_actions': ScientificValidator._generate_recommendations(warnings_list)
        }

    @staticmethod
    def _determine_reliability(quality_score: float) -> str:
        """Determine reliability level based on quality score"""
        if quality_score >= 0.8:
            return "high"
        elif quality_score >= 0.6:
            return "medium"
        elif quality_score >= 0.4:
            return "low"
        else:
            return "unreliable"

    @staticmethod
    def _generate_recommendations(warnings: List[str]) -> List[str]:
        """Generate recommendations based on warnings"""
        recommendations = []

        if any("meter" in w.lower() for w in warnings):
            recommendations.append("Consider manual meter verification by prosody expert")
            recommendations.append("Check if poem follows non-classical metrical patterns")

        if any("syllable" in w.lower() for w in warnings):
            recommendations.append("Verify phonetic transcription accuracy")
            recommendations.append("Consider dialect-specific pronunciation rules")

        if any("consistency" in w.lower() for w in warnings):
            recommendations.append("Analyze if poem intentionally varies meter")
            recommendations.append("Check for textual corruption or transcription errors")

        return recommendations


# Main integration class
class EnhancedTajikPoemAnalyzer:
    """
    Main analyzer class with Phase 1 enhancements integrated
    """

    def __init__(self, config=None):
        self.config = config or AnalysisConfig()
        self.enhanced_structural_analyzer = EnhancedStructuralAnalyzer(self.config)
        self.validator = ScientificValidator()

        logger.info("EnhancedTajikPoemAnalyzer initialized with Phase 1 improvements")

    def analyze_poem_enhanced(self, poem_content: str) -> Dict[str, Any]:
        """
        Enhanced poem analysis with validation
        """
        try:
            # Perform enhanced structural analysis
            structural_analysis = self.enhanced_structural_analyzer.analyze(poem_content)

            # Validate results
            validation = self.validator.validate_analysis_quality(structural_analysis)

            # Log warnings if any
            for warning in validation['warnings']:
                logger.warning(f"Analysis warning: {warning}")

            return {
                'structural_analysis': structural_analysis,
                'validation': validation,
                'metadata': {
                    'analysis_timestamp': "timestamp_unavailable",
                    'analyzer_version': "Phase1_Enhanced",
                    'phonetic_confidence': getattr(structural_analysis.aruz_analysis, 'line_scansion', [{}])[
                        0].confidence if structural_analysis.aruz_analysis.line_scansion else 0.0
                }
            }

        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            raise


def main():
    """
    Demonstration of enhanced analyzer with scientific validation
    """
    # Sample Tajik poetry for testing
    sample_poem = """
    Дар кӯҳсори ватан гулҳо мешуканд,
    Дили шоир аз муҳаббат меларазад.

    Баҳори нав ба замин таҷдид меорад,
    Навиди хушҳолии мардум мерасад.
    """

    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedTajikPoemAnalyzer()

        # Perform analysis
        result = analyzer.analyze_poem_enhanced(sample_poem)

        # Display results
        print("=== ENHANCED TAJIK POETRY ANALYSIS ===\n")

        structural = result['structural_analysis']
        validation = result['validation']

        print(f"Lines: {structural.lines}")
        print(f"Average syllables per line: {structural.avg_syllables}")
        print(f"Identified meter: {structural.aruz_analysis.identified_meter}")
        print(f"Meter confidence: {structural.meter_confidence.value}")
        print(f"Rhyme pattern: {structural.rhyme_pattern}")
        print(f"Prosodic consistency: {structural.prosodic_consistency:.2f}")

        print(f"\n=== QUALITY VALIDATION ===")
        print(f"Quality score: {validation['quality_score']:.2f}")
        print(f"Reliability level: {validation['reliability_level']}")

        if validation['warnings']:
            print(f"\nWarnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        if validation['recommended_actions']:
            print(f"\nRecommendations:")
            for rec in validation['recommended_actions']:
                print(f"  - {rec}")

        # Display detailed line analysis
        print(f"\n=== DETAILED LINE ANALYSIS ===")
        for i, rhyme in enumerate(structural.rhyme_scheme):
            print(f"Line {i + 1}:")
            print(f"  Qāfiyeh: {rhyme.qafiyeh}")
            print(f"  Radīf: {rhyme.radif}")
            print(f"  Rhyme type: {rhyme.rhyme_type}")
            print(f"  Confidence: {rhyme.confidence:.2f}")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()


class EnhancedSyllableAnalyzer:
    """
    Enhanced syllable analysis with proper weight calculation
    """

    def __init__(self):
        self.phonetics = PersianPhonetics()
        logger.info("EnhancedSyllableAnalyzer initialized")

    def analyze_syllable_structure(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive syllable analysis
        """
        try:
            phonetic_analysis = self.phonetics.to_phonetic(text)

            if not phonetic_analysis.syllable_boundaries:
                logger.warning(f"No syllable boundaries found for: {text[:50]}...")
                return self._create_empty_syllable_analysis()

            syllables = []
            total_weight = 0

            for i in range(len(phonetic_analysis.syllable_boundaries) - 1):
                start = phonetic_analysis.syllable_boundaries[i]
                end = phonetic_analysis.syllable_boundaries[i + 1]

                syl_phonetic = phonetic_analysis.phonetic_transcription[start:end]
                is_final = (i == len(phonetic_analysis.syllable_boundaries) - 2)

                # Calculate weight
                weight = self._calculate_detailed_weight(syl_phonetic, is_final)
                syllables.append({
                    'text': syl_phonetic,
                    'weight': weight.value,
                    'position': i,
                    'is_final': is_final
                })

                if weight == SyllableWeight.HEAVY:
                    total_weight += 2
                else:
                    total_weight += 1

            return {
                'syllables': syllables,
                'total_syllables': len(syllables),
                'heavy_syllables': sum(1 for s in syllables if s['weight'] == '—'),
                'light_syllables': sum(1 for s in syllables if s['weight'] == 'U'),
                'total_prosodic_weight': total_weight,
                'phonetic_confidence': phonetic_analysis.confidence,
                'stress_pattern': phonetic_analysis.stress_pattern
            }

        except Exception as e:
            logger.error(f"Syllable analysis failed: {e}")
            return self._create_empty_syllable_analysis()

    def _calculate_detailed_weight(self, phonetic_syl: str, is_final: bool) -> SyllableWeight:
        """
        Detailed syllable weight calculation with Persian phonotactics
        """
        if not phonetic_syl:
            return SyllableWeight.UNKNOWN

        # Check for long vowels
        if any(lv in phonetic_syl for lv in self.phonetics.long_vowels):
            return SyllableWeight.HEAVY

        # Check for diphthongs
        if any(diphthong in phonetic_syl for diphthong in self.phonetics.diphthongs):
            return SyllableWeight.HEAVY

        # Consonant cluster analysis
        consonants = re.findall(r'[bcdfghjklmnpqrstvwxyzħʔɣʤʧʒʃ]+', phonetic_syl)

        # CVC at word end -> Heavy
        if is_final and consonants and phonetic_syl[-1] not in self.phonetics.short_vowels:
            return SyllableWeight.HEAVY

        # CVC before consonant cluster -> Heavy
        if len(consonants) > 1:
            return SyllableWeight.HEAVY

        return SyllableWeight.LIGHT

    def _create_empty_syllable_analysis(self) -> Dict[str, Any]:
        """Create empty analysis for error cases"""
        return {
            'syllables': [],
            'total_syllables': 0,
            'heavy_syllables': 0,
            'light_syllables': 0,
            'total_prosodic_weight': 0,
            'phonetic_confidence': 0.0,
            'stress_pattern': []
        }


@dataclass
class EnhancedStructuralAnalysis:
    """Enhanced structural analysis results"""
    lines: int
    syllable_analysis: Dict[str, Any]
    aruz_analysis: AruzAnalysis
    rhyme_scheme: List[RhymeAnalysis]
    rhyme_pattern: str
    avg_syllables: float
    prosodic_consistency: float
    meter_confidence: MeterConfidence


class EnhancedStructuralAnalyzer:
    """
    Enhanced structural analyzer integrating all Phase 1 improvements
    """

    def __init__(self, config):
        self.config = config
        self.aruz_analyzer = AruzMeterAnalyzer()
        self.rhyme_analyzer = AdvancedRhymeAnalyzer()
        self.syllable_analyzer = EnhancedSyllableAnalyzer()

        logger.info("EnhancedStructuralAnalyzer initialized with all Phase 1 components")

    def analyze(self, poem_content: str) -> EnhancedStructuralAnalysis:
        """
        Comprehensive structural analysis
        """
        try:
            lines = [line.strip() for line in poem_content.split('\n') if line.strip()]

            if not lines:
                raise ValueError("No valid lines found in poem")

            # Analyze each line
            line_analyses = []
            syllable_counts = []

            for line in lines:
                # Aruz analysis
                aruz = self.aruz_analyzer.analyze_meter(line)

                # Rhyme analysis
                rhyme = self.rhyme_analyzer.analyze_rhyme(line)

                # Syllable analysis
                syllables = self.syllable_analyzer.analyze_syllable_structure(line)

                line_analyses.append({
                    'line': line,
                    'aruz': aruz,
                    'rhyme': rhyme,
                    'syllables': syllables
                })

                syllable_counts.append(syllables['total_syllables'])

            # Generate overall rhyme scheme
            rhyme_pattern = self._generate_rhyme_pattern([la['rhyme'] for la in line_analyses])

            # Calculate prosodic consistency
            prosodic_consistency = self._calculate_prosodic_consistency(line_analyses)

            # Determine overall meter confidence
            meter_confidences = [la['aruz'].confidence for la in line_analyses]
            overall_meter_confidence = self._determine_overall_confidence(meter_confidences)

            # Aggregate syllable analysis
            total_syllables = sum(syllable_counts)
            avg_syllables = total_syllables / len(lines) if lines else 0

            aggregate_syllable_analysis = {
                'total_syllables': total_syllables,
                'avg_syllables_per_line': avg_syllables,
                'syllable_distribution': syllable_counts,
                'line_analyses': [la['syllables'] for la in line_analyses]
            }

            return EnhancedStructuralAnalysis(
                lines=len(lines),
                syllable_analysis=aggregate_syllable_analysis,
                aruz_analysis=line_analyses[0]['aruz'] if line_analyses else AruzAnalysis("unknown", "",
                                                                                          MeterConfidence.NONE, 0.0, [],
                                                                                          []),
                rhyme_scheme=[la['rhyme'] for la in line_analyses],
                rhyme_pattern=rhyme_pattern,
                avg_syllables=round(avg_syllables, 2),
                prosodic_consistency=prosodic_consistency,
                meter_confidence=overall_meter_confidence
            )

        except Exception as e:
            logger.error(f"Enhanced structural analysis failed: {e}")
            raise

    def _generate_rhyme_pattern(self, rhyme_analyses: List[RhymeAnalysis]) -> str:
        """
        Generate rhyme scheme pattern based on phonetic similarity
        """
        if not rhyme_analyses:
            return ""

        pattern = []
        rhyme_groups = {}
        next_label = 'A'

        for rhyme in rhyme_analyses:
            # Find matching group
            matched_group = None

            for existing_rhyme, label in rhyme_groups.items():
                similarity = self.rhyme_analyzer.calculate_rhyme_similarity(rhyme, existing_rhyme)
                if similarity > 0.7:  # Threshold for rhyme match
                    matched_group = label
                    break

            if matched_group:
                pattern.append(matched_group)
            else:
                pattern.append(next_label)
                rhyme_groups[rhyme] = next_label
                next_label = chr(ord(next_label) + 1)

        return ''.join(pattern)

    def _calculate_prosodic_consistency(self, line_analyses: List[Dict]) -> float:
        """
        Calculate how consistent the prosodic pattern is across lines
        """
        if not line_analyses:
            return 0.0

        # Check meter consistency
        meters = [la['aruz'].identified_meter for la in line_analyses]
        most_common_meter = max(set(meters), key=meters.count) if meters else "unknown"
        meter_consistency = meters.count(most_common_meter) / len(meters)

        # Check syllable count consistency
        syllable_counts = [la['syllables']['total_syllables'] for la in line_analyses]
        if syllable_counts:
            avg_syllables = sum(syllable_counts) / len(syllable_counts)
            syllable_variance = sum((c - avg_syllables) ** 2 for c in syllable_counts) / len(syllable_counts)
            syllable_consistency = max(0, 1 - (syllable_variance / avg_syllables) if avg_syllables > 0 else 0)
        else:
            syllable_consistency = 0

        # Combine metrics
        return (meter_consistency + syllable_consistency) / 2

    def _determine_overall_confidence(self, confidences: List[MeterConfidence]) -> MeterConfidence:
        """
        Determine overall meter confidence from individual line confidences
        """
        if not confidences:
            return MeterConfidence.NONE

        # Count confidence levels
        confidence_counts = {conf: confidences.count(conf) for conf in MeterConfidence}

        # Determine based on majority
        total_lines = len(confidences)

        if confidence_counts[MeterConfidence.HIGH] / total_lines >= 0.7:
            return MeterConfidence.HIGH
        elif confidence_counts[MeterConfidence.MEDIUM] / total_lines >= 0.5:
            return MeterConfidence.MEDIUM
        elif confidence_counts[MeterConfidence.LOW] / total_lines >= 0.3:
            return MeterConfidence.LOW
        else:
            return MeterConfidence.NONE


class ScientificValidator:
    """
    Validation and quality control for scientific research
    """

    @staticmethod
    def validate_analysis_quality(analysis: EnhancedStructuralAnalysis) -> Dict[str, Any]:
        """
        Validate analysis quality and provide warnings
        """
        warnings_list = []
        quality_score = 1.0

        # Check meter confidence
        if analysis.meter_confidence == MeterConfidence.NONE:
            warnings_list.append("No reliable meter detected - results may be unreliable")
            quality_score *= 0.5
        elif analysis.meter_confidence == MeterConfidence.LOW:
            warnings_list.append("Low meter confidence - manual verification recommended")
            quality_score *= 0.7

        # Check prosodic consistency
        if analysis.prosodic_consistency < 0.5:
            warnings_list.append("Low prosodic consistency - poem may be free verse or damaged")
            quality_score *= 0.6

        # Check syllable analysis
        if not analysis.syllable_analysis['line_analyses']:
            warnings_list.append("Syllable analysis failed - phonetic transcription may be unreliable")
            quality_score *= 0.4

        # Check line count
        if analysis.lines < 2:
            warnings_list.append("Very short poem - statistical analysis not reliable")
            quality_score *= 0.3

        return {
            'quality_score': quality_score,
            'warnings': warnings_list,
            'reliability_level': ScientificValidator._determine_reliability(quality_score),
            'recommended_actions': ScientificValidator._generate_recommendations(warnings_list)
        }

    @staticmethod
    def _determine_reliability(quality_score: float) -> str:
        """Determine reliability level based on quality score"""
        if quality_score >= 0.8:
            return "high"
        elif quality_score >= 0.6:
            return "medium"
        elif quality_score >= 0.4:
            return "low"
        else:
            return "unreliable"

    @staticmethod
    def _generate_recommendations(warnings: List[str]) -> List[str]:
        """Generate recommendations based on warnings"""
        recommendations = []

        if any("meter" in w.lower() for w in warnings):
            recommendations.append("Consider manual meter verification by prosody expert")
            recommendations.append("Check if poem follows non-classical metrical patterns")

        if any("syllable" in w.lower() for w in warnings):
            recommendations.append("Verify phonetic transcription accuracy")
            recommendations.append("Consider dialect-specific pronunciation rules")

        if any("consistency" in w.lower() for w in warnings):
            recommendations.append("Analyze if poem intentionally varies meter")
            recommendations.append("Check for textual corruption or transcription errors")

        return recommendations


# Main integration class
class EnhancedTajikPoemAnalyzer:
    """
    Main analyzer class with Phase 1 enhancements integrated
    """

    def __init__(self, config=None):
        self.config = config or AnalysisConfig()
        self.enhanced_structural_analyzer = EnhancedStructuralAnalyzer(self.config)
        self.validator = ScientificValidator()

        logger.info("EnhancedTajikPoemAnalyzer initialized with Phase 1 improvements")

    def analyze_poem_enhanced(self, poem_content: str) -> Dict[str, Any]:
        """
        Enhanced poem analysis with validation
        """
        try:
            # Perform enhanced structural analysis
            structural_analysis = self.enhanced_structural_analyzer.analyze(poem_content)

            # Validate results
            validation = self.validator.validate_analysis_quality(structural_analysis)

            # Log warnings if any
            for warning in validation['warnings']:
                logger.warning(f"Analysis warning: {warning}")

            return {
                'structural_analysis': structural_analysis,
                'validation': validation,
                'metadata': {
                    'analysis_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else "timestamp_unavailable",
                    'analyzer_version': "Phase1_Enhanced",
                    'phonetic_confidence': getattr(structural_analysis.aruz_analysis, 'line_scansion', [{}])[
                        0].confidence if structural_analysis.aruz_analysis.line_scansion else 0.0
                }
            }

        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            raise


def main():
    """
    Demonstration of enhanced analyzer with scientific validation
    """
    # Sample Tajik poetry for testing
    sample_poem = """
    Дар кӯҳсори ватан гулҳо мешуканд,
    Дили шоир аз муҳаббат меларазад.

    Баҳори нав ба замин таҷдид меорад,
    Навиди хушҳолии мардум мерасад.
    """

    try:
        # Initialize enhanced analyzer
        analyzer = EnhancedTajikPoemAnalyzer()

        # Perform analysis
        result = analyzer.analyze_poem_enhanced(sample_poem)

        # Display results
        print("=== ENHANCED TAJIK POETRY ANALYSIS ===\n")

        structural = result['structural_analysis']
        validation = result['validation']

        print(f"Lines: {structural.lines}")
        print(f"Average syllables per line: {structural.avg_syllables}")
        print(f"Identified meter: {structural.aruz_analysis.identified_meter}")
        print(f"Meter confidence: {structural.meter_confidence.value}")
        print(f"Rhyme pattern: {structural.rhyme_pattern}")
        print(f"Prosodic consistency: {structural.prosodic_consistency:.2f}")

        print(f"\n=== QUALITY VALIDATION ===")
        print(f"Quality score: {validation['quality_score']:.2f}")
        print(f"Reliability level: {validation['reliability_level']}")

        if validation['warnings']:
            print(f"\nWarnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        if validation['recommended_actions']:
            print(f"\nRecommendations:")
            for rec in validation['recommended_actions']:
                print(f"  - {rec}")

        # Display detailed line analysis
        print(f"\n=== DETAILED LINE ANALYSIS ===")
        for i, rhyme in enumerate(structural.rhyme_scheme):
            print(f"Line {i + 1}:")
            print(f"  Qāfiyeh: {rhyme.qafiyeh}")
            print(f"  Radīf: {rhyme.radif}")
            print(f"  Rhyme type: {rhyme.rhyme_type}")
            print(f"  Confidence: {rhyme.confidence:.2f}")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Analysis failed: {e}")

        return matches / max_len

    def _create_empty_rhyme_analysis(self) -> RhymeAnalysis:
        """Create empty rhyme analysis for error cases"""
        return RhymeAnalysis(
            qafiyeh="",
            radif="",
            phonetic_rhyme="",
            rhyme_type="none",
            confidence=0.0
        )

        # Simple character-by-character comparison
        matches = sum(1 for i in range(min(len1, len2)) if pattern1[i] == pattern2[i])

        # Penalize length differences
        length_penalty = abs(len1 - len2) / max_len

        # Calculate similarity
        similarity = (matches / max_len) * (1 - length_penalty * 0.5)

        return max(0.0, similarity)

    def _create_empty_analysis(self) -> AruzAnalysis:
        """Create empty analysis for error cases"""
        return AruzAnalysis(
            identified_meter="unknown",
            pattern_match="",
            confidence=MeterConfidence.NONE,
            pattern_accuracy=0.0,
            variations_detected=[],
            line_scansion=[]
        )


class AdvancedRhymeAnalyzer:
    """
    Advanced rhyme analysis with phonetic awareness and qāfiyeh/radīf detection
    """

    def __init__(self):
        self.phonetics = PersianPhonetics()

        # Persian/Tajik stop words (common function words)
        self.stop_words = {
            "ва", "дар", "бо", "аз", "то", "барои", "чун", "ки", "агар",
            "ё", "на", "ҳам", "низ", "ба", "аммо", "лекин", "пас"
        }

        logger.info("AdvancedRhymeAnalyzer initialized")

    def analyze_rhyme(self, line: str) -> RhymeAnalysis:
        """
        Perform comprehensive rhyme analysis
        """
        try:
            # Extract words from line
            words = re.findall(r'[\wӣӯ]+', line)
            if not words:
                return self._create_empty_rhyme_analysis()

            # Find meaningful words (exclude stop words)
            meaningful_words = [w for w in words if w.lower() not in self.stop_words]
            if not meaningful_words:
                meaningful_words = words  # Use all words if no meaningful ones found

            # Get last meaningful word as potential rhyme carrier
            rhyme_word = meaningful_words[-1]

            # Check for radīf (repeated refrain after rhyme)
            radif = self._extract_radif(words, rhyme_word)

            # Extract qāfiyeh (actual rhyming sound)
            qafiyeh = self._extract_qafiyeh(rhyme_word)

            # Get phonetic representation
            phonetic_analysis = self.phonetics.to_phonetic(qafiyeh)
            phonetic_rhyme = phonetic_analysis.phonetic_transcription

            # Determine rhyme type
            rhyme_type = self._classify_rhyme_type(qafiyeh, phonetic_rhyme)

            return RhymeAnalysis(
                qafiyeh=qafiyeh,
                radif=radif,
                phonetic_rhyme=phonetic_rhyme,
                rhyme_type=rhyme_type,
                confidence=phonetic_analysis.confidence
            )

        except Exception as e:
            logger.error(f"Rhyme analysis failed for line '{line[:50]}...': {e}")
            return self._create_empty_rhyme_analysis()

    def _extract_radif(self, words: List[str], rhyme_word: str) -> str:
        """
        Extract radīf (repeated refrain that comes after the rhyme)
        """
        if not words or not rhyme_word:
            return ""

        try:
            rhyme_index = words.index(rhyme_word)
            if rhyme_index < len(words) - 1:
                # Everything after the rhyme word is potential radīf
                return " ".join(words[rhyme_index + 1:])
        except ValueError:
            pass

        return ""

    def _extract_qafiyeh(self, word: str) -> str:
        """
        Extract qāfiyeh (the actual rhyming sound) from a word

        In Persian prosody, qāfiyeh typically includes:
        - The last consonant + preceding vowel
        - May extend further back depending on the meter
        """
        if not word:
            return ""

        # Simple extraction: last 2-3 characters
        # Real implementation would need morphological analysis
        if len(word) >= 3:
            return word[-3:]
        elif len(word) >= 2:
            return word[-2:]
        else:
            return word

    def _classify_rhyme_type(self, qafiyeh: str, phonetic: str) -> str:
        """
        Classify the type of rhyme
        """
        if not qafiyeh:
            return "none"

        # Basic classification
        if len(phonetic) >= 3:
            return "rich"  # Multiple sounds rhyming
        elif len(phonetic) >= 2:
            return "perfect"  # Standard rhyme
        else:
            return "minimal"  # Single sound

    def calculate_rhyme_similarity(self, rhyme1: RhymeAnalysis, rhyme2: RhymeAnalysis) -> float:
        """
        Calculate phonetic similarity between two rhymes
        """
        if not rhyme1.phonetic_rhyme or not rhyme2.phonetic_rhyme:
            return 0.0

        # Simple phonetic similarity
        phone1 = rhyme1.phonetic_rhyme
        phone2 = rhyme2.phonetic_rhyme

        # Character-level similarity
        matches = sum(1 for a, b in zip(phone1, phone2) if a == b)
        max_len = max(len(phone1), len(phone2))

        if max_len == 0:
            return 1.0