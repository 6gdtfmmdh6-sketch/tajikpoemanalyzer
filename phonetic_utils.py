#!/usr/bin/env python3
"""
Phonetic Utilities and Supporting Data for Tajik Poetry Analysis

This module provides:
1. Enhanced phonetic transcription
2. Syllable boundary detection algorithms
3. Persian phonotactic rules
4. Validation utilities for phonetic analysis

SCIENTIFIC LIMITATIONS ACKNOWLEDGED:
- Simplified phonological model
- No morphophonological alternations
- Limited dialect coverage
- Requires linguistic validation for production use
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PersianPhonology:
    """
    Enhanced Persian/Tajik phonological system

    This implements a more sophisticated phonological model than the basic
    character mapping in the main analyzer.
    """

    def __init__(self):
        # Complete Persian/Tajik phoneme inventory
        self.consonants = {
            # Stops
            'پ': 'p', 'ب': 'b', 'ت': 't', 'د': 'd', 'ک': 'k', 'گ': 'g',
            'ق': 'q', 'ʔ': 'ʔ',

            # Fricatives
            'ف': 'f', 'و': 'v', 'ث': 's', 'ذ': 'z', 'س': 's', 'ز': 'z',
            'ش': 'ʃ', 'ژ': 'ʒ', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z',
            'ح': 'ħ', 'ع': 'ʕ', 'خ': 'x', 'غ': 'ɣ', 'ه': 'h',

            # Affricates
            'ج': 'ʤ', 'چ': 'ʧ',

            # Nasals
            'م': 'm', 'ن': 'n',

            # Liquids
            'ل': 'l', 'ر': 'r',

            # Glides
            'ی': 'j', 'و': 'w',

            # Tajik Cyrillic specific
            'қ': 'q', 'ғ': 'ɣ', 'ҳ': 'h', 'ҷ': 'ʤ'
        }

        self.vowels = {
            # Short vowels
            'ا': 'a', 'ِ': 'e', 'ُ': 'o', 'َ': 'a',

            # Long vowels (Persian)
            'آ': 'aː', 'ی': 'iː', 'و': 'uː',

            # Tajik Cyrillic vowels
            'а': 'a', 'о': 'o', 'у': 'u', 'е': 'e', 'и': 'i',
            'ӣ': 'iː', 'ӯ': 'uː', 'э': 'e',

            # Compound vowels
            'я': 'ja', 'ю': 'ju', 'ё': 'jo'
        }

        self.diphthongs = {
            'ای': 'aj', 'او': 'aw', 'وی': 'oj',
            'ай': 'aj', 'ау': 'aw', 'ой': 'oj', 'уй': 'uj'
        }

        # Phonotactic constraints for syllable structure
        self.syllable_patterns = [
            'CV',  # Open syllable - always light
            'CVC',  # Closed syllable - context dependent
            'CVV',  # Long vowel - always heavy
            'CVVC',  # Long vowel + consonant - always heavy
        ]

        # Consonant clusters allowed in Persian
        self.allowed_clusters = {
            # Initial clusters
            'initial': ['br', 'dr', 'fr', 'gr', 'kr', 'pr', 'tr', 'ʃr', 'xr'],

            # Final clusters
            'final': ['st', 'ʃt', 'xt', 'ft', 'rd', 'rm', 'rn', 'rl']
        }

        logger.info("Enhanced Persian phonology system initialized")

    def enhanced_phonetic_transcription(self, text: str) -> Tuple[str, float]:
        """
        Enhanced phonetic transcription with confidence scoring

        Returns:
            Tuple of (phonetic_string, confidence_score)
        """
        if not text:
            return "", 0.0

        # Normalize input
        text = text.strip()
        confidence = 1.0
        result = []

        i = 0
        while i < len(text):
            # Check for diphthongs first (longer sequences)
            if i < len(text) - 1:
                digraph = text[i:i + 2]
                if digraph in self.diphthongs:
                    result.append(self.diphthongs[digraph])
                    i += 2
                    continue

            # Single character mapping
            char = text[i]
            if char in self.consonants:
                result.append(self.consonants[char])
            elif char in self.vowels:
                result.append(self.vowels[char])
            elif char.isspace():
                result.append(' ')
            elif char in '.,;:!?':
                # Punctuation - skip
                pass
            else:
                # Unknown character - reduce confidence
                result.append(char)
                confidence *= 0.9
                logger.debug(f"Unknown character in phonetic transcription: {char}")

            i += 1

        return ''.join(result), confidence

    def find_syllable_boundaries_advanced(self, phonetic: str) -> List[int]:
        """
        Advanced syllable boundary detection using phonotactic constraints
        """
        if not phonetic:
            return [0]

        boundaries = [0]
        vowel_positions = self._find_vowel_nuclei(phonetic)

        if len(vowel_positions) <= 1:
            boundaries.append(len(phonetic))
            return boundaries

        # Place boundaries based on phonotactic principles
        for i in range(len(vowel_positions) - 1):
            current_vowel = vowel_positions[i]
            next_vowel = vowel_positions[i + 1]

            # Find consonants between vowels
            intervening = phonetic[current_vowel + 1:next_vowel]
            consonants = re.findall(r'[bcdfghjklmnpqrstvwxyzħʕʤʧʃʒɣ]+', intervening)

            if not consonants:
                # V.V - boundary after first vowel
                boundary = current_vowel + 1
            elif len(consonants[0]) == 1:
                # V.CV - boundary before consonant
                boundary = phonetic.find(consonants[0], current_vowel)
            else:
                # VCC.V or VC.CV - split consonant cluster
                boundary = self._split_consonant_cluster(phonetic, consonants[0], current_vowel, next_vowel)

            if boundary > boundaries[-1] and boundary < len(phonetic):
                boundaries.append(boundary)

        boundaries.append(len(phonetic))
        return sorted(set(boundaries))

    def _find_vowel_nuclei(self, phonetic: str) -> List[int]:
        """Find positions of vowel nuclei (including long vowels and diphthongs)"""
        positions = []
        vowel_chars = set('aeiouaːeːiːoːuː')

        i = 0
        while i < len(phonetic):
            if phonetic[i] in vowel_chars:
                positions.append(i)
                # Skip rest of long vowel or diphthong
                while i + 1 < len(phonetic) and phonetic[i + 1] in 'ːjw':
                    i += 1
            i += 1

        return positions

    def _split_consonant_cluster(self, phonetic: str, cluster: str, vowel1_pos: int, vowel2_pos: int) -> int:
        """
        Split consonant cluster according to Persian phonotactic rules
        """
        cluster_start = phonetic.find(cluster, vowel1_pos)

        if len(cluster) == 2:
            c1, c2 = cluster[0], cluster[1]

            # Check if C1C2 can be syllable-initial
            if cluster in self.allowed_clusters['initial']:
                # Don't split - assign to second syllable
                return cluster_start
            else:
                # Split after first consonant
                return cluster_start + 1

        else:
            # Longer cluster - split in middle
            return cluster_start + len(cluster) // 2

    def calculate_syllable_weight_advanced(self, syllable_phonetic: str, position: str = 'medial') -> str:
        """
        Advanced syllable weight calculation

        Args:
            syllable_phonetic: Phonetic representation of syllable
            position: 'initial', 'medial', or 'final'

        Returns:
            '—' for heavy, 'U' for light, '?' for uncertain
        """
        if not syllable_phonetic:
            return '?'

        # Check for long vowels or diphthongs -> always heavy
        if 'ː' in syllable_phonetic or any(d in syllable_phonetic for d in ['aj', 'aw', 'oj', 'uj']):
            return '—'

        # Count vowels and consonants
        vowels = re.findall(r'[aeiou]', syllable_phonetic)
        consonants = re.findall(r'[bcdfghjklmnpqrstvwxyzħʕʤʧʃʒɣ]', syllable_phonetic)

        if not vowels:
            return '?'  # No vowel nucleus

        # CV -> light
        if len(consonants) <= 1 and len(vowels) == 1:
            return 'U'

        # CVC in final position -> heavy
        if position == 'final' and len(consonants) >= 2:
            return '—'

        # CVC before consonant cluster -> heavy
        if len(consonants) >= 2:
            return '—'

        # Default to light
        return 'U'


class PersianMorphophonology:
    """
    Simplified Persian morphophonological rules

    LIMITATION: This is a basic implementation. Full morphophonology
    would require comprehensive morphological analysis.
    """

    def __init__(self):
        # Common morphophonological alternations
        self.alternations = {
            # Vowel harmony patterns (simplified)
            'harmony': {
                'e_a': ['e', 'a'],  # front/back alternation
                'i_u': ['i', 'u']  # high vowel alternation
            },

            # Consonant assimilation
            'assimilation': {
                'voicing': {'p': 'b', 't': 'd', 'k': 'g'},
                'place': {'n': 'm'}  # before labials
            }
        }

    def apply_morphophonological_rules(self, phonetic: str, morphological_context: str = None) -> str:
        """
        Apply basic morphophonological rules

        LIMITATION: This is highly simplified. Real implementation
        would need full morphological parser.
        """
        # This is a placeholder for morphophonological processing
        # In a full implementation, this would:
        # 1. Parse morphological structure
        # 2. Apply context-sensitive phonological rules
        # 3. Handle stress assignment
        # 4. Apply vowel harmony

        return phonetic  # Return unchanged for now


def create_sample_lexicon():
    """
    Create a sample Tajik lexicon for testing

    In production, this would be replaced with a comprehensive
    lexicon from linguistic resources.
    """
    sample_lexicon = {
        # Common words with phonetic transcriptions
        "муҳаббат": {"phonetic": "muħabbat", "syllables": ["mu", "ħab", "bat"], "weights": ["U", "—", "—"]},
        "ишқ": {"phonetic": "iʃq", "syllables": ["iʃq"], "weights": ["—"]},
        "дил": {"phonetic": "dil", "syllables": ["dil"], "weights": ["—"]},
        "гул": {"phonetic": "gul", "syllables": ["gul"], "weights": ["—"]},
        "баҳор": {"phonetic": "baħor", "syllables": ["ba", "ħor"], "weights": ["U", "—"]},
        "дарё": {"phonetic": "darjo", "syllables": ["dar", "jo"], "weights": ["—", "U"]},
        "кӯҳ": {"phonetic": "kuːh", "syllables": ["kuːh"], "weights": ["—"]},
        "ватан": {"phonetic": "vatan", "syllables": ["va", "tan"], "weights": ["U", "—"]},
        "шаб": {"phonetic": "ʃab", "syllables": ["ʃab"], "weights": ["—"]},
        "рӯз": {"phonetic": "ruːz", "syllables": ["ruːz"], "weights": ["—"]},
        "хуш": {"phonetic": "xuʃ", "syllables": ["xuʃ"], "weights": ["—"]},
        "меорад": {"phonetic": "meorad", "syllables": ["me", "o", "rad"], "weights": ["U", "U", "—"]},
        "мешавад": {"phonetic": "meʃavad", "syllables": ["me", "ʃa", "vad"], "weights": ["U", "U", "—"]},
        "мерасад": {"phonetic": "merasad", "syllables": ["me", "ra", "sad"], "weights": ["U", "U", "—"]},

        # Function words
        "ва": {"phonetic": "va", "syllables": ["va"], "weights": ["U"]},
        "дар": {"phonetic": "dar", "syllables": ["dar"], "weights": ["—"]},
        "бо": {"phonetic": "bo", "syllables": ["bo"], "weights": ["U"]},
        "аз": {"phonetic": "az", "syllables": ["az"], "weights": ["—"]},
        "то": {"phonetic": "to", "syllables": ["to"], "weights": ["U"]},
        "барои": {"phonetic": "baroi", "syllables": ["ba", "roi"], "weights": ["U", "—"]},
        "чун": {"phonetic": "ʧun", "syllables": ["ʧun"], "weights": ["—"]},
        "ки": {"phonetic": "ki", "syllables": ["ki"], "weights": ["U"]},
        "агар": {"phonetic": "agar", "syllables": ["a", "gar"], "weights": ["U", "—"]},

        # Poetic/classical vocabulary
        "қосид": {"phonetic": "qosid", "syllables": ["qo", "sid"], "weights": ["U", "—"]},
        "фардо": {"phonetic": "fardo", "syllables": ["far", "do"], "weights": ["—", "U"]},
        "видоъ": {"phonetic": "vidoʔ", "syllables": ["vi", "doʔ"], "weights": ["U", "—"]},
        "дуруд": {"phonetic": "durud", "syllables": ["du", "rud"], "weights": ["U", "—"]},
        "солик": {"phonetic": "solik", "syllables": ["so", "lik"], "weights": ["U", "—"]},
        "танҳо": {"phonetic": "tanho", "syllables": ["tan", "ho"], "weights": ["—", "U"]},
        "хонақоҳ": {"phonetic": "xonaqoh", "syllables": ["xo", "na", "qoh"], "weights": ["U", "U", "—"]},
        "сина": {"phonetic": "sina", "syllables": ["si", "na"], "weights": ["U", "U"]},
        "ҷаннат": {"phonetic": "ʤannat", "syllables": ["ʤan", "nat"], "weights": ["—", "—"]},
        "ибодат": {"phonetic": "ibodat", "syllables": ["i", "bo", "dat"], "weights": ["U", "U", "—"]}
    }

    return sample_lexicon


def validate_phonetic_transcription(original: str, phonetic: str, expected_syllables: int = None) -> Dict[str, Any]:
    """
    Validate phonetic transcription quality

    This function provides quality control for phonetic analysis
    """
    validation_result = {
        "is_valid": True,
        "confidence": 1.0,
        "warnings": [],
        "errors": []
    }

    # Check for obvious errors
    if not phonetic:
        validation_result["errors"].append("Empty phonetic transcription")
        validation_result["is_valid"] = False
        return validation_result

    # Check length ratio (phonetic shouldn't be much longer than original)
    length_ratio = len(phonetic) / max(1, len(original))
    if length_ratio > 2.0:
        validation_result["warnings"].append(f"Phonetic transcription unusually long (ratio: {length_ratio:.2f})")
        validation_result["confidence"] *= 0.8

    # Check for untranscribed characters (non-IPA)
    untranscribed = set()
    for char in phonetic:
        if char.isalpha() and char not in 'aeiouaːeːiːoːuːbcdfghjklmnpqrstvwxyzħʕʤʧʃʒɣʔjw':
            untranscribed.add(char)

    if untranscribed:
        validation_result["warnings"].append(f"Untranscribed characters found: {untranscribed}")
        validation_result["confidence"] *= 0.7

    # Check syllable count if provided
    if expected_syllables:
        phonology = PersianPhonology()
        boundaries = phonology.find_syllable_boundaries_advanced(phonetic)
        actual_syllables = len(boundaries) - 1

        if abs(actual_syllables - expected_syllables) > 1:
            validation_result["warnings"].append(
                f"Syllable count mismatch: expected {expected_syllables}, found {actual_syllables}"
            )
            validation_result["confidence"] *= 0.6

    return validation_result


class ProsodyValidator:
    """
    Validation utilities for prosodic analysis
    """

    @staticmethod
    def validate_meter_analysis(meter_name: str, pattern: str, confidence: float) -> Dict[str, Any]:
        """
        Validate meter analysis results
        """
        validation = {
            "is_valid": True,
            "confidence_assessment": "unknown",
            "warnings": [],
            "recommendations": []
        }

        # Assess confidence level
        if confidence >= 0.9:
            validation["confidence_assessment"] = "high"
        elif confidence >= 0.7:
            validation["confidence_assessment"] = "medium"
            validation["recommendations"].append("Consider manual verification")
        elif confidence >= 0.5:
            validation["confidence_assessment"] = "low"
            validation["warnings"].append("Low confidence - results may be unreliable")
            validation["recommendations"].append("Manual verification strongly recommended")
        else:
            validation["confidence_assessment"] = "very_low"
            validation["warnings"].append("Very low confidence - analysis likely failed")
            validation["is_valid"] = False
            validation["recommendations"].append("Consider different analytical approach")

        # Check pattern validity
        if pattern and not re.match(r'^[U—?]+$', pattern):
            validation["warnings"].append("Invalid prosodic pattern characters")
            validation["is_valid"] = False

        # Check for known problematic patterns
        if pattern and pattern.count('?') > len(pattern) * 0.3:
            validation["warnings"].append("Too many uncertain syllable weights")
            validation["confidence_assessment"] = "unreliable"

        return validation

    @staticmethod
    def validate_rhyme_analysis(rhymes: List[str], confidences: List[float]) -> Dict[str, Any]:
        """
        Validate rhyme scheme analysis
        """
        validation = {
            "is_valid": True,
            "rhyme_quality": "unknown",
            "warnings": [],
            "statistics": {}
        }

        if not rhymes:
            validation["warnings"].append("No rhymes detected")
            validation["is_valid"] = False
            return validation

        # Calculate statistics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        validation["statistics"] = {
            "total_lines": len(rhymes),
            "unique_rhymes": len(set(rhymes)),
            "average_confidence": avg_confidence,
            "rhyme_density": len(set(rhymes)) / len(rhymes) if rhymes else 0
        }

        # Assess rhyme quality
        if avg_confidence >= 0.8:
            validation["rhyme_quality"] = "high"
        elif avg_confidence >= 0.6:
            validation["rhyme_quality"] = "medium"
        else:
            validation["rhyme_quality"] = "low"
            validation["warnings"].append("Low rhyme detection confidence")

        # Check for reasonable rhyme schemes
        rhyme_density = validation["statistics"]["rhyme_density"]
        if rhyme_density > 0.8:
            validation["warnings"].append("Very few repeated rhymes - may not be rhyming verse")

        return validation


def export_validation_report(analysis_results: Dict[str, Any], output_file: str):
    """
    Export comprehensive validation report for scientific documentation
    """
    report = {
        "analysis_metadata": {
            "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "timestamp_unavailable",
            "analyzer_version": "Phase1_Enhanced",
            "validation_level": "scientific"
        },
        "input_validation": {},
        "analysis_validation": {},
        "quality_metrics": {},
        "recommendations": []
    }

    # This would be expanded to include comprehensive validation
    # For now, basic structure is provided

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Validation report exported to {output_file}")
    except Exception as e:
        logger.error(f"Failed to export validation report: {e}")


# Test and validation functions
def run_phonetic_tests():
    """
    Run tests on phonetic analysis components
    """
    print("=== PHONETIC ANALYSIS TESTS ===\n")

    phonology = PersianPhonology()

    test_cases = [
        ("муҳаббат", "muħabbat", 3),
        ("дарё", "darjo", 2),
        ("кӯҳ", "kuːh", 1),
        ("баҳор", "baħor", 2)
    ]

    for original, expected_phonetic, expected_syllables in test_cases:
        print(f"Testing: {original}")

        # Test phonetic transcription
        phonetic, confidence = phonology.enhanced_phonetic_transcription(original)
        print(f"  Phonetic: {phonetic} (confidence: {confidence:.2f})")
        print(f"  Expected: {expected_phonetic}")

        # Test syllable boundaries
        boundaries = phonology.find_syllable_boundaries_advanced(phonetic)
        actual_syllables = len(boundaries) - 1
        print(f"  Syllables: {actual_syllables} (expected: {expected_syllables})")

        # Validate
        validation = validate_phonetic_transcription(original, phonetic, expected_syllables)
        print(f"  Validation: {validation['confidence']:.2f} confidence")
        if validation['warnings']:
            print(f"  Warnings: {validation['warnings']}")

        print()


def run_prosody_tests():
    """
    Run tests on prosodic analysis components
    """
    print("=== PROSODIC ANALYSIS TESTS ===\n")

    from enhanced_tajik_analyzer import AruzMeterAnalyzer

    analyzer = AruzMeterAnalyzer()

    test_lines = [
        "Дар кӯҳсори ватан гулҳо мешуканд",
        "Дили шоир аз муҳаббат меларазад",
        "Баҳори нав ба замин таҷдид меорад"
    ]

    for line in test_lines:
        print(f"Testing: {line}")

        analysis = analyzer.analyze_meter(line)
        print(f"  Meter: {analysis.identified_meter}")
        print(f"  Pattern: {analysis.pattern_match}")
        print(f"  Confidence: {analysis.confidence.value}")
        print(f"  Accuracy: {analysis.pattern_accuracy:.2f}")

        # Validate
        validation = ProsodyValidator.validate_meter_analysis(
            analysis.identified_meter,
            analysis.pattern_match,
            analysis.pattern_accuracy
        )
        print(f"  Validation: {validation['confidence_assessment']}")
        if validation['warnings']:
            print(f"  Warnings: {validation['warnings']}")

        print()


if __name__ == "__main__":
    # Run tests when script is executed directly
    print("TESTING ENHANCED PHONETIC AND PROSODIC ANALYSIS\n")
    print("=" * 60)

    try:
        run_phonetic_tests()
        run_prosody_tests()

        # Create sample lexicon file
        lexicon = create_sample_lexicon()
        with open('tajik_lexicon.json', 'w', encoding='utf-8') as f:
            json.dump(list(lexicon.keys()), f, ensure_ascii=False, indent=2)
        print("Sample lexicon created: tajik_lexicon.json")

    except Exception as e:
        print(f"Testing failed: {e}")
        logger.error(f"Testing failed: {e}")
