#!/usr/bin/env python3
"""
Tajik Poetry Analyzer - Consolidated Version
Scientific Research Grade with Proper ʿArūḍ Analysis

This implementation provides:
1. Proper ʿArūḍ (Classical Arabic-Persian prosody) analysis with 16 meters
2. Phonetic-based rhyme detection (Qāfiyeh/Radīf)
3. Accurate syllable weight calculation
4. Scientific error handling and validation
5. Excel report generation

Consolidated from app2.py and _tajik_analyzer.py
Removed: ContentAnalyzer, LotmanSemioticAnalyzer, TranslationTheoreticalAnalyzer, SemanticFieldAnalyzer
"""

import re
import json
import logging
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
import statistics
import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.chart import BarChart, Reference
from analyzer import TajikPoemAnalyzer, StructuralAnalysis, ContentAnalysis, LiteraryAssessment
from modern_verse_analyzer import ModernVerseAnalyzer, ModernVerseMetrics, FreeVerseClassifier
from corpus_manager import TajikCorpusManager
from dataclasses import dataclass, field
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class SyllableWeight(Enum):
    """Prosodic weight classification"""
    HEAVY = "—"  # Long syllable
    LIGHT = "U"  # Short syllable (using U instead of ∪ for compatibility)
    ANCEPS = "×"  # Variable weight
    UNKNOWN = "?"  # Uncertain weight


class MeterConfidence(Enum):
    """Confidence levels for meter identification"""
    HIGH = "high"  # >90% pattern match
    MEDIUM = "medium"  # 70-90% pattern match
    LOW = "low"  # 50-70% pattern match
    NONE = "none"  # <50% pattern match


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for poetry analysis"""
    lexicon_path: str = 'data/tajik_lexicon.json'
    min_poem_length: int = 10
    max_neologisms: int = 10
    min_title_length: int = 3
    max_title_length: int = 50

    # Extended theme taxonomy
    themes: Dict[str, List[str]] = field(default_factory=lambda: {
        "Love": ["муҳаббат", "ишқ", "дил", "маҳбуб", "ёр", "дилбар", "ошиқ", "маъшуқ"],
        "Nature": ["дарё", "кӯҳ", "гул", "баҳор", "навбаҳор", "осмон", "офтоб", "моҳ", "ситора"],
        "Homeland": ["ватан", "тоҷикистон", "чашма", "диёр", "марзу бум", "кишвар"],
        "Religion": ["худо", "ҷаннат", "ибодат", "намоз", "масҷид", "аллоҳ", "паёмбар"],
        "Mysticism": ["тариқат", "мақом", "ҳақиқат", "маърифат", "ваҳдат", "фано"],
        "Philosophy": ["ҳикмат", "дониш", "хирад", "ақл", "маънӣ", "ҷаҳон", "ҳастӣ"]
    })


@dataclass
class PoemData:
    """Data structure for a single poem"""
    title: str
    content: str
    poem_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProsodicSyllable:
    """Represents a syllable with prosodic information"""
    text: str
    weight: SyllableWeight
    phonetic: Optional[str] = None
    position: int = 0
    confidence: float = 1.0
    stress_level: int = 0


@dataclass
class AruzPattern:
    """Represents a classical ʿArūḍ meter pattern"""
    name: str
    pattern: str
    description: str
    variations: List[str] = field(default_factory=list)
    frequency_weight: float = 1.0


@dataclass
class PhoneticAnalysis:
    """Results of phonetic analysis"""
    phonetic_transcription: str
    syllable_boundaries: List[int]
    stress_pattern: List[int]
    confidence: float
    phoneme_inventory: Dict[str, int] = field(default_factory=dict)


@dataclass
class RhymeAnalysis:
    """Advanced rhyme analysis results"""
    qafiyeh: str  # The actual rhyming sound
    radif: str  # Repeated refrain after rhyme
    phonetic_rhyme: str  # Phonetic representation
    rhyme_type: str  # perfect, imperfect, eye-rhyme, etc.
    rhyme_position: str = "end"
    confidence: float = 0.0


@dataclass
class AruzAnalysis:
    """Results of ʿArūḍ meter analysis"""
    identified_meter: str
    pattern_match: str
    confidence: MeterConfidence
    pattern_accuracy: float
    variations_detected: List[str]
    line_scansion: List[ProsodicSyllable]
    caesura_positions: List[int] = field(default_factory=list)


@dataclass
class StructuralAnalysis:
    """ structural analysis results"""
    lines: int
    syllables_per_line: List[int]
    syllable_patterns: List[List[ProsodicSyllable]]
    aruz_analysis: AruzAnalysis
    rhyme_scheme: List[RhymeAnalysis]
    rhyme_pattern: str
    stanza_structure: str
    avg_syllables: float
    prosodic_consistency: float
    meter_confidence: MeterConfidence


@dataclass
class ContentAnalysis:
    """Content analysis results including lexical features"""
    word_frequencies: List[Tuple[str, int]]
    neologisms: List[str]
    archaisms: List[str]
    theme_distribution: Dict[str, int]
    lexical_diversity: float
    stylistic_register: str
    total_words: int
    unique_words: int

@dataclass
class ModernVerseMetrics:
    """Metriken für moderne/freie Verse"""
    enjambement_count: int = 0
    enjambement_ratio: float = 0.0
    semantic_density: float = 0.0  # Wörter pro Zeile
    line_length_variation: float = 0.0  # CV der Silben pro Zeile
    prose_poetry_score: float = 0.0
    visual_structure_score: float = 0.0
    caesura_distribution: List[int] = field(default_factory=list)
    syntactic_parallelism: float = 0.0
    lexical_repetition_score: float = 0.0
    
    # Experimentelle Metriken
    breath_group_length: float = 0.0  # Durchschn. Satzlänge in Wörtern
    pause_frequency: float = 0.0  # Interpunktion pro Zeile

class ModernVerseAnalyzer:
    """Spezialisierter Analyzer für freie Verse"""
    
    def __init__(self):
        self.punctuation = set('.,!?;:—–-()[]{}"\'«»')
        self.sentence_enders = set('.!?;')
        
    def analyze(self, poem_content: str, syllable_counts: List[int] = None) -> ModernVerseMetrics:
        """Führt umfassende Analyse freier Verse durch"""
        lines = [line.rstrip() for line in poem_content.split('\n') if line.strip()]
        
        if not lines:
            return ModernVerseMetrics()
            
        # Enjambement-Analyse
        enjambement_count = self._count_enjambements(lines)
        
        # Semantische Dichte (Wörter pro Zeile)
        semantic_density = self._calculate_semantic_density(lines)
        
        # Zeilenlängen-Variation
        if syllable_counts:
            line_length_variation = self._calculate_line_variation(syllable_counts)
        else:
            # Fallback: Zeichen pro Zeile
            char_counts = [len(line) for line in lines]
            line_length_variation = statistics.stdev(char_counts) / statistics.mean(char_counts) if char_counts else 0
        
        # Prosa-Poesie Score
        prose_score = self._calculate_prose_poetry_score(lines)
        
        # Visuelle Struktur
        visual_score = self._analyze_visual_structure(poem_content)
        
        # Caesura-Verteilung
        caesura_dist = self._analyze_caesura_distribution(lines)
        
        # Syntaktischer Parallelismus
        parallelism = self._calculate_syntactic_parallelism(lines)
        
        # Lexikalische Wiederholung
        repetition_score = self._calculate_lexical_repetition(lines)
        
        # Atemgruppen-Länge
        breath_groups = self._analyze_breath_groups(poem_content)
        
        # Pausenfrequenz
        pause_freq = self._calculate_pause_frequency(lines)
        
        return ModernVerseMetrics(
            enjambement_count=enjambement_count,
            enjambement_ratio=enjambement_count / max(len(lines) - 1, 1),
            semantic_density=semantic_density,
            line_length_variation=line_length_variation,
            prose_poetry_score=prose_score,
            visual_structure_score=visual_score,
            caesura_distribution=caesura_dist,
            syntactic_parallelism=parallelism,
            lexical_repetition_score=repetition_score,
            breath_group_length=breath_groups,
            pause_frequency=pause_freq
        )
    
    def _count_enjambements(self, lines: List[str]) -> int:
        """Zählt Enjambements (Zeilensprünge)"""
        count = 0
        for i in range(len(lines) - 1):
            current_line = lines[i].strip()
            next_line = lines[i + 1].strip()
            
            if not current_line or not next_line:
                continue
                
            # Enjambement, wenn Zeile nicht mit Satzzeichen endet
            # UND nächste Zeile nicht mit Großbuchstaben beginnt (kein neuer Satz)
            if (current_line[-1] not in self.sentence_enders and
                not next_line[0].isupper()):
                count += 1
                
        return count
    
    def _calculate_semantic_density(self, lines: List[str]) -> float:
        """Berechnet semantische Dichte (Wörter pro Zeile)"""
        word_counts = [len(re.findall(r'[\wӣӯ]+', line)) for line in lines]
        return statistics.mean(word_counts) if word_counts else 0
    
    def _calculate_line_variation(self, syllable_counts: List[int]) -> float:
        """Berechnet Variationskoeffizient der Zeilenlängen"""
        if len(syllable_counts) < 2:
            return 0
            
        mean_val = statistics.mean(syllable_counts)
        if mean_val == 0:
            return 0
            
        stdev = statistics.stdev(syllable_counts)
        return stdev / mean_val
    
    def _calculate_prose_poetry_score(self, lines: List[str]) -> float:
        """
        Berechnet Prosa-Poesie Score (0 = rein poetisch, 1 = prosaisch)
        """
        if not lines:
            return 0
            
        scores = []
        
        for line in lines:
            # Kürzere Zeilen sind poetischer
            length_score = min(1.0, len(line) / 100)
            
            # Satzzeichen am Ende = prosaisch
            punctuation_score = 1.0 if line[-1] in self.sentence_enders else 0
            
            # Enthält Alltagssprache?
            prose_words = {'ва', 'ки', 'дар', 'бо', 'аз', 'то', 'барои', 'аммо', 'лекин'}
            words = set(re.findall(r'[\wӣӯ]+', line.lower()))
            prose_word_score = len(words & prose_words) / max(len(words), 1)
            
            line_score = (length_score * 0.3 + 
                         punctuation_score * 0.4 + 
                         prose_word_score * 0.3)
            scores.append(line_score)
            
        return statistics.mean(scores) if scores else 0
    
    def _analyze_visual_structure(self, poem_content: str) -> float:
        """Analysiert visuelle Struktur (Einzüge, Leerzeilen)"""
        lines = poem_content.split('\n')
        
        if len(lines) < 2:
            return 0
            
        # Zähle Einzüge
        indent_count = 0
        for line in lines:
            if line.startswith((' ', '\t')) and line.strip():
                indent_count += 1
        
        # Zähle Leerzeilen-Blöcke
        empty_line_blocks = 0
        in_empty_block = False
        
        for line in lines:
            if not line.strip():
                if not in_empty_block:
                    empty_line_blocks += 1
                    in_empty_block = True
            else:
                in_empty_block = False
        
        visual_score = (indent_count / len(lines) * 0.5 + 
                       empty_line_blocks / len(lines) * 0.5)
        
        return min(1.0, visual_score)
    
    def _analyze_caesura_distribution(self, lines: List[str]) -> List[int]:
        """Analysiert Caesura-Verteilung"""
        # Vereinfachte Caesura-Erkennung an Kommas, Gedankenstrichen
        caesura_positions = []
        
        for i, line in enumerate(lines):
            if ',' in line or '—' in line or '–' in line or ';' in line:
                caesura_positions.append(i)
                
        return caesura_positions
    
    def _calculate_syntactic_parallelism(self, lines: List[str]) -> float:
        """Berechnet syntaktischen Parallelismus"""
        if len(lines) < 2:
            return 0
            
        parallel_pairs = 0
        total_pairs = 0
        
        for i in range(len(lines) - 1):
            line1 = lines[i].lower()
            line2 = lines[i + 1].lower()
            
            # Entferne Interpunktion
            line1 = re.sub(r'[^\wӣӯ\s]', '', line1)
            line2 = re.sub(r'[^\wӣӯ\s]', '', line2)
            
            # Wörter zählen
            words1 = line1.split()
            words2 = line2.split()
            
            if len(words1) > 1 and len(words2) > 1:
                # Prüfe auf ähnliche Wortreihenfolge (Anfänge)
                if words1[0] == words2[0]:
                    parallel_pairs += 1
                total_pairs += 1
                
        return parallel_pairs / total_pairs if total_pairs > 0 else 0
    
    def _calculate_lexical_repetition(self, lines: List[str]) -> float:
        """Berechnet lexikalische Wiederholung"""
        all_words = []
        for line in lines:
            words = re.findall(r'[\wӣӯ]+', line.lower())
            all_words.extend(words)
            
        if not all_words:
            return 0
            
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Anteil der Wörter, die mehr als einmal vorkommen
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        total_unique_words = len(word_counts)
        
        return repeated_words / total_unique_words if total_unique_words > 0 else 0
    
    def _analyze_breath_groups(self, poem_content: str) -> float:
        """Analysiert Atemgruppen (Satzlängen)"""
        # Satzenden erkennen
        sentences = re.split(r'[.!?;]+', poem_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0
            
        # Wörter pro Satz
        words_per_sentence = []
        for sentence in sentences:
            words = re.findall(r'[\wӣӯ]+', sentence)
            if words:
                words_per_sentence.append(len(words))
                
        return statistics.mean(words_per_sentence) if words_per_sentence else 0
    
    def _calculate_pause_frequency(self, lines: List[str]) -> float:
        """Berechnet Pausenfrequenz (Interpunktion pro Zeile)"""
        if not lines:
            return 0
            
        punctuation_count = 0
        for line in lines:
            punctuation_count += sum(1 for char in line if char in self.punctuation)
            
        return punctuation_count / len(lines)

class FreeVerseClassifier:
    """Klassifiziert, ob ein Gedicht freie Verse enthält"""
    
    @staticmethod
    def is_free_verse(structural_analysis, modern_metrics: ModernVerseMetrics) -> bool:
        """
        Entscheidet, ob ein Gedicht als freier Vers klassifiziert werden soll
        """
        criteria = {
            'meter_confidence_low': structural_analysis.meter_confidence.value in ['low', 'none'],
            'prosodic_inconsistency': structural_analysis.prosodic_consistency < 0.5,
            'enjambement_high': modern_metrics.enjambement_ratio > 0.3,
            'line_variation_high': modern_metrics.line_length_variation > 0.4,
            'prose_score_high': modern_metrics.prose_poetry_score > 0.6
        }
        
        # Gewichtete Entscheidung
        weights = {
            'meter_confidence_low': 2.0,
            'prosodic_inconsistency': 1.5,
            'enjambement_high': 1.0,
            'line_variation_high': 1.0,
            'prose_score_high': 0.5
        }
        
        score = sum(weights[k] for k, v in criteria.items() if v)
        
        return score >= 2.5  # Schwellenwert


@dataclass
class LiteraryAssessment:
    """Multi-perspective literary assessment"""
    german_perspective: int
    persian_tradition: int
    tajik_elements: int
    modernist_features: int


@dataclass
class ComprehensiveAnalysis:
    """Complete analysis results"""
    structural: StructuralAnalysis
    content: ContentAnalysis
    literary: LiteraryAssessment
    quality_metrics: Dict[str, float]

# =============================================================================
#  STRUCTURAL ANALYSIS (Neu!)
# =============================================================================

@dataclass
class StructuralAnalysis(StructuralAnalysis):
    """Direkt nach den anderen @dataclasses"""
    modern_metrics: Optional[ModernVerseMetrics] = None
    is_free_verse: bool = False
    free_verse_confidence: float = 0.0
    modern_features: Dict[str, float] = field(default_factory=dict)

@dataclass 
class ComprehensiveAnalysis:
    """Nach ComprehensiveAnalysis"""
    structural: StructuralAnalysis
    content: ContentAnalysis
    literary: LiteraryAssessment
    quality_metrics: Dict[str, float]
    corpus_ready: bool = False
    contribution_id: Optional[str] = None


# =============================================================================
# PHONETICS
# =============================================================================

class PersianTajikPhonetics:
    """Comprehensive Persian/Tajik phonetic analyzer"""

    def __init__(self):
        # IPA mapping for Tajik/Persian
        self.phoneme_map = {
            # Consonants (Cyrillic)
            'б': 'b', 'п': 'p', 'т': 't', 'ҷ': 'ʤ', 'ч': 'ʧ',
            'х': 'x', 'д': 'd', 'р': 'r', 'з': 'z', 'ж': 'ʒ',
            'с': 's', 'ш': 'ʃ', 'ғ': 'ʁ', 'ф': 'f', 'қ': 'q',
            'к': 'k', 'г': 'g', 'л': 'l', 'м': 'm', 'н': 'n',
            'в': 'v', 'ҳ': 'h', 'й': 'j',
            # Vowels (Cyrillic)
            'а': 'a', 'о': 'o', 'у': 'u', 'э': 'e', 'и': 'i',
            'ӣ': 'iː', 'ӯ': 'uː', 'я': 'ja', 'ю': 'ju', 'ё': 'jo',
            'е': 'e',
            # Arabic script consonants
            'ب': 'b', 'پ': 'p', 'ت': 't', 'ث': 's', 'ج': 'ʤ', 'چ': 'ʧ',
            'ح': 'ħ', 'خ': 'x', 'د': 'd', 'ذ': 'z', 'ر': 'r', 'ز': 'z',
            'ژ': 'ʒ', 'س': 's', 'ش': 'ʃ', 'ص': 's', 'ض': 'z', 'ط': 't',
            'ظ': 'z', 'ع': 'ʔ', 'غ': 'ɣ', 'ف': 'f', 'ق': 'q', 'ک': 'k',
            'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'و': 'w', 'ه': 'h',
            'ی': 'j',
        }

        self.vowels = set('аоуэиӣӯяюёе')
        self.long_vowels = set('ӣӯ')
        self.short_vowels = {'a', 'e', 'i', 'o', 'u'}
        self.long_vowels_ipa = {'aː', 'eː', 'iː', 'oː', 'uː'}
        self.consonants = set('бпттҷчхдрзжсшғфқкглмнвҳй')
        self.sonorous = set('рлмнвй')
        self.diphthongs = {'ай', 'ой', 'уй', 'ей', 'ӯй', 'ав', 'ов'}
        self.diphthongs_ipa = {'aj', 'aw', 'oj', 'ej'}

    def analyze_phonetics(self, text: str) -> PhoneticAnalysis:
        """Complete phonetic analysis"""
        text = unicodedata.normalize('NFC', text.lower())

        # Generate phonetic transcription
        phonetic = self._to_ipa(text)

        # Find syllable boundaries
        syllables = self._syllabify(text)
        boundaries = [s[0] for s in syllables] + [len(text)] if syllables else []

        # Determine stress pattern
        stress_pattern = self._determine_stress(syllables)

        # Count phonemes
        phoneme_inventory = Counter(phonetic)

        return PhoneticAnalysis(
            phonetic_transcription=phonetic,
            syllable_boundaries=boundaries,
            stress_pattern=stress_pattern,
            phoneme_inventory=dict(phoneme_inventory),
            confidence=0.85
        )

    def to_phonetic(self, text: str) -> PhoneticAnalysis:
        """Alias for analyze_phonetics for compatibility"""
        return self.analyze_phonetics(text)

    def _to_ipa(self, text: str) -> str:
        """Convert to IPA transcription"""
        result = []
        for char in text:
            if char in self.phoneme_map:
                result.append(self.phoneme_map[char])
            elif char.isspace():
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)

    def _syllabify(self, text: str) -> List[Tuple[int, str]]:
        """Syllabify text according to Persian/Tajik rules"""
        syllables = []
        i = 0

        while i < len(text):
            if text[i].isspace():
                i += 1
                continue

            syl_start = i

            # Skip initial consonants
            while i < len(text) and text[i] in self.consonants:
                i += 1

            # Must have a vowel nucleus
            if i < len(text) and text[i] in self.vowels:
                # Check for diphthong
                if i + 1 < len(text) and text[i:i + 2] in self.diphthongs:
                    i += 2
                else:
                    i += 1

                # Coda consonants
                while i < len(text) and text[i] in self.consonants:
                    if i + 1 < len(text) and text[i + 1] in self.vowels:
                        if text[i] in self.sonorous:
                            i += 1
                        break
                    i += 1

                syllables.append((syl_start, text[syl_start:i]))
            else:
                i += 1

        return syllables

    def _determine_stress(self, syllables: List[Tuple[int, str]]) -> List[int]:
        """Determine stress pattern (Persian typically has final stress)"""
        if not syllables:
            return []

        stress = [0] * len(syllables)
        stress[-1] = 2  # Primary stress on final syllable

        for i, (_, syl) in enumerate(syllables[:-1]):
            if any(v in syl for v in self.long_vowels):
                stress[i] = 1

        return stress

    def calculate_syllable_weight(self, syllable: str) -> SyllableWeight:
        """Calculate syllable weight for prosody"""
        syllable = syllable.strip()

        if not syllable:
            return SyllableWeight.UNKNOWN

        # Check for long vowels
        if any(v in syllable for v in self.long_vowels):
            return SyllableWeight.HEAVY

        # Check for diphthongs
        if any(d in syllable for d in self.diphthongs):
            return SyllableWeight.HEAVY

        # Check for closed syllables (CVC)
        vowel_count = sum(1 for c in syllable if c in self.vowels)
        consonant_after_vowel = False

        for i, char in enumerate(syllable):
            if char in self.vowels and i + 1 < len(syllable):
                if syllable[i + 1] in self.consonants:
                    consonant_after_vowel = True
                    break

        if vowel_count > 0 and consonant_after_vowel:
            return SyllableWeight.HEAVY

        return SyllableWeight.LIGHT


# =============================================================================
# ARUZ METER ANALYZER (16 Classical Meters)
# =============================================================================

class AruzMeterAnalyzer:
    """
    Classical ʿArūḍ (Arabic-Persian prosody) analyzer
    Implements the 16 classical Arabic meters adapted for Persian/Tajik poetry
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.phonetics = PersianTajikPhonetics()

        # Classical ʿArūḍ meters with their patterns
        # Pattern notation: — = heavy syllable, U = light syllable
        self.aruz_meters = {
            "ṭawīl": AruzPattern(
                name="ṭawīl",
                pattern="U—UU—U—UU—",
                description="فعولن مفاعيلن فعولن مفاعيلن",
                variations=["U—UU—U—UU", "U—UU—U—U—"],
                frequency_weight=1.5
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
                description="مفاعیلن مفاعیلن مفاعیلن مفاعیلن",
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
                pattern="—U——U——U—",
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
            "munsarih": AruzPattern(
                name="munsarih",
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
                description="مفاعیلن فاعلاتن مفاعیلن",
                variations=["—U—U—U", "U—U—U—"],
                frequency_weight=0.4
            ),
            "muqtaḍab": AruzPattern(
                name="muqtaḍab",
                pattern="U—U—U—",
                description="مفعولات مستفعلن",
                variations=["U—U—U", "UU—U—"],
                frequency_weight=0.3
            ),
            "mujtath": AruzPattern(
                name="mujtath",
                pattern="UU—U—UU—U—",
                description="مستفعلن فاعلاتن",
                variations=["UU—U", "UUU—"],
                frequency_weight=0.3
            ),
            "mutadārik": AruzPattern(
                name="mutadārik",
                pattern="—U—U—U—U",
                description="فاعلن فاعلن فاعلن فاعلن",
                variations=["U—U—U—", "UU—U—U"],
                frequency_weight=0.5
            ),
            "madīd": AruzPattern(
                name="madīd",
                pattern="—U——U—U—",
                description="فاعلاتن فاعلن فاعلاتن",
                variations=["—U——U—", "U——U—U—"],
                frequency_weight=0.4
            ),
        }

        logger.info(f"AruzMeterAnalyzer initialized with {len(self.aruz_meters)} classical meters")

    def analyze_meter(self, line: str) -> AruzAnalysis:
        """Analyze a line of poetry for ʿArūḍ meter"""
        try:
            phonetic_analysis = self.phonetics.analyze_phonetics(line)
            syllables = self._extract_prosodic_syllables(line, phonetic_analysis)

            if not syllables:
                logger.warning(f"No syllables found in line: {line[:50]}...")
                return self._create_empty_analysis()

            pattern = "".join([syl.weight.value for syl in syllables])
            best_match = self._find_best_meter_match(pattern)
            caesuras = self._find_caesuras(syllables)

            return AruzAnalysis(
                identified_meter=best_match["meter"],
                pattern_match=pattern,
                confidence=best_match["confidence"],
                pattern_accuracy=best_match["accuracy"],
                variations_detected=best_match["variations"],
                line_scansion=syllables,
                caesura_positions=caesuras
            )

        except Exception as e:
            logger.error(f"Meter analysis failed for line '{line[:50]}...': {e}")
            return self._create_empty_analysis()

    def _extract_prosodic_syllables(self, line: str, phonetic: PhoneticAnalysis) -> List[ProsodicSyllable]:
        """Extract syllables with prosodic information"""
        syllables = []
        words = line.split()
        position = 0

        for word in words:
            word_syllables = self.phonetics._syllabify(word)

            for i, (start, syl_text) in enumerate(word_syllables):
                weight = self.phonetics.calculate_syllable_weight(syl_text)
                phonetic_syl = self.phonetics._to_ipa(syl_text)
                stress = 2 if i == len(word_syllables) - 1 else 0

                syllables.append(ProsodicSyllable(
                    text=syl_text,
                    weight=weight,
                    phonetic=phonetic_syl,
                    position=position,
                    stress_level=stress,
                    confidence=phonetic.confidence
                ))
                position += 1

        return syllables

    def _find_best_meter_match(self, pattern: str) -> Dict[str, Any]:
        """Find the best matching ʿArūḍ meter"""
        best_match = {
            "meter": "unknown",
            "pattern": pattern,
            "confidence": MeterConfidence.NONE,
            "accuracy": 0.0,
            "variations": []
        }

        best_score = 0.0

        for meter_name, meter_info in self.aruz_meters.items():
            score = self._pattern_similarity(pattern, meter_info.pattern)

            variation_scores = []
            for variation in meter_info.variations:
                var_score = self._pattern_similarity(pattern, variation)
                variation_scores.append((variation, var_score))

            max_variation_score = max([score] + [s for _, s in variation_scores])
            weighted_score = max_variation_score * meter_info.frequency_weight

            if weighted_score > best_score:
                best_score = weighted_score
                best_match.update({
                    "meter": meter_name,
                    "pattern": meter_info.pattern,
                    "accuracy": max_variation_score,
                    "variations": [var for var, s in variation_scores if s > 0.7]
                })

        if best_score >= 0.9:
            best_match["confidence"] = MeterConfidence.HIGH
        elif best_score >= 0.7:
            best_match["confidence"] = MeterConfidence.MEDIUM
        elif best_score >= 0.5:
            best_match["confidence"] = MeterConfidence.LOW

        return best_match

    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """Calculate similarity between two prosodic patterns"""
        if not pattern1 or not pattern2:
            return 0.0

        len1, len2 = len(pattern1), len(pattern2)
        max_len = max(len1, len2)

        if max_len == 0:
            return 1.0

        matches = sum(1 for i in range(min(len1, len2)) if pattern1[i] == pattern2[i])
        length_penalty = abs(len1 - len2) / max_len
        similarity = (matches / max_len) * (1 - length_penalty * 0.5)

        return max(0.0, similarity)

    def _find_caesuras(self, syllables: List[ProsodicSyllable]) -> List[int]:
        """Find caesura positions"""
        caesuras = []
        for i in range(1, len(syllables) - 1):
            if (syllables[i - 1].stress_level > 0 and
                    syllables[i].stress_level == 0 and
                    i % 4 == 0):
                caesuras.append(i)
        return caesuras

    def _create_empty_analysis(self) -> AruzAnalysis:
        """Create empty analysis for error cases"""
        return AruzAnalysis(
            identified_meter="unknown",
            pattern_match="",
            confidence=MeterConfidence.NONE,
            pattern_accuracy=0.0,
            variations_detected=[],
            line_scansion=[],
            caesura_positions=[]
        )


# =============================================================================
# RHYME ANALYZER
# =============================================================================

class AdvancedRhymeAnalyzer:
    """Advanced rhyme analysis with phonetic awareness and qāfiyeh/radīf detection"""

    def __init__(self):
        self.phonetics = PersianTajikPhonetics()
        self.stop_words = {
            "ва", "дар", "бо", "аз", "то", "барои", "чун", "ки", "агар",
            "ё", "на", "ҳам", "низ", "ба", "аммо", "лекин", "пас"
        }

    def analyze_rhyme(self, line: str) -> RhymeAnalysis:
        """Perform comprehensive rhyme analysis"""
        try:
            words = re.findall(r'[\wӣӯ]+', line)
            if not words:
                return self._empty_rhyme()

            meaningful_words = [w for w in words if w.lower() not in self.stop_words]
            if not meaningful_words:
                meaningful_words = words

            rhyme_word = meaningful_words[-1]
            radif = self._extract_radif(words, rhyme_word)
            qafiyeh = self._extract_qafiyeh(rhyme_word)

            phonetic_analysis = self.phonetics.analyze_phonetics(qafiyeh)
            phonetic_rhyme = phonetic_analysis.phonetic_transcription
            rhyme_type = self._classify_rhyme_type(qafiyeh, phonetic_rhyme)

            return RhymeAnalysis(
                qafiyeh=qafiyeh,
                radif=radif,
                phonetic_rhyme=phonetic_rhyme,
                rhyme_type=rhyme_type,
                rhyme_position="end",
                confidence=phonetic_analysis.confidence
            )

        except Exception as e:
            logger.error(f"Rhyme analysis failed for line '{line[:50]}...': {e}")
            return self._empty_rhyme()

    def _extract_radif(self, words: List[str], rhyme_word: str) -> str:
        """Extract radīf (repeated refrain)"""
        if not words or not rhyme_word:
            return ""
        try:
            rhyme_index = words.index(rhyme_word)
            if rhyme_index < len(words) - 1:
                return " ".join(words[rhyme_index + 1:])
        except ValueError:
            pass
        return ""

    def _extract_qafiyeh(self, word: str) -> str:
        """Extract qāfiyeh (rhyming element)"""
        if not word:
            return ""
        if len(word) >= 3:
            return word[-3:]
        elif len(word) >= 2:
            return word[-2:]
        return word

    def _classify_rhyme_type(self, qafiyeh: str, phonetic: str) -> str:
        """Classify rhyme type"""
        if not qafiyeh:
            return "none"
        if len(phonetic) >= 3:
            return "rich"
        elif len(phonetic) >= 2:
            return "perfect"
        return "minimal"

    def calculate_rhyme_similarity(self, rhyme1: RhymeAnalysis, rhyme2: RhymeAnalysis) -> float:
        """Calculate phonetic similarity between two rhymes"""
        if not rhyme1.phonetic_rhyme or not rhyme2.phonetic_rhyme:
            return 0.0

        phone1, phone2 = rhyme1.phonetic_rhyme, rhyme2.phonetic_rhyme
        matches = sum(1 for a, b in zip(phone1, phone2) if a == b)
        max_len = max(len(phone1), len(phone2))

        if max_len == 0:
            return 1.0

        radif_bonus = 0.2 if rhyme1.radif == rhyme2.radif and rhyme1.radif else 0.0
        return min(1.0, (matches / max_len) + radif_bonus)

    def _empty_rhyme(self) -> RhymeAnalysis:
        """Return empty rhyme analysis"""
        return RhymeAnalysis(
            qafiyeh="",
            radif="",
            phonetic_rhyme="",
            rhyme_type="none",
            rhyme_position="none",
            confidence=0.0
        )


# =============================================================================
# STRUCTURAL ANALYZER
# =============================================================================

class StructuralAnalyzer:
    """ structural analyzer"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.aruz_analyzer = AruzMeterAnalyzer(self.config)
        self.rhyme_analyzer = AdvancedRhymeAnalyzer()
        self.phonetics = PersianTajikPhonetics()

    def analyze(self, poem_content: str) -> StructuralAnalysis:
        """Comprehensive structural analysis"""
        lines = [line.strip() for line in poem_content.split('\n') if line.strip()]

        if not lines:
            raise ValueError("No valid lines found in poem")

        line_analyses = []
        syllable_counts = []
        syllable_patterns = []
        rhyme_analyses = []

        for line in lines:
            phonetic = self.phonetics.analyze_phonetics(line)
            syllables = self.aruz_analyzer._extract_prosodic_syllables(line, phonetic)
            syllable_patterns.append(syllables)
            syllable_counts.append(len(syllables))

            rhyme = self.rhyme_analyzer.analyze_rhyme(line)
            rhyme_analyses.append(rhyme)

            aruz = self.aruz_analyzer.analyze_meter(line)

            line_analyses.append({
                'syllables': syllables,
                'rhyme': rhyme,
                'aruz': aruz
            })

        rhyme_pattern = self._generate_rhyme_pattern(rhyme_analyses)
        stanza_structure = self._detect_stanza_structure(lines, rhyme_pattern)
        avg_syllables = sum(syllable_counts) / len(syllable_counts) if syllable_counts else 0
        prosodic_consistency = self._calculate_prosodic_consistency(line_analyses)

        meters = [la['aruz'] for la in line_analyses]
        overall_aruz = self._determine_overall_meter(meters)

        return StructuralAnalysis(
            lines=len(lines),
            syllables_per_line=syllable_counts,
            syllable_patterns=syllable_patterns,
            aruz_analysis=overall_aruz,
            rhyme_scheme=rhyme_analyses,
            rhyme_pattern=rhyme_pattern,
            stanza_structure=stanza_structure,
            avg_syllables=round(avg_syllables, 2),
            prosodic_consistency=prosodic_consistency,
            meter_confidence=overall_aruz.confidence
        )

    def _generate_rhyme_pattern(self, rhyme_analyses: List[RhymeAnalysis]) -> str:
        """Generate rhyme scheme pattern"""
        if not rhyme_analyses:
            return ""

        pattern = []
        rhyme_groups = {}
        next_label = 'A'

        for rhyme in rhyme_analyses:
            rhyme_key = (rhyme.qafiyeh, rhyme.radif, rhyme.phonetic_rhyme)
            matched = False

            for prev_key, label in rhyme_groups.items():
                prev_rhyme = RhymeAnalysis(
                    qafiyeh=prev_key[0],
                    radif=prev_key[1],
                    phonetic_rhyme=prev_key[2],
                    rhyme_type="",
                    rhyme_position="end",
                    confidence=0.0
                )
                similarity = self.rhyme_analyzer.calculate_rhyme_similarity(rhyme, prev_rhyme)
                if similarity > 0.7:
                    pattern.append(label)
                    matched = True
                    break

            if not matched:
                pattern.append(next_label)
                rhyme_groups[rhyme_key] = next_label
                next_label = chr(ord(next_label) + 1)

        return ''.join(pattern)

    def _detect_stanza_structure(self, lines: List[str], rhyme_pattern: str) -> str:
        """Detect stanza structure"""
        if len(lines) <= 2:
            return "monostich" if len(lines) == 1 else "couplet"

        if rhyme_pattern.startswith('AA') and all(c in ['A', 'B'] for c in rhyme_pattern):
            return "ghazal"

        if len(lines) == 4 and rhyme_pattern in ['AABA', 'AAAA']:
            return "rubaiyat"

        if len(lines) > 10 and rhyme_pattern[:2] == 'AA':
            return "qasida"

        return "free_verse"

    def _calculate_prosodic_consistency(self, line_analyses: List[Dict]) -> float:
        """Calculate prosodic consistency"""
        if not line_analyses:
            return 0.0

        meters = [la['aruz'].identified_meter for la in line_analyses]
        unique_meters = set(meters)
        meter_consistency = 1.0 / len(unique_meters) if unique_meters else 0.0

        syllable_counts = [len(la['syllables']) for la in line_analyses]
        if syllable_counts:
            avg = sum(syllable_counts) / len(syllable_counts)
            variance = sum((c - avg) ** 2 for c in syllable_counts) / len(syllable_counts)
            syllable_consistency = 1.0 / (1.0 + variance / max(avg, 1))
        else:
            syllable_consistency = 0.0

        return (meter_consistency + syllable_consistency) / 2

    def _determine_overall_meter(self, meters: List[AruzAnalysis]) -> AruzAnalysis:
        """Determine overall meter"""
        if not meters:
            return AruzAnalysis(
                identified_meter="unknown",
                pattern_match="",
                confidence=MeterConfidence.NONE,
                pattern_accuracy=0.0,
                variations_detected=[],
                line_scansion=[],
                caesura_positions=[]
            )

        meter_counts = Counter(m.identified_meter for m in meters)
        most_common = meter_counts.most_common(1)[0][0]

        return max(
            (m for m in meters if m.identified_meter == most_common),
            key=lambda m: m.pattern_accuracy
        )


# =============================================================================
# CONTENT ANALYZER (Lexicon, Neologisms, Themes)
# =============================================================================

class ContentAnalyzer:
    """ content analyzer with lexicon support and neologism detection"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.lexicon = self._load_lexicon()
        
        # Define archaic words (classical Persian/Tajik)
        self.archaisms = {
            'зи', 'ки', 'чу', 'зеро', 'балки', 'андар', 'бар', 'аз-ан-ки',
            'ҳамана', 'бадин', 'бад-он', 'з-он', 'к-он', 'чунон', 'чунин',
            'инак', 'онак', 'биҳишт', 'дӯзах', 'фалак', 'қазо', 'қадар'
        }
        
        logger.info(f"ContentAnalyzer initialized with {len(self.lexicon)} lexicon entries")

    def _load_lexicon(self) -> Set[str]:
        """Load lexicon from configured file path"""
        try:
            lexicon_path = Path(self.config.lexicon_path)
            if lexicon_path.exists():
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(data, list):
                        return set(word.lower() for word in data)
                    elif isinstance(data, dict):
                        return set(word.lower() for word in data.keys())
                    return set()
            else:
                logger.warning(f"Lexicon file not found at: {lexicon_path}")
        except Exception as e:
            logger.error(f"Error loading lexicon: {e}")
        return set()

    def analyze(self, poem_content: str) -> ContentAnalysis:
        """Comprehensive content analysis"""
        # Extract words
        words = re.findall(r'[\wӣӯ]+', poem_content.lower())
        word_freq = Counter(words)
        
        # Find neologisms and archaisms
        neologisms = self._find_neologisms(words)
        archaisms = self._find_archaisms(words)
        
        # Analyze themes
        theme_distribution = self._analyze_themes(words)
        
        # Calculate lexical diversity (Type-Token Ratio)
        total_words = len(words)
        unique_words = len(set(words))
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Determine stylistic register
        stylistic_register = self._determine_register(words, archaisms, neologisms)
        
        return ContentAnalysis(
            word_frequencies=word_freq.most_common(20),
            neologisms=neologisms[:self.config.max_neologisms],
            archaisms=list(archaisms),
            theme_distribution=theme_distribution,
            lexical_diversity=round(lexical_diversity, 3),
            stylistic_register=stylistic_register,
            total_words=total_words,
            unique_words=unique_words
        )

    def _find_neologisms(self, words: List[str]) -> List[str]:
        """Find neologisms (words not in standard lexicon)"""
        if not self.lexicon:
            logger.warning("No lexicon loaded - neologism detection disabled")
            return []
        
        neologisms = []
        for word in set(words):
            if word not in self.lexicon and word not in self.archaisms:
                # Filter out numbers and very short words
                if not word.isdigit() and len(word) > 2:
                    neologisms.append(word)
        
        return sorted(neologisms)

    def _find_archaisms(self, words: List[str]) -> Set[str]:
        """Find archaic words"""
        return set(word for word in words if word in self.archaisms)

    def _analyze_themes(self, words: List[str]) -> Dict[str, int]:
        """Analyze thematic distribution"""
        theme_counts = {}
        
        for theme, keywords in self.config.themes.items():
            count = sum(1 for word in words if word in keywords)
            theme_counts[theme] = count
        
        return theme_counts

    def _determine_register(self, words: List[str], archaisms: Set[str],
                           neologisms: List[str]) -> str:
        """Determine stylistic register"""
        total_words = len(words)
        
        if not total_words:
            return "unknown"
        
        archaic_ratio = len(archaisms) / total_words
        neologism_ratio = len(neologisms) / total_words
        
        if archaic_ratio > 0.05:
            return "classical"
        elif neologism_ratio > 0.05:
            return "modern"
        elif archaic_ratio > 0.02 and neologism_ratio < 0.02:
            return "neo-classical"
        else:
            return "contemporary"

    def build_vocabulary_from_corpus(self, corpus_path: str) -> Dict[str, int]:
        """Build vocabulary dictionary from corpus file"""
        vocabulary = Counter()
        
        try:
            corpus_file = Path(corpus_path)
            if not corpus_file.exists():
                logger.error(f"Corpus file not found: {corpus_path}")
                return {}
            
            logger.info(f"Building vocabulary from {corpus_path}...")
            
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    words = re.findall(r'[\wӣӯ]+', line.lower())
                    vocabulary.update(words)
            
            logger.info(f"Built vocabulary with {len(vocabulary)} unique words")
            return dict(vocabulary)
            
        except Exception as e:
            logger.error(f"Error building vocabulary: {e}")
            return {}

    def save_vocabulary(self, vocabulary: Dict[str, int], output_path: str):
        """Save vocabulary to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(vocabulary, f, ensure_ascii=False, indent=2)
            logger.info(f"Vocabulary saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving vocabulary: {e}")


# =============================================================================
# POEM SPLITTER
# =============================================================================

class PoemSplitter:
    """Advanced poem splitter for Tajik Cyrillic poetry collections"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(f"{__name__}.PoemSplitter")

    def get_split_suggestions(self, text: str) -> List[int]:
        """Returns line indices where a new poem is likely to start"""
        lines = text.split('\n')
        suggestions = []

        for i, line in enumerate(lines):
            score = 0

            if self._looks_like_title(line):
                score += 2

            if i > 0 and not lines[i - 1].strip() and len(line.strip()) > 0:
                score += 1.5

            if re.match(r'^[\*\-=]{3,}$', line.strip()):
                suggestions.append(max(0, i - 1))
                continue

            if re.match(r'^\s*[\d]+[\.\)]\s*[A-ZА-Я]', line):
                score += 1

            if i > 0 and not lines[i - 1].strip() and line.strip() and line.strip()[0].isupper():
                score += 0.5

            if score >= 1.5:
                suggestions.append(i)

        if suggestions:
            filtered = [suggestions[0]]
            for s in suggestions[1:]:
                if s - filtered[-1] > 3:
                    filtered.append(s)
            suggestions = filtered

        return suggestions

    def _looks_like_title(self, line: str) -> bool:
        """Simple heuristic to recognize title lines"""
        line = line.strip()
        if not line or len(line) > 150:
            return False

        if line.endswith(('.', '!', '?', ':', ',')):
            return False

        if not line[0].isupper():
            return False

        if line.isupper():
            return False

        return True


# =============================================================================
# LITERARY ASSESSOR
# =============================================================================

class LiteraryAssessor:
    """Multi-perspective literary assessment"""

    @staticmethod
    def assess(structural: StructuralAnalysis, content: Optional[ContentAnalysis] = None) -> LiteraryAssessment:
        """Literary assessment based on structural and content analysis"""

        # German perspective - formal perfection
        german_score = 0
        if structural.rhyme_pattern in ['ABAB', 'AABB', 'ABBA', 'ABCABC']:
            german_score += 2
        if 8 <= structural.avg_syllables <= 12:
            german_score += 2
        if structural.prosodic_consistency > 0.8:
            german_score += 1

        # Persian tradition - classical forms
        persian_score = 0
        if structural.aruz_analysis.identified_meter != "unknown":
            persian_score += 2
        if structural.stanza_structure in ['ghazal', 'qasida', 'rubaiyat']:
            persian_score += 2
        if structural.prosodic_consistency > 0.7:
            persian_score += 1
        # Bonus for classical register
        if content and content.stylistic_register in ['classical', 'neo-classical']:
            persian_score += 1

        # Tajik elements
        tajik_score = 0
        if structural.lines >= 4:
            tajik_score += 1
        if structural.stanza_structure in ['ghazal', 'rubaiyat']:
            tajik_score += 2
        # Bonus for homeland themes
        if content:
            tajik_score += min(2, content.theme_distribution.get('Homeland', 0))

        # Modernist features
        modern_score = 0
        if structural.stanza_structure == "free_verse":
            modern_score += 2
        if structural.prosodic_consistency < 0.5:
            modern_score += 1
        # Bonus for neologisms and high lexical diversity
        if content:
            if len(content.neologisms) > 3:
                modern_score += 1
            if content.lexical_diversity > 0.7:
                modern_score += 1

        return LiteraryAssessment(
            german_perspective=min(5, german_score),
            persian_tradition=min(5, persian_score),
            tajik_elements=min(5, tajik_score),
            modernist_features=min(5, modern_score)
        )


# =============================================================================
# QUALITY VALIDATOR
# =============================================================================

class QualityValidator:
    """Validate analysis quality for scientific rigor"""

    @staticmethod
    def validate_analysis(analysis: ComprehensiveAnalysis) -> Dict[str, Any]:
        """Validate analysis quality"""
        warnings = []
        recommendations = []
        quality_score = 1.0

        if analysis.structural.meter_confidence == MeterConfidence.NONE:
            warnings.append("No reliable meter detected")
            recommendations.append("Manual prosodic verification recommended")
            quality_score *= 0.7

        if analysis.structural.prosodic_consistency < 0.5:
            warnings.append("Low prosodic consistency")
            recommendations.append("Check for textual corruption or free verse intention")
            quality_score *= 0.8

        if analysis.structural.lines < 2:
            warnings.append("Very short poem")
            recommendations.append("Statistical analysis not reliable for single lines")
            quality_score *= 0.5

        reliability = "high" if quality_score > 0.8 else "medium" if quality_score > 0.6 else "low"

        return {
            'quality_score': round(quality_score, 2),
            'reliability': reliability,
            'warnings': warnings,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# EXCEL REPORTER
# =============================================================================

class ExcelReporter:
    """Excel report generation"""

    def __init__(self):
        self.header_font = Font(bold=True, color="FFFFFF")
        self.header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
        self.border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )
        self.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def create_report(self, results: List[Dict[str, Any]], filename: str = "tajik_poetry_analysis.xlsx"):
        """Create Excel report"""
        try:
            wb = openpyxl.Workbook()
            self._create_overview_sheet(wb, results)
            self._create_structural_sheet(wb, results)
            self._create_quality_sheet(wb, results)

            wb.save(filename)
            logger.info(f"Report saved as: {filename}")

        except Exception as e:
            logger.error(f"Error creating report: {e}")
            raise

    def _create_overview_sheet(self, wb: openpyxl.Workbook, results: List[Dict[str, Any]]):
        """Create overview sheet"""
        ws = wb.active
        ws.title = "Overview"

        headers = [
            "ID", "Title", "Lines", "Meter", "Confidence", "Rhyme Pattern",
            "Stanza Form", "Avg Syllables", "Quality Score"
        ]

        for col_num, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_num, value=header)
            cell.font = self.header_font
            cell.fill = self.header_fill
            cell.border = self.border

        for row_num, result in enumerate(results, 2):
            analysis = result["analysis"]
            validation = result.get("validation", {})

            values = [
                result["poem_id"],
                result["title"],
                analysis.structural.lines,
                analysis.structural.aruz_analysis.identified_meter,
                analysis.structural.meter_confidence.value,
                analysis.structural.rhyme_pattern,
                analysis.structural.stanza_structure,
                analysis.structural.avg_syllables,
                validation.get("quality_score", "N/A")
            ]

            for col_num, value in enumerate(values, 1):
                cell = ws.cell(row=row_num, column=col_num, value=value)
                cell.border = self.border

    def _create_structural_sheet(self, wb: openpyxl.Workbook, results: List[Dict[str, Any]]):
        """Create structural analysis sheet"""
        ws = wb.create_sheet(title="Structural Analysis")

        headers = [
            "Poem ID", "Line #", "Line Text", "Syllables", "Meter Pattern",
            "Qāfiyeh", "Radīf", "Rhyme Type"
        ]

        for col_num, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_num, value=header)
            cell.font = self.header_font
            cell.fill = self.header_fill

        row_num = 2
        for result in results:
            poem_id = result["poem_id"]
            content = result["content"]
            structural = result["analysis"].structural

            lines = [line.strip() for line in content.split('\n') if line.strip()]

            for line_idx, line in enumerate(lines):
                syllable_count = structural.syllables_per_line[line_idx] if line_idx < len(
                    structural.syllables_per_line) else 0

                if line_idx < len(structural.syllable_patterns):
                    pattern = ''.join([s.weight.value for s in structural.syllable_patterns[line_idx]])
                else:
                    pattern = ""

                if line_idx < len(structural.rhyme_scheme):
                    rhyme = structural.rhyme_scheme[line_idx]
                    qafiyeh = rhyme.qafiyeh
                    radif = rhyme.radif
                    rhyme_type = rhyme.rhyme_type
                else:
                    qafiyeh = radif = rhyme_type = ""

                values = [
                    poem_id, line_idx + 1, line, syllable_count, pattern,
                    qafiyeh, radif, rhyme_type
                ]

                for col_num, value in enumerate(values, 1):
                    ws.cell(row=row_num, column=col_num, value=value)

                row_num += 1

    def _create_quality_sheet(self, wb: openpyxl.Workbook, results: List[Dict[str, Any]]):
        """Create quality metrics sheet"""
        ws = wb.create_sheet(title="Quality Metrics")

        headers = ["Poem ID", "Quality Score", "Reliability", "Warnings", "Recommendations"]

        for col_num, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_num, value=header)
            cell.font = self.header_font
            cell.fill = self.header_fill

        for row_num, result in enumerate(results, 2):
            validation = result.get("validation", {})

            values = [
                result["poem_id"],
                validation.get("quality_score", "N/A"),
                validation.get("reliability", "N/A"),
                "; ".join(validation.get("warnings", [])),
                "; ".join(validation.get("recommendations", []))
            ]

            for col_num, value in enumerate(values, 1):
                ws.cell(row=row_num, column=col_num, value=value)


# =============================================================================
# MAIN ANALYZER
# =============================================================================


@dataclass
class EnhancedStructuralAnalysis(StructuralAnalysis):
    """Erweiterte Strukturanalyse mit modernen Metriken"""
    modern_metrics: Optional[ModernVerseMetrics] = None
    is_free_verse: bool = False
    free_verse_confidence: float = 0.0
    modern_features: Dict[str, float] = field(default_factory=dict)

@dataclass 
class EnhancedComprehensiveAnalysis:
    """Erweiterte Gesamtanalyse"""
    structural: EnhancedStructuralAnalysis
    content: ContentAnalysis
    literary: LiteraryAssessment
    quality_metrics: Dict[str, float]
    corpus_ready: bool = False
    contribution_id: Optional[str] = None

class EnhancedTajikPoemAnalyzer(TajikPoemAnalyzer):
    """
    Erweiterter Analyzer mit:
    1. Freie-Vers-Erkennung
    2. Moderne Metriken
    3. Korpus-Beitragsfunktionalität
    """
    
    def __init__(self, config=None, enable_corpus: bool = True):
        super().__init__(config)
        self.modern_analyzer = ModernVerseAnalyzer()
        self.free_verse_classifier = FreeVerseClassifier()
        
        if enable_corpus:
            self.corpus_manager = TajikCorpusManager()
        else:
            self.corpus_manager = None
            
    def analyze_poem(self, poem_content: str, 
                    contributor_info: Optional[Dict] = None) -> EnhancedComprehensiveAnalysis:
        """
        Erweiterte Gedichtanalyse mit freier Vers-Erkennung
        """
        # Führe Basisanalyse durch
        basic_analysis = super().analyze_poem(poem_content)
        
        # Analysiere moderne Metriken
        modern_metrics = self.modern_analyzer.analyze(
            poem_content, 
            basic_analysis.structural.syllables_per_line
        )
        
        # Klassifiziere freie Verse
        is_free_verse = self.free_verse_classifier.is_free_verse(
            basic_analysis.structural,
            modern_metrics
        )
        
        # Passe Strukturanalyse für freie Verse an
        structural = self._enhance_structural_analysis(
            basic_analysis.structural,
            modern_metrics,
            is_free_verse
        )
        
        # Erweiterte Qualitätsmetriken
        quality_metrics = self._enhance_quality_metrics(
            basic_analysis.quality_metrics,
            structural,
            modern_metrics
        )
        
        # Bereite Korpus-Beitrag vor
        contribution_id = None
        if self.corpus_manager and contributor_info:
            contribution = self.corpus_manager.prepare_contribution(
                {
                    "poem_id": f"poem_{hash(poem_content) & 0xffffffff}",
                    "title": "Analyzed Poem",
                    "content": poem_content,
                    "analysis": basic_analysis,
                    "validation": quality_metrics
                },
                poem_content,
                contributor_info
            )
            contribution_id = contribution["contribution_id"]
            
        return EnhancedComprehensiveAnalysis(
            structural=structural,
            content=basic_analysis.content,
            literary=basic_analysis.literary,
            quality_metrics=quality_metrics,
            corpus_ready=self.corpus_manager is not None,
            contribution_id=contribution_id
        )
    
    def _enhance_structural_analysis(self, structural: StructuralAnalysis,
                                   modern_metrics: ModernVerseMetrics,
                                   is_free_verse: bool) -> EnhancedStructuralAnalysis:
        """Erweitert Strukturanalyse für freie Verse"""
        
        # Passe Metrik-Name für freie Verse an
        identified_meter = structural.aruz_analysis.identified_meter
        meter_confidence = structural.meter_confidence
        
        if is_free_verse and identified_meter == "ṭawīl":
            # ṭawīl ist oft falsch-positiv für freie Verse
            identified_meter = "free_verse"
            meter_confidence = MeterConfidence.LOW
            
        elif is_free_verse:
            identified_meter = "free_verse"
            
        # Vereinfache Reimmuster für freie Verse
        rhyme_pattern = structural.rhyme_pattern
        if is_free_verse and len(rhyme_pattern) > 20:
            unique_rhymes = len(set(rhyme_pattern))
            rhyme_pattern = f"free_rhyme_{unique_rhymes}unique"
            
        # Extrahiere moderne Features
        modern_features = {
            "enjambement_density": modern_metrics.enjambement_ratio,
            "line_variation": modern_metrics.line_length_variation,
            "prose_tendency": modern_metrics.prose_poetry_score,
            "visual_complexity": modern_metrics.visual_structure_score,
            "syntactic_parallelism": modern_metrics.syntactic_parallelism,
            "lexical_repetition": modern_metrics.lexical_repetition_score
        }
        
        return EnhancedStructuralAnalysis(
            lines=structural.lines,
            syllables_per_line=structural.syllables_per_line,
            syllable_patterns=structural.syllable_patterns,
            aruz_analysis=structural.aruz_analysis,
            rhyme_scheme=structural.rhyme_scheme,
            rhyme_pattern=rhyme_pattern,
            stanza_structure=structural.stanza_structure,
            avg_syllables=structural.avg_syllables,
            prosodic_consistency=structural.prosodic_consistency,
            meter_confidence=meter_confidence,
            modern_metrics=modern_metrics,
            is_free_verse=is_free_verse,
            free_verse_confidence=self._calculate_free_verse_confidence(
                structural, modern_metrics),
            modern_features=modern_features
        )
    
    def _enhance_quality_metrics(self, basic_metrics: Dict,
                               structural: EnhancedStructuralAnalysis,
                               modern_metrics: ModernVerseMetrics) -> Dict:
        """Erweitert Qualitätsmetriken"""
        enhanced = basic_metrics.copy()
        
        if structural.is_free_verse:
            enhanced["free_verse_analysis"] = {
                "confidence": structural.free_verse_confidence,
                "enjambement_score": modern_metrics.enjambement_ratio,
                "prose_poetry_score": modern_metrics.prose_poetry_score,
                "line_variation_score": modern_metrics.line_length_variation,
                "assessment": self._assess_free_verse_quality(structural, modern_metrics)
            }
            
            # Passe Warnungen für freie Verse an
            if "Low prosodic consistency" in enhanced.get("warnings", []):
                enhanced["warnings"].remove("Low prosodic consistency")
                enhanced["warnings"].append("Free verse detected - prosodic analysis limited")
                
        return enhanced
    
    def _calculate_free_verse_confidence(self, structural: StructuralAnalysis,
                                       modern_metrics: ModernVerseMetrics) -> float:
        """Berechnet Konfidenz für freie Vers-Klassifikation"""
        # Kombiniere mehrere Indikatoren
        indicators = [
            (structural.prosodic_consistency < 0.4, 0.8),
            (modern_metrics.enjambement_ratio > 0.3, 0.6),
            (modern_metrics.line_length_variation > 0.5, 0.7),
            (modern_metrics.prose_poetry_score > 0.6, 0.5),
            (structural.meter_confidence.value in ['low', 'none'], 0.9)
        ]
        
        confidence = sum(weight for condition, weight in indicators if condition)
        return min(1.0, confidence / 2.5)  # Normalisiere
    
    def _assess_free_verse_quality(self, structural: EnhancedStructuralAnalysis,
                                 modern_metrics: ModernVerseMetrics) -> str:
        """Bewertet Qualität freier Verse"""
        scores = []
        
        # Enjambement-Bewertung
        if 0.2 <= modern_metrics.enjambement_ratio <= 0.6:
            scores.append(1.0)
        else:
            scores.append(0.5)
            
        # Zeilenvariations-Bewertung
        if 0.3 <= modern_metrics.line_length_variation <= 0.8:
            scores.append(1.0)
        else:
            scores.append(0.5)
            
        # Visuelle Struktur
        if modern_metrics.visual_structure_score > 0.2:
            scores.append(0.8)
            
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score > 0.8:
            return "excellent_free_verse"
        elif avg_score > 0.6:
            return "good_free_verse"
        elif avg_score > 0.4:
            return "experimental_free_verse"
        else:
            return "irregular_free_verse"
    
    def contribute_to_corpus(self, analysis: EnhancedComprehensiveAnalysis,
                           user_info: Dict, save_locally: bool = True):
        """
        Trägt Analyse zum Korpus bei
        """
        if not self.corpus_manager:
            raise ValueError("Korpus-Manager nicht initialisiert")
            
        # Bereite Beitrag vor
        contribution = self.corpus_manager.prepare_contribution(
            {
                "poem_id": analysis.contribution_id or f"poem_{hash(str(analysis))}",
                "title": "User Contribution",
                "content": "",  # Wird vom Aufrufer bereitgestellt
                "analysis": analysis,
                "validation": analysis.quality_metrics
            },
            "",  # Rohtext wird separat benötigt
            user_info
        )
        
        if save_locally:
            self.corpus_manager.save_contribution(contribution)
            
        return contribution
    
    def export_for_git(self) -> str:
        """Exportiert Beiträge und gibt Git-Befehle zurück"""
        if not self.corpus_manager:
            return "Korpus-Manager nicht aktiviert"
            
        # Exportiere Beiträge
        export_path = self.corpus_manager.export_contributions_for_git()
        
        # Generiere Git-Befehle
        commands = self.corpus_manager.generate_git_commands()
        
        return f"""
        Beiträge exportiert nach: {export_path}
        
        Git-Befehle zum Hochladen:
        {commands}
        
        Nach dem Push: Erstellen Sie einen Pull Request auf GitHub!
        """
