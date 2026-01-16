# modern_verse_analyzer.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
import statistics

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
