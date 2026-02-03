#!/usr/bin/env python3
"""
Integration Guide for Fixed Tajik Poetry Analyzer
==================================================

This file shows how to integrate the three fixes into the existing analyzer.py

Fixes Applied:
--------------
1. phonetic_system.py - Replaces IPA with phoneme classes + IJMES/DMG
2. radif_detector.py - Structural Radīf detection (not statistical)
3. poem_splitter.py - Proper title extraction with metadata communication

How to Use:
-----------
Option A: Replace classes in analyzer.py (recommended for clean code)
Option B: Monkey-patch existing analyzer (quick fix)
Option C: Create wrapper classes (backward compatible)

This file demonstrates Option B (quick integration).
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import fixes
from phonetic_system import TajikPhonology, ScientificTransliterator
from radif_detector import ClassicalRadifDetector, RadifResult
from poem_splitter import EnhancedPoemSplitter, PoemData, analyze_poem_safely

# Import original analyzer
from analyzer import (
    TajikPoemAnalyzer,
    EnhancedTajikPoemAnalyzer,
    StructuralAnalyzer,
    PersianTajikPhonetics,
    EnhancedRadifDetector,
)


def patch_analyzer():
    """
    Patch the existing analyzer to use the new fixed components.
    
    Call this at startup before creating analyzer instances.
    """
    
    # === PATCH 1: Replace phonetics with TajikPhonology ===
    
    original_phonetics = PersianTajikPhonetics
    
    class PatchedPhonetics(original_phonetics):
        """Phonetics class that uses TajikPhonology for prosodic analysis."""
        
        def __init__(self):
            super().__init__()
            self.tajik_phonology = TajikPhonology()
            self.transliterator = ScientificTransliterator()
        
        def get_prosodic_pattern(self, text: str) -> str:
            """Use new phonology for prosodic pattern."""
            return self.tajik_phonology.get_prosodic_pattern(text)
        
        def to_ijmes(self, text: str) -> str:
            """Get IJMES transliteration for scientific citation."""
            return self.tajik_phonology.to_ijmes(text)
        
        def to_dmg(self, text: str) -> str:
            """Get DMG transliteration for German academic citation."""
            return self.tajik_phonology.to_dmg(text)
    
    # Monkey-patch
    import analyzer
    analyzer.PersianTajikPhonetics = PatchedPhonetics
    print("[PATCH] PersianTajikPhonetics → PatchedPhonetics (with IJMES/DMG)")
    
    
    # === PATCH 2: Replace Radīf detector ===
    
    original_radif = EnhancedRadifDetector
    
    class PatchedRadifDetector(original_radif):
        """Radīf detector that uses classical Persian rules."""
        
        def __init__(self):
            super().__init__()
            self.classical_detector = ClassicalRadifDetector()
        
        def detect_radif_pattern(self, poem_content: str):
            """Use classical detection instead of statistical."""
            result = self.classical_detector.detect(poem_content)
            
            # Convert to original format for compatibility
            from analyzer import RadifAnalysis
            return RadifAnalysis(
                radif_present=result.radif_present,
                radif_text=result.radif_text,
                radif_frequency=result.frequency,
                lines_without_radif=result.cleaned_lines,
                original_lines=poem_content.split('\n'),
                meter_corrected=None
            )
    
    # Monkey-patch
    analyzer.EnhancedRadifDetector = PatchedRadifDetector
    print("[PATCH] EnhancedRadifDetector → PatchedRadifDetector (classical rules)")
    
    
    # === PATCH 3: Fix poem splitting ===
    
    # Store original methods
    original_split_poems_enhanced = EnhancedTajikPoemAnalyzer._split_poems
    original_split_poems_classical = TajikPoemAnalyzer._split_poems
    
    def patched_split_poems(self, text: str):
        """Use EnhancedPoemSplitter with proper metadata."""
        splitter = EnhancedPoemSplitter()
        poems = splitter.split_text(text)
        
        # Convert to original PoemData format
        from analyzer import PoemData as OriginalPoemData
        result = []
        for poem in poems:
            result.append(OriginalPoemData(
                title=poem.title,
                content=poem.get_analysis_text(),  # KEY: Use analysis text (title excluded)
                poem_id=poem.poem_id,
                metadata={
                    'title_extracted': poem.metadata.title_extracted,
                    'title_line_number': poem.metadata.title_line_number,
                    'raw_content': poem.metadata.raw_content,
                }
            ))
        return result
    
    # Monkey-patch both analyzers
    EnhancedTajikPoemAnalyzer._split_poems = patched_split_poems
    TajikPoemAnalyzer._split_poems = patched_split_poems
    print("[PATCH] _split_poems → patched_split_poems (title metadata)")
    
    print("\n[SUCCESS] All patches applied. Analyzer is now using fixed components.\n")


# === DEMONSTRATION ===

def demo_with_patches():
    """Demonstrate the patched analyzer."""
    
    # Apply patches
    patch_analyzer()
    
    # Create analyzer (now uses patched components)
    analyzer = EnhancedTajikPoemAnalyzer()
    
    # Test poem with title and Radīf
    test_poem = """
ВАҚТ

Лаҳзаи ширин гузашт, ин аст вақт
Дил ба ёди ёр месӯзад ҳар вақт
Зиндагӣ мисли шароре аст вақт
Бояд онро қадр донист ҳар вақт
"""
    
    print("=== Testing Patched Analyzer ===\n")
    print(f"Input poem:\n{test_poem}")
    print("-" * 50)
    
    # Analyze
    results = analyzer.analyze_text(test_poem)
    
    if results:
        result = results[0]
        analysis = result['analysis']
        
        print(f"\nTitle: {result['title']}")
        print(f"Lines analyzed: {analysis.structural.lines}")
        print(f"Avg syllables: {analysis.structural.avg_syllables}")
        print(f"Radīf detected: {[r.radif for r in analysis.structural.rhyme_scheme if r.radif]}")
        print(f"Stanza structure: {analysis.structural.stanza_structure}")
        
        # Test IJMES/DMG output
        phonology = TajikPhonology()
        print(f"\nIJMES title: {phonology.to_ijmes('ВАҚТ')}")
        print(f"DMG title: {phonology.to_dmg('ВАҚТ')}")


def demo_standalone():
    """Demonstrate standalone usage of new modules."""
    
    print("=== Standalone Module Demo ===\n")
    
    # 1. Phonology
    from phonetic_system import TajikPhonology
    phonology = TajikPhonology()
    
    line = "Дар кӯҳсори ватан гулҳо мешукуфанд"
    print(f"Line: {line}")
    print(f"IJMES: {phonology.to_ijmes(line)}")
    print(f"DMG: {phonology.to_dmg(line)}")
    print(f"Prosodic: {phonology.get_prosodic_pattern(line)}")
    print(f"Syllables: {phonology.count_syllables(line)}")
    print()
    
    # 2. Radīf detection
    from radif_detector import ClassicalRadifDetector
    detector = ClassicalRadifDetector()
    
    poem = """
Лаҳзаи ширин гузашт, ин аст вақт
Дил ба ёди ёр месӯзад ҳар вақт
Зиндагӣ мисли шароре аст вақт
Бояд онро қадр донист ҳар вақт
"""
    
    result = detector.detect(poem)
    print(f"Radīf: '{result.radif_text}'")
    print(f"Type: {result.radif_type}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Qāfiyeh: {result.qafiyeh_before_radif}")
    print()
    
    # 3. Poem splitting
    from poem_splitter import EnhancedPoemSplitter
    splitter = EnhancedPoemSplitter()
    
    text = """
ВАҚТ

Лаҳзаи ширин гузашт
Дил ба ёди ёр месӯзад

*****

МОДАР

Эй модар, эй меҳрубон
Дар оғӯши ту ёфтам ҷон
"""
    
    poems = splitter.split_text(text)
    for poem in poems:
        print(f"Title: '{poem.title}'")
        print(f"Title extracted: {poem.metadata.title_extracted}")
        print(f"Analysis text: {poem.get_analysis_text()[:30]}...")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration demo for fixed analyzer")
    parser.add_argument('--patched', action='store_true', 
                        help='Demo with patched analyzer')
    parser.add_argument('--standalone', action='store_true',
                        help='Demo standalone modules')
    
    args = parser.parse_args()
    
    if args.patched:
        demo_with_patches()
    elif args.standalone:
        demo_standalone()
    else:
        print("Usage:")
        print("  python integration_demo.py --patched    # Demo patched analyzer")
        print("  python integration_demo.py --standalone # Demo standalone modules")
        print()
        print("Running both demos...\n")
        demo_standalone()
        print("\n" + "=" * 60 + "\n")
        demo_with_patches()
