#!/usr/bin/env python3
"""
Debug script to test the analysis functionality
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from analyzer import (
    TajikPoemAnalyzer,
    EnhancedTajikPoemAnalyzer,
    AnalysisConfig,
    MeterConfidence
)

def test_analysis():
    """Test basic analysis functionality"""
    print("Testing Tajik Poetry Analyzer...")
    print("=" * 60)
    
    # Sample Tajik poem
    sample_poem = """Булбул дар боғ мехонад суруди зебо,
Гул мешукуфад дар субҳи навбаҳор.
Дили ман бо ишқи ватан пур аст,
Табиати зебои Тоҷикистон."""
    
    print("Sample poem:")
    print(sample_poem)
    print("=" * 60)
    
    # Test Classical Analyzer
    print("\n1. Testing Classical Analyzer:")
    try:
        config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
        classical_analyzer = TajikPoemAnalyzer(config=config)
        
        print("  ✓ Analyzer initialized")
        
        analysis = classical_analyzer.analyze_poem(sample_poem)
        print("  ✓ Analysis completed")
        
        print(f"\n  Results:")
        print(f"    Lines: {analysis.structural.lines}")
        print(f"    Meter: {analysis.structural.aruz_analysis.identified_meter}")
        print(f"    Confidence: {analysis.structural.meter_confidence.value}")
        print(f"    Rhyme Pattern: {analysis.structural.rhyme_pattern}")
        print(f"    Total Words: {analysis.content.total_words}")
        print(f"    Lexical Diversity: {analysis.content.lexical_diversity:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_analysis():
    """Test enhanced analysis functionality"""
    print("\n2. Testing Enhanced Analyzer:")
    try:
        config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
        enhanced_analyzer = EnhancedTajikPoemAnalyzer(config=config, enable_corpus=False)
        
        print("  ✓ Enhanced analyzer initialized")
        
        analysis = enhanced_analyzer.analyze_poem(sample_poem)
        print("  ✓ Enhanced analysis completed")
        
        print(f"\n  Results:")
        print(f"    Lines: {analysis.structural.lines}")
        print(f"    Meter: {analysis.structural.aruz_analysis.identified_meter}")
        print(f"    Is Free Verse: {analysis.structural.is_free_verse}")
        print(f"    Free Verse Confidence: {analysis.structural.free_verse_confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First check if analyzer module is available
    try:
        from analyzer import TajikPoemAnalyzer
        print("✓ Analyzer module imported successfully")
    except ImportError as e:
        print(f"✗ Cannot import analyzer: {e}")
        sys.exit(1)
    
    # Run tests
    success1 = test_analysis()
    success2 = test_enhanced_analysis()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
