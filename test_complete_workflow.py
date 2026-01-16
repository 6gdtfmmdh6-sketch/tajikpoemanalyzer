#!/usr/bin/env python3
"""
Test: Vollständiger Workflow-Test
Testet ob Original-Analyzer + PDF-Handler zusammenarbeiten
"""

from app2 import TajikPoemAnalyzer, AnalysisConfig, PoemData
from pdf_handler import read_file_with_pdf_support
from pathlib import Path

def test_workflow():
    """Test kompletten Workflow: Datei lesen -> Analyzer -> Ausgabe"""

    print("=" * 70)
    print("TEST: Vollständiger Workflow")
    print("=" * 70)

    # 1. Test: Analyzer initialisieren
    print("\n✓ Test 1: Analyzer initialisieren...")
    config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
    analyzer = TajikPoemAnalyzer(config=config)
    print(f"  Analyzer geladen")

    # 2. Test: Text-Datei lesen
    print("\n✓ Test 2: Text-Datei lesen...")
    poems_path = Path('data/poems.txt')
    if poems_path.exists():
        text = read_file_with_pdf_support(poems_path)
        print(f"  {len(text)} Zeichen gelesen aus {poems_path}")

        # Gedichte trennen (nach *****)
        poems = [p.strip() for p in text.split('*****') if len(p.strip()) > 50]
        print(f"  {len(poems)} Gedichte gefunden")

        # 3. Test: Erstes Gedicht analysieren
        if poems:
            print("\n✓ Test 3: Gedicht analysieren...")
            poem_text = poems[0]

            analysis = analyzer.analyze_poem(poem_text)

            print(f"  Zeilen: {analysis.structural.lines}")
            print(f"  Durchschn. Silben: {analysis.structural.avg_syllables:.1f}")
            print(f"  Strophenstruktur: {analysis.structural.stanza_structure}")
            print(f"  Reimschema: {analysis.structural.rhyme_pattern}")

            if hasattr(analysis.structural, 'aruz_analysis'):
                print(f"  Aruz-Metrum: {analysis.structural.aruz_analysis.identified_meter}")
                print(f"  Konfidenz: {analysis.structural.aruz_analysis.confidence.value}")

            print(f"\n  Top 3 Wörter:")
            for word, count in analysis.content.word_frequencies[:3]:
                print(f"    - {word}: {count}x")

            print("\n✅ WORKFLOW ERFOLGREICH")
            return True
    else:
        print(f"  ⚠️  {poems_path} nicht gefunden")
        return False

    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_workflow()
