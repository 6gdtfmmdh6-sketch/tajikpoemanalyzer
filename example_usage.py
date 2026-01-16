#!/usr/bin/env python3
"""
Beispiel: Tajik Poetry Analyzer mit PDF-Unterstützung

Zeigt wie der Original-Analyzer mit PDF/OCR verwendet wird
"""

from pathlib import Path
from app2 import TajikPoemAnalyzer, AnalysisConfig
from pdf_handler import read_file_with_pdf_support


def analyze_file(file_path: str):
    """
    Analysiere Gedichte aus Datei (TXT oder PDF)

    Args:
        file_path: Pfad zur Eingabedatei
    """
    print(f"\n{'='*70}")
    print(f"Analysiere: {file_path}")
    print(f"{'='*70}\n")

    # 1. Datei einlesen (unterstützt TXT und PDF)
    print("📖 Lese Datei...")
    text = read_file_with_pdf_support(Path(file_path))
    print(f"   ✓ {len(text)} Zeichen gelesen")

    # 2. Analyzer initialisieren
    print("\n🔧 Initialisiere Analyzer...")
    config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
    analyzer = TajikPoemAnalyzer(config=config)
    print("   ✓ Analyzer bereit")

    # 3. Text in Gedichte aufteilen
    print("\n✂️  Teile Text in Gedichte...")
    # Einfache Aufteilung an Doppel-Newlines
    poems = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
    print(f"   ✓ {len(poems)} Gedichte gefunden")

    # 4. Analysiere jedes Gedicht
    print(f"\n{'='*70}")
    print("ANALYSE-ERGEBNISSE")
    print(f"{'='*70}\n")

    for i, poem_text in enumerate(poems[:5], 1):  # Erste 5 Gedichte
        print(f"\n📝 Gedicht {i}")
        print("-" * 70)

        try:
            # Analysiere
            analysis = analyzer.analyze_poem(poem_text)

            # Zeige Ergebnisse
            print(f"Titel: Gedicht {i}")
            print(f"\nInhalt (erste 100 Zeichen):")
            print(f"   {poem_text[:100]}...")

            print(f"\n📊 Strukturelle Analyse:")
            print(f"   Zeilen: {analysis.structural.lines}")
            print(f"   Durchschn. Silben/Zeile: {analysis.structural.avg_syllables:.1f}")
            print(f"   Strophenstruktur: {analysis.structural.stanza_structure}")
            print(f"   Reimschema: {analysis.structural.rhyme_pattern}")

            if hasattr(analysis.structural, 'aruz_analysis'):
                aruz = analysis.structural.aruz_analysis
                print(f"   Aruz-Metrum: {aruz.identified_meter}")
                print(f"   Konfidenz: {aruz.confidence.value}")
                print(f"   Pattern-Genauigkeit: {aruz.pattern_accuracy:.1%}")

            print(f"\n📚 Inhaltliche Analyse:")
            print(f"   Top 5 Wörter:")
            for word, count in analysis.content.word_frequencies[:5]:
                print(f"      - {word}: {count}x")

            if analysis.content.neologisms:
                print(f"   Neologismen: {', '.join(analysis.content.neologisms[:3])}")

            print(f"   Themen:")
            for theme, count in analysis.content.theme_distribution.items():
                if count > 0:
                    print(f"      - {theme}: {count} Erwähnungen")

            if hasattr(analysis, 'quality_metrics'):
                print(f"\n⭐ Qualität:")
                for metric, score in analysis.quality_metrics.items():
                    if isinstance(score, (int, float)):
                        print(f"   {metric}: {score:.2f}")

        except Exception as e:
            print(f"   ❌ Fehler bei Analyse: {e}")

        print()

    print(f"{'='*70}\n")


def main():
    """Hauptfunktion"""
    # Beispiel 1: Text-Datei
    print("\n🎯 BEISPIEL 1: Text-Datei")
    try:
        analyze_file('data/poems.txt')
    except FileNotFoundError:
        print("   ⚠️  data/poems.txt nicht gefunden")

    # Beispiel 2: PDF-Datei (falls vorhanden)
    print("\n🎯 BEISPIEL 2: PDF-Datei")
    pdf_path = 'beispiel.pdf'
    if Path(pdf_path).exists():
        analyze_file(pdf_path)
    else:
        print(f"   ⚠️  {pdf_path} nicht gefunden")
        print("   💡 Tipp: Lege eine PDF-Datei mit Gedichten an")

    # Beispiel 3: Einzelnes Gedicht
    print("\n🎯 BEISPIEL 3: Einzelnes Gedicht")
    sample_poem = """
Булбул дар боғ мехонад суруди зебо,
Гул мешукуфад дар субҳи навбаҳор.
Дили ман бо ишқи ватан пур аст,
Табиати зебои Тоҷикистон.
    """

    config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
    analyzer = TajikPoemAnalyzer(config=config)

    try:
        analysis = analyzer.analyze_poem(sample_poem.strip())
        print(f"Strophenstruktur: {analysis.structural.stanza_structure}")
        print(f"Reimschema: {analysis.structural.rhyme_pattern}")
        print(f"Zeilen: {analysis.structural.lines}")
    except Exception as e:
        print(f"Fehler: {e}")


if __name__ == "__main__":
    main()
