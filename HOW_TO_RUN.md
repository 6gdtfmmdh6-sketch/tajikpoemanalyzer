# Wie starte ich die Anwendung?

## Option 1: Web-UI (Empfohlen für PDF-Analyse)

### Start
```bash
cd /Users/moritzschlenstedt/tajik-poem-analyzer
streamlit run ui.py
```

### Was passiert dann?
1. Terminal zeigt: "You can now view your Streamlit app in your browser."
2. Browser öffnet sich automatisch (normalerweise auf `http://localhost:8501`)
3. Falls nicht, öffne manuell: `http://localhost:8501`

### Verwendung im Browser
1. **PDF hochladen**: Klicke auf "Browse files" und wähle eine PDF-Datei
   - Unterstützt normale PDFs (Text wird extrahiert)
   - Unterstützt gescannte PDFs (OCR wird automatisch verwendet)
2. **Text prüfen**: Erweitere "Extrahierter Text anzeigen" um zu sehen was gelesen wurde
3. **Analyse starten**: Klicke auf "Analyse starten"
4. **Ergebnisse**: Für jedes Gedicht siehst du:
   - Strukturelle Analyse (Zeilen, Silben, Metrum, Reimschema)
   - Inhaltliche Analyse (Häufigste Wörter, Themen)
   - Qualitätsmetriken

### UI-Features
- ✅ Einfaches, schlichtes Design
- ✅ PDF-Upload direkt im Browser
- ✅ Automatische Gedicht-Erkennung (trennt an `*****` oder Leerzeilen)
- ✅ Fortschrittsanzeige während Analyse
- ✅ Übersichtliche Ergebnisdarstellung

## Option 2: Kommandozeile (Python)

### Einzelnes Gedicht analysieren
```python
from app2 import TajikPoemAnalyzer, AnalysisConfig

# Analyzer initialisieren
config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
analyzer = TajikPoemAnalyzer(config=config)

# Gedicht analysieren
poem_text = """
Булбул дар боғ мехонад,
Гул мешукуфад.
Баҳор омад,
Дил шод аст.
"""

analysis = analyzer.analyze_poem(poem_text)
print(f"Aruz-Metrum: {analysis.structural.aruz_analysis.identified_meter}")
print(f"Reimschema: {analysis.structural.rhyme_pattern}")
```

### PDF analysieren
```python
from app2 import TajikPoemAnalyzer, AnalysisConfig
from pdf_handler import read_file_with_pdf_support

# PDF lesen (unterstützt normale und gescannte PDFs)
text = read_file_with_pdf_support('mein_gedicht.pdf')

# Analyzer initialisieren
config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
analyzer = TajikPoemAnalyzer(config=config)

# Gedichte analysieren
poems = text.split('*****')  # Oder andere Trennung
for poem_text in poems:
    if len(poem_text.strip()) > 20:
        analysis = analyzer.analyze_poem(poem_text)
        print(f"Metrum: {analysis.structural.aruz_analysis.identified_meter}")
```

## Option 3: Beispiel-Skript ausführen

```bash
python3 example_usage.py
```

Dies analysiert die Beispielgedichte in `data/poems.txt` und zeigt vollständige Ergebnisse.

## Tests ausführen

### Workflow-Test
```bash
python3 test_complete_workflow.py
```

**Erwartete Ausgabe:**
```
======================================================================
TEST: Vollständiger Workflow
======================================================================

✓ Test 1: Analyzer initialisieren...
  Analyzer geladen

✓ Test 2: Text-Datei lesen...
  6945 Zeichen gelesen aus data/poems.txt
  10 Gedichte gefunden

✓ Test 3: Gedicht analysieren...
  Zeilen: 14
  Durchschn. Silben: 15.6
  Aruz-Metrum: ramal
  Konfidenz: medium

✅ WORKFLOW ERFOLGREICH
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit
```

### "ModuleNotFoundError: No module named 'app2'"
Stelle sicher, dass du im richtigen Verzeichnis bist:
```bash
cd /Users/moritzschlenstedt/tajik-poem-analyzer
```

### OCR funktioniert nicht
Installiere System-Dependencies:

**macOS:**
```bash
brew install poppler tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-fas
```

### "Lexikon nicht gefunden"
Prüfe ob die Daten-Dateien existieren:
```bash
ls -lh data/
```

Sollte zeigen:
```
tajik_lexicon.json    1.4M
tajik_corpus.txt      404M
poems.txt             12K
```

## Verzeichnis-Struktur

```
tajik-poem-analyzer/
├── ui.py                      # Web-UI (START HIER!)
├── app2.py                    # Haupt-Analyzer
├── pdf_handler.py             # PDF/OCR Integration
├── example_usage.py           # Beispiel-Skript
├── test_complete_workflow.py  # Test
├── data/
│   ├── tajik_lexicon.json     # 1.4MB - KRITISCH
│   ├── tajik_corpus.txt       # 404MB - KRITISCH
│   └── poems.txt              # Beispiele
├── README.md                  # Vollständige Dokumentation
├── QUICK_START.md             # Schnellstart
├── HOW_TO_RUN.md              # Diese Datei
└── STATUS.md                  # Projekt-Status
```

## Wichtig

### API-Verwendung
```python
# RICHTIG ✅
from app2 import TajikPoemAnalyzer, AnalysisConfig
config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
analyzer = TajikPoemAnalyzer(config=config)
analysis = analyzer.analyze_poem("Gedicht als String")

# FALSCH ❌
from enhanced_tajik_analyzer import ...  # Falsches Modul
poem = PoemData(content="...")
analysis = analyzer.analyze_poem(poem)  # Erwartet String, nicht PoemData!
```

## Nächste Schritte

1. **Starte die UI**: `streamlit run ui.py`
2. **Lade eine PDF hoch** mit tadschikischen/persischen Gedichten
3. **Analysiere** und sieh dir die Ergebnisse an
4. Bei Fragen siehe README.md oder QUICK_START.md
