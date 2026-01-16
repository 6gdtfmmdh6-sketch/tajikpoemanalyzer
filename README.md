# Tajik Poetry Analyzer

Wissenschaftliches Tool zur Analyse tadschikischer/persischer Poesie mit Aruz-Metrik-Analyse, phonetischer Transkription und PDF/OCR-Unterstützung.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### Kern-Features
- ✅ **Wissenschaftliche Aruz-Metrik-Analyse** - 8 klassische Bahren (Hazaj, Ramal, Mutaqarib, Rajaz, Kamil, Tawil, Basit, Wafir)
- ✅ **Phonetische Transkription** - Persisch/Tajik Phonem-Mapping
- ✅ **Reimschema-Erkennung** - Qafiyeh/Radif Analyse
- ✅ **Strukturanalyse** - Zeilen, Silben, Strophenformen
- ✅ **Inhaltsanalyse** - Worte, Themen, Neologismen, Archaismen
- ✅ **Lexikon-basierte Analyse** - 1.4MB Tajik-Lexikon
- ✅ **Korpus-Validierung** - 404MB Tajik-Korpus (via Git LFS)

### PDF/OCR Features
- 🆕 **PDF-Textextraktion** - Normale PDFs mit PyPDF2
- 🆕 **OCR für gescannte PDFs** - Tesseract mit Farsi/Tajik/Russisch
- 🆕 **Web-UI** - Streamlit Browser-Interface für PDF-Upload
- 🆕 **Automatische Encoding-Erkennung**
- 🆕 **Bidirektionale Textunterstützung** - Arabisch/Persisch

## Installation

### 1. Repository klonen
```bash
git clone https://github.com/6gdtfmmdh6-sketch/tajikpoemanalyzer.git
cd tajikpoemanalyzer
```

### 2. Git LFS installieren (für Korpus-Datei)
```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Git LFS initialisieren
git lfs install
git lfs pull
```

### 3. Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. System-Dependencies für OCR (Optional)
```bash
# macOS
brew install poppler tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-fas tesseract-ocr-rus
```

## Verwendung

### Web-UI (Empfohlen)
```bash
streamlit run ui.py
```

Dann im Browser:
1. PDF hochladen
2. "Analyse starten" klicken
3. Ergebnisse ansehen

### Kommandozeile
```python
from app2 import TajikPoemAnalyzer, AnalysisConfig
from pdf_handler import read_file_with_pdf_support

# PDF lesen
text = read_file_with_pdf_support('gedichte.pdf')

# Analyzer initialisieren
config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
analyzer = TajikPoemAnalyzer(config=config)

# Analysieren
for poem_text in text.split('\n\n'):
    if len(poem_text) > 20:
        analysis = analyzer.analyze_poem(poem_text)
        print(f"Aruz-Metrum: {analysis.structural.aruz_analysis.identified_meter}")
        print(f"Reimschema: {analysis.structural.rhyme_pattern}")
```

### Beispiel ausführen
```bash
python3 example_usage.py
```

## Projektstruktur

```
tajikpoemanalyzer/
├── ui.py                      # Streamlit Web-UI
├── app2.py                    # Haupt-Analyzer (85KB)
├── pdf_handler.py             # PDF/OCR Integration
├── ocr_processor.py           # OCR Engine
├── phonetic_utils.py          # Phonetische Utilities
├── data/
│   ├── tajik_lexicon.json     # 1.4MB Lexikon
│   ├── tajik_corpus.txt       # 404MB Korpus (Git LFS)
│   └── poems.txt              # Beispiele
├── tests/
│   └── test_validation_suite.py
└── requirements.txt
```

## Analyse-Ergebnisse

Der Analyzer liefert:
- **Strukturell**: Zeilen, Silben, Strophen, Aruz-Metrum, Reimschema
- **Inhaltlich**: Worte, Themen, Neologismen, Archaismen
- **Qualität**: Literarische Bewertung
- **Theoretisch**: Übersetzungstheorie (Ette/Bachmann-Medick), Semiotik (Lotman)

## Tests

```bash
# Workflow-Test
python3 test_complete_workflow.py

# Alle Tests
pytest tests/
```

## Daten-Dateien

### Lexikon (1.4 MB)
- **Zweck**: Wörterbuch für Wort-Analyse
- **Inhalt**: Tadschikische Wörter
- **Status**: ✅ In Git enthalten

### Korpus (404 MB)
- **Zweck**: Statistische Analyse
- **Inhalt**: Tadschikischer Text-Korpus
- **Status**: ⚠️ Via Git LFS (nach `git lfs pull` verfügbar)

## Technische Details

### Aruz-Metrik
Das System implementiert klassische persische Prosodie:
- Hazaj, Ramal, Mutaqarib, Rajaz, Kamil, Tawil, Basit, Wafir
- Silbengewicht-Berechnung (schwer/leicht)
- Pattern-Matching mit Konfidenz-Scores

### OCR
- Tesseract mit Farsi/Tajik/Russisch Support
- Automatische Erkennung gescannter PDFs
- Async-Verarbeitung für Performance

## Zusammenarbeit

Für deine Mitarbeiter:

```bash
# Repository klonen
git clone https://github.com/6gdtfmmdh6-sketch/tajikpoemanalyzer.git
cd tajikpoemanalyzer

# Git LFS installieren und Korpus herunterladen
brew install git-lfs  # oder apt-get install git-lfs
git lfs install
git lfs pull

# Dependencies installieren
pip install -r requirements.txt

# UI starten
streamlit run ui.py
```

## Dokumentation

- [QUICK_START.md](QUICK_START.md) - Schnellstart-Anleitung
- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Ausführliche Anleitung
- [DATA_README.md](DATA_README.md) - Informationen zu Daten-Dateien

## Lizenz

MIT License

## Credits

- Original Analyzer: Wissenschaftliche Implementierung mit echter Aruz-Analyse
- PDF/OCR Integration: Erweiterte Funktionalität für digitale Korpora
- Korpus: Tadschikische Textsammlung (404MB)

## Support

Bei Problemen:
1. Siehe [HOW_TO_RUN.md](HOW_TO_RUN.md) für Troubleshooting
2. Teste mit `python3 test_complete_workflow.py`
3. Erstelle ein Issue auf GitHub
