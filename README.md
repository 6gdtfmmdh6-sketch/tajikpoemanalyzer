# Tajik Poetry Analyzer

Analysewerkzeug für tadschikische Lyrik mit Fokus auf **freien Vers (shi'ri nou)** und klassische Strukturanalyse.

## Features

- **Strukturanalyse**: Zeilen, Silben, Strophenformen
- **ʿArūḍ-Metren**: 16 klassische arabisch-persische Metren (experimentell)
- **Qāfiyeh/Radīf**: Phonetische Reimerkennung
- **Free-Verse-Metriken**: Enjambement, Zeilenlängenvariation, semantische Dichte
- **Neologismen**: Erkennung anhand eines 68.000+ Wort-Lexikons
- **Themenanalyse**: Liebe, Natur, Heimat, Religion, Mystik, Philosophie
- **Excel-Export**: Umfassende Analyseberichte

---

## Schnellstart

### Option A: Docker (empfohlen)
```bash
cd ~/tajikpoemanalyzer
./deploy.sh
# Öffne http://localhost:8501
```
→ Siehe [DOCKER.md](DOCKER.md) für Details.

### Option B: Lokal
```bash
cd ~/tajikpoemanalyzer
pip install -r requirements.txt
streamlit run ui.py
```

---

## Textformat

Gedichte müssen als **UTF-8 .txt-Datei** vorliegen, getrennt durch `*****`:

```
ТӮФОНҲОИ СОКИТ

Дар ин хароси чодуйй
чизе бигӯ
бо ин гумкардаҳои хеш

*****

ҲАҶМИ БОРОНИ

Ман истодаам
дар имтидоди ғамгини зиндагӣ
```

### ⚠️ PDF-Hinweis
Automatische OCR für tadschikisches Kyrillisch ist unzuverlässig. Empfehlung:
1. PDF an Claude/ChatGPT hochladen
2. Transkription anfordern mit `*****` als Trenner
3. Als .txt speichern

---

## Python API

```python
from analyzer import EnhancedTajikPoemAnalyzer

analyzer = EnhancedTajikPoemAnalyzer()

poem = """
Ман истодаам
дар имтидоди ғамгини зиндагӣ,
дар ҳаҷми боронии лаҳзаҳо
"""

result = analyzer.analyze_poem(poem)

print(f"Zeilen: {result.structural.lines}")
print(f"Freier Vers: {result.structural.is_free_verse}")
print(f"Metrum: {result.structural.aruz_analysis.identified_meter}")
print(f"Neologismen: {result.content.neologisms}")
```

---

## Dateistruktur

```
tajikpoemanalyzer/
├── analyzer.py          # Haupt-Analysemodul
├── ui.py                # Streamlit Web-Interface
├── data/
│   ├── tajik_lexicon.json   # 68.060 Wörter
│   └── tajik_corpus.txt     # Korpus (404 MB)
├── Dockerfile           # Container-Definition
├── docker-compose.yml   # Service-Konfiguration
└── deploy.sh            # Deployment-Skript
```

---

## Tadschikische Sonderzeichen

| Zeichen | Unicode | Beschreibung |
|---------|---------|--------------|
| Ӣ ӣ | U+04E2/E3 | i mit Makron |
| Ӯ ӯ | U+04EE/EF | u mit Makron |
| Ҷ ҷ | U+04B6/B7 | dsch |
| Ҳ ҳ | U+04B2/B3 | h (pharyngal) |
| Қ қ | U+049A/9B | q (uvular) |
| Ғ ғ | U+0492/93 | gh |

---

## Lizenz

MIT License
