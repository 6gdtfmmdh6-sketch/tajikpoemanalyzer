# Tajik Poetry Analyzer

Analysis tool for Tajik poetry with focus on **free verse (shi'ri nou)** and classical structure analysis.

## Features

- **Structure analysis**: Lines, syllables, stanza forms
- **ʿArūḍ meters**: 16 classical Arabic-Persian meters (experimental)
- **Qāfiyeh/Radīf**: Phonetic rhyme recognition
- **Free verse metrics**: Enjambment, line length variation, semantic density
- **Neologisms**: Detection based on a 68,000+ word lexicon
- **Theme analysis**: Love, nature, homeland, religion, mysticism, philosophy
- **Excel export**: Comprehensive analysis reports

---

## Data Policy: Texts stay local, data travels

**Never online:** full poem texts (`data/*.txt` volumes, `tajik_corpus/corpus/master.json`,
`tajik_corpus/contributions/`, the poetry library, snapshots). All of these are
git-ignored and docker-ignored. At runtime Docker mounts them as volumes; they are
never baked into the image.

**Shareable:** the derived feature layer. `python knowledge_export.py features`
(or the **Export** page in the app) produces `tajik_corpus/exports/features_public.json`:
per poem a SHA-256 text hash, the incipit (first line, ≤60 chars), bibliographic
metadata and the complete analysis (meter, rhyme, radīf, syllables, MTLD, themes).
Texts cannot be reconstructed from this file — publish it freely (GitHub, Zenodo).

**Backup / hand-over:** `python knowledge_export.py snapshot` bundles the complete
private research state into `snapshots/knowledge_snapshot_<date>.tar.gz`. Restore
with `python knowledge_export.py restore <file>` or via the Export page. Share this
archive only within a defined research group (cf. § 60d UrhG), never publicly.

**One-time cleanup:** old commits still contain full texts. Run
`scripts/purge_history.sh` (requires `git-filter-repo`), then force-push.

## Repository layout

- `analyzer.py` — core analysis (to be split into a package; see Roadmap)
- `knowledge_export.py` — feature export / snapshot / restore
- `pages/` — Streamlit UI (1 Analyze, 2 Library, 3 Visualize, 4 Corpus, 5 Export)
- `scripts/` — one-off maintenance scripts (batch migration, history purge)
- `experimental/` — unwired prototypes (Flask API, network analysis, MARC21 export,
  integration demo). Not imported by the app.

## Roadmap

1. Integrate `radif_detector.py` (structural `ClassicalRadifDetector`) into
   `analyzer.py`, replacing the statistical `EnhancedRadifDetector` — see
   `experimental/integration_demo.py`.
2. Split `analyzer.py` (5k lines, 44 classes) into modules: phonetics, aruz,
   structure, content, reporting.
3. `LiteraryAssessor` scores are an **experimental heuristic** — do not cite them
   as measurements.

---

## Quick Start

### Option A: Docker (recommended)
```bash
cd ~/tajikpoemanalyzer
./deploy.sh
# Open http://localhost:8501
```
→ See [DOCKER.md](DOCKER.md) for details.

### Option B: Local
```bash
cd ~/tajikpoemanalyzer
pip install -r requirements.txt
streamlit run ui.py
```

---

## Text Format

Poems must be provided as **UTF-8 .txt files**, separated by `*****`:

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

### ⚠️ PDF Note
Automatic OCR for Tajik Cyrillic is unreliable. Recommendation:
1. Upload PDF to Claude/ChatGPT
2. Request transcription with `*****` as separator
3. Save as .txt

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

print(f"Lines: {result.structural.lines}")
print(f"Free verse: {result.structural.is_free_verse}")
print(f"Meter: {result.structural.aruz_analysis.identified_meter}")
print(f"Neologisms: {result.content.neologisms}")
```

---

## File Structure

```
tajikpoemanalyzer/
├── analyzer.py          # Main analysis module
├── ui.py                # Streamlit web interface
├── data/
│   ├── tajik_lexicon.json   # 68,060 words
│   └── tajik_corpus.txt     # Corpus (404 MB)
├── Dockerfile           # Container definition
├── docker-compose.yml   # Service configuration
└── deploy.sh            # Deployment script
```

---

## Tajik Special Characters

| Character | Unicode | Description |
|-----------|---------|-------------|
| Ӣ ӣ | U+04E2/E3 | i with macron |
| Ӯ ӯ | U+04EE/EF | u with macron |
| Ҷ ҷ | U+04B6/B7 | dzh |
| Ҳ ҳ | U+04B2/B3 | h (pharyngeal) |
| Қ қ | U+049A/9B | q (uvular) |
| Ғ ғ | U+0492/93 | gh |

---

## License

MIT License

# Таҳлилгари Шеъри Тоҷикӣ

Абзоре барои таҳлили шеъри тоҷикӣ бо диққати асосӣ ба **шеъри назм (шиъри нав)** ва таҳлили сохти классикӣ.

## Хусусиятҳо

- **Таҳлили сохт**: Сатрҳо, ҳиҷоҳо, шаклҳои строфаҳо
- **Вазнҳои Арудӣ**: 16 вазни классикии арабӣ-форсӣ (таҷрибавӣ)
- **Қофия/Радиф**: Шинохти фонетикии қофия
- **Ченакҳои шеъри назм**: Энжамбеман, ихтилофи дарозии сатр, зичии маъноӣ
- **Навкалимаҳо**: Шинохт бар асоси луғати 68,000+ калима
- **Таҳлили мавзӯъҳо**: Ишқ, табиат, ватан, дин, ирфон, фалсафа
- **Содироти Excel**: Ҳисоботҳои таҳлилии фарох

---

## Оғози Зуд

### Имкони A: Docker (тавсияшаванда)
```bash
cd ~/tajikpoemanalyzer
./deploy.sh
# Кушодан http://localhost:8501
```
→ Барои тафсилот [DOCKER.md](DOCKER.md)-ро бубинед.

### Имкони B: Маҳаллӣ
```bash
cd ~/tajikpoemanalyzer
pip install -r requirements.txt
streamlit run ui.py
```

---

## Формати Матн

Шеърҳо бояд ҳамчун **файлҳои .txt бо рамзи UTF-8** бо истифода аз `*****` ҳамчун ҷудокунӣ пешкаш шаванд:

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

### ⚠️ Тавсия оид ба PDF
OCR худкори барои ҳуруфи кириллии тоҷикӣ нисбатан номуътамад аст. Тавсия:
1. PDF-ро ба Claude/ChatGPT бор кунед
2. Транскрипсиони бо ҷудокундаи `*****` талаб кунед
3. Ҳамчун .txt захира кунед

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

print(f"Сатрҳо: {result.structural.lines}")
print(f"Шеъри назм: {result.structural.is_free_verse}")
print(f"Вазн: {result.structural.aruz_analysis.identified_meter}")
print(f"Навкалимаҳо: {result.content.neologisms}")
```

---

## Сохтори Файлҳо

```
tajikpoemanalyzer/
├── analyzer.py          # Модули асосии таҳлил
├── ui.py                # Интерфейси вебии Streamlit
├── data/
│   ├── tajik_lexicon.json   # 68,060 калима
│   └── tajik_corpus.txt     # Корпус (404 MB)
├── Dockerfile           # Таърифи контейнер
├── docker-compose.yml   # Танзими хидмат
└── deploy.sh            # Скрипти ба кор андохтан
```

---

## Аломатҳои Вижаи Тоҷикӣ

| Аломат | Unicode | Тавсиф |
|--------|---------|---------|
| Ӣ ӣ | U+04E2/E3 | и бо макрон |
| Ӯ ӯ | U+04EE/EF | у бо макрон |
| Ҷ ҷ | U+04B6/B7 | ҷ |
| Ҳ ҳ | U+04B2/B3 | ҳ (ҳалқӣ) |
| Қ қ | U+049A/9B | қ (қалъаӣ) |
| Ғ ғ | U+0492/93 | ғ |

---

## Иҷозатнома

Иҷозатномаи MIT
