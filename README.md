# Tajik Poetry Analyzer

A scientific-grade tool for analyzing Tajik and Persian poetry using classical ʿArūḍ (عروض) prosody with 16 classical Arabic-Persian meters.

## Features

- **16 Classical ʿArūḍ Meters**: ṭawīl, basīṭ, wāfir, kāmil, mutaqārib, hazaj, rajaz, ramal, sarīʿ, munsarih, khafīf, muḍāriʿ, muqtaḍab, mujtath, mutadārik, madīd
- **Qāfiyeh/Radīf Detection**: Advanced phonetic-based rhyme analysis
- **Neologism Detection**: Identifies words not in the 68,000+ word lexicon
- **Thematic Analysis**: Love, Nature, Homeland, Religion, Mysticism, Philosophy
- **Lexical Diversity**: Type-Token Ratio calculation
- **Stylistic Register**: Classical, Neo-classical, Contemporary, Modern
- **Excel Reports**: Comprehensive analysis export

## ⚠️ Important: Text Input Requirements

### The Problem with PDFs

**Automatic PDF recognition for Tajik Cyrillic text is currently unreliable.** This is due to:

1. **OCR limitations**: Tajik-specific characters (ӣ, ӯ, ҷ, ҳ, қ, ғ) are frequently misrecognized
2. **Layout issues**: Poetry formatting (line breaks, stanzas) is often destroyed
3. **Mixed scripts**: Some PDFs contain Arabic script mixed with Cyrillic, causing confusion
4. **Quality variance**: Scanned PDFs produce especially poor results

### Recommended Workflow

#### Option 1: Manual Transcription (Most Reliable)
```
1. Open your PDF
2. Manually copy/transcribe poems into a .txt file
3. Use ***** as separator between poems
4. Run the analyzer on the .txt file
```

#### Option 2: AI-Assisted Transcription (Recommended)
```
1. Upload your PDF to an AI assistant (Claude, ChatGPT, etc.)
2. Ask: "Please transcribe these Tajik poems into clean text, 
   preserving line breaks. Use ***** as separator between poems."
3. Review and correct any errors
4. Save as .txt file
5. Run the analyzer
```

#### Option 3: Clean Digital Text
If you have access to clean digital text (not scanned):
```
1. Copy text directly from the source
2. Format with ***** separators between poems
3. Run the analyzer
```

### Text Format Example

```
ТӮФОНҲОИ СОКИТ

Дар ин хароси чодуйй
чизе бигӯ
бо ин гумкардаҳои хеш
—Куҷост нишони кӯдакӣ?
—Дар хотираҳои занҷирӣ,
Дар ёди сафедорҳои ғамгини кӯчаҳои
фаромӯшӣ

*****

ҲАҶМИ БОРОНИ

Ман истодаам
дар имтидоди ғамгини зиндагӣ,
дар ҳаҷми боронии лаҳзаҳо

*****

ЭЙ ПИНҲОН!

Бозрасид
он зани муноди рӯъёҳои ман.
```

## Installation

```bash
git clone https://github.com/6gdtfmmdh6-sketch/tajikpoemanalyzer.git
cd tajikpoemanalyzer
pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)
```bash
streamlit run ui.py
```

### Python API
```python
from analyzer import TajikPoemAnalyzer, AnalysisConfig

config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
analyzer = TajikPoemAnalyzer(config)

poem = """
Дар кӯҳсори ватан гулҳо мешукуфанд,
Дили ошиқ аз муҳаббат меларзад.
"""

result = analyzer.analyze_poem(poem)

print(f"Meter: {result.structural.aruz_analysis.identified_meter}")
print(f"Rhyme Pattern: {result.structural.rhyme_pattern}")
print(f"Neologisms: {result.content.neologisms}")
print(f"Themes: {result.content.theme_distribution}")
```

### Batch Analysis
```python
# Analyze multiple poems from a file
results = analyzer.analyze_file('data/poems.txt', 'output/analysis.xlsx')
```

## Data Files

| File | Size | Description |
|------|------|-------------|
| `tajik_lexicon.json` | 1.4 MB | 68,060 Tajik words for neologism detection |
| `tajik_corpus.txt` | 404 MB | Large corpus for vocabulary building |
| `shahnama.txt` | 154 KB | Shahnameh excerpts for testing |

## Analysis Output

### Structural Analysis
- Line count, syllable patterns
- ʿArūḍ meter identification with confidence level
- Rhyme scheme (AABB, ABAB, etc.)
- Stanza form (ghazal, rubaiyat, qasida, free verse)

### Content Analysis
- Word frequencies
- Neologisms (words not in lexicon)
- Archaisms (classical vocabulary)
- Thematic distribution
- Lexical diversity score
- Stylistic register

### Quality Metrics
- Analysis confidence score
- Reliability assessment
- Warnings and recommendations

## Technical Notes

### Supported Characters
The analyzer fully supports Tajik Cyrillic including:
- Standard Cyrillic: А-Я, а-я
- Tajik-specific: Ӣ ӣ, Ӯ ӯ, Ҷ ҷ, Ҳ ҳ, Қ қ, Ғ ғ

### Meter Detection Accuracy
- **High confidence (>90%)**: Clear classical meters
- **Medium confidence (70-90%)**: Some variation from standard patterns
- **Low confidence (50-70%)**: Possible free verse or mixed meters
- **None (<50%)**: Unable to identify meter reliably

## Contributing

Contributions welcome! Areas of interest:
- Improved OCR pipeline for Tajik text
- Additional meter patterns and variations
- Expanded lexicon coverage
- Better archaism detection

## License

MIT License

## Acknowledgments

- Classical ʿArūḍ theory based on al-Khalīl ibn Aḥmad's system
- Tajik lexicon derived from contemporary Tajik corpus data
