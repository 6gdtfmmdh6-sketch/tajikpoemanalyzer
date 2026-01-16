# Tajik Poetry Analyzer
 
Scientific tool for analyzing Tajik/Persian poetry with Aruz metric analysis, phonetic transcription, and PDF/OCR support.
 
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
 
## Features
 
### Core Features
- **Scientific Aruz Metric Analysis** - 8 classical meters (Hazaj, Ramal, Mutaqarib, Rajaz, Kamil, Tawil, Basit, Wafir)
- **Phonetic Transcription** - Persian/Tajik phoneme mapping
- **Rhyme Scheme Detection** - Qafiyeh/Radif analysis
- **Structural Analysis** - Lines, syllables, stanza forms
- **Content Analysis** - Words, themes, neologisms, archaisms
- **Lexicon-based Analysis** - 1.4MB Tajik lexicon
- **Corpus Validation** - 404MB Tajik corpus (via Git LFS)
 
### PDF/OCR Features
- **PDF Text Extraction** - Normal PDFs with PyPDF2
- **OCR for Scanned PDFs** - Tesseract with Farsi/Tajik/Russian
- **Web UI** - Streamlit browser interface for PDF upload
- **Automatic Encoding Detection**
- **Bidirectional Text Support** - Arabic/Persian
 
## Installation
 
### 1. Clone Repository
```bash
git clone https://github.com/6gdtfmmdh6-sketch/tajikpoemanalyzer.git
cd tajikpoemanalyzer
```
 
### 2. Install Git LFS (for corpus file)
```bash
# macOS
brew install git-lfs
 
# Ubuntu/Debian
sudo apt-get install git-lfs
 
# Initialize Git LFS
git lfs install
git lfs pull
```
 
### 3. Python Dependencies
```bash
pip install -r requirements.txt
```
 
### 4. System Dependencies for OCR (Optional)
```bash
# macOS
brew install poppler tesseract tesseract-lang
 
# Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-fas tesseract-ocr-rus
```
 
## Usage
 
### Web UI (Recommended)
```bash
streamlit run ui.py
```
 
Then in browser:
1. Upload PDF
2. Click "Start Analysis"
3. View results
 
### Command Line
```python
from app2 import TajikPoemAnalyzer, AnalysisConfig
from pdf_handler import read_file_with_pdf_support
 
# Read PDF
text = read_file_with_pdf_support('poems.pdf')
 
# Initialize analyzer
config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
analyzer = TajikPoemAnalyzer(config=config)
 
# Analyze
for poem_text in text.split('\n\n'):
    if len(poem_text) > 20:
        analysis = analyzer.analyze_poem(poem_text)
        print(f"Aruz Meter: {analysis.structural.aruz_analysis.identified_meter}")
        print(f"Rhyme Scheme: {analysis.structural.rhyme_pattern}")
```
 
### Run Example
```bash
python3 example_usage.py
```
 
## Project Structure
 
```
tajikpoemanalyzer/
├── ui.py                      # Streamlit Web UI
├── app2.py                    # Main analyzer (85KB)
├── pdf_handler.py             # PDF/OCR integration
├── ocr_processor.py           # OCR engine
├── phonetic_utils.py          # Phonetic utilities
├── data/
│   ├── tajik_lexicon.json     # 1.4MB lexicon
│   ├── tajik_corpus.txt       # 404MB corpus (Git LFS)
│   └── poems.txt              # Examples
├── tests/
│   └── test_validation_suite.py
└── requirements.txt
```
 
## Analysis Results
 
The analyzer provides:
- **Structural**: Lines, syllables, stanzas, Aruz meter, rhyme scheme
- **Content**: Words, themes, neologisms, archaisms
- **Quality**: Literary evaluation
- **Theoretical**: Translation theory (Ette/Bachmann-Medick), Semiotics (Lotman)
 
## Tests
 
```bash
# Workflow test
python3 test_complete_workflow.py
 
# All tests
pytest tests/
```
 
## Data Files
 
### Lexicon (1.4 MB)
- **Purpose**: Dictionary for word analysis
- **Content**: Tajik words
- **Status**: Included in Git
 
### Corpus (404 MB)
- **Purpose**: Statistical analysis
- **Content**: Tajik text corpus
- **Status**: Via Git LFS (available after `git lfs pull`)
 
## Technical Details
 
### Aruz Metrics
The system implements classical Persian prosody:
- Hazaj, Ramal, Mutaqarib, Rajaz, Kamil, Tawil, Basit, Wafir
- Syllable weight calculation (heavy/light)
- Pattern matching with confidence scores
 
### OCR
- Tesseract with Farsi/Tajik/Russian support
- Automatic detection of scanned PDFs
- Async processing for performance
 
## Collaboration
 
For collaborators:
 
```bash
# Clone repository
git clone https://github.com/6gdtfmmdh6-sketch/tajikpoemanalyzer.git
cd tajikpoemanalyzer
 
# Install Git LFS and download corpus
brew install git-lfs  # or apt-get install git-lfs
git lfs install
git lfs pull
 
# Install dependencies
pip install -r requirements.txt
 
# Start UI
streamlit run ui.py
```
 
## Documentation
 
- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Detailed instructions
- [DATA_README.md](DATA_README.md) - Information about data files
 
## License
 
MIT License
 
## Credits
 
- Original Analyzer: Scientific implementation with authentic Aruz analysis
- PDF/OCR Integration: Extended functionality for digital corpora
- Corpus: Tajik text collection (404MB)
 
## Support
 
For issues:
1. See [HOW_TO_RUN.md](HOW_TO_RUN.md) for troubleshooting
2. Test with `python3 test_complete_workflow.py`
3. Create an issue on GitHub
