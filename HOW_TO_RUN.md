# How to Run the Application
 
## Option 1: Web UI (Recommended for PDF Analysis)
 
### Start
```bash
cd /Users/moritzschlenstedt/tajik-poem-analyzer
streamlit run ui.py
```
 
### What happens next?
1. Terminal shows: "You can now view your Streamlit app in your browser."
2. Browser opens automatically (usually at `http://localhost:8501`)
3. If not, open manually: `http://localhost:8501`
 
### Usage in Browser
1. **Upload PDF**: Click "Browse files" and select a PDF file
   - Supports normal PDFs (text extraction)
   - Supports scanned PDFs (OCR automatically used)
2. **Check text**: Expand "Show extracted text" to see what was read
3. **Start analysis**: Click "Start Analysis"
4. **Results**: For each poem you see:
   - Structural analysis (lines, syllables, meter, rhyme scheme)
   - Content analysis (most frequent words, themes)
   - Quality metrics
 
### UI Features
- Simple, clean design
- PDF upload directly in browser
- Automatic poem detection (splits at `*****` or blank lines)
- Progress indicator during analysis
- Clear result presentation
 
## Option 2: Command Line (Python)
 
### Analyze Single Poem
```python
from app2 import TajikPoemAnalyzer, AnalysisConfig
 
# Initialize analyzer
config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
analyzer = TajikPoemAnalyzer(config=config)
 
# Analyze poem
poem_text = """
Булбул дар боғ мехонад,
Гул мешукуфад.
Баҳор омад,
Дил шод аст.
"""
 
analysis = analyzer.analyze_poem(poem_text)
print(f"Aruz Meter: {analysis.structural.aruz_analysis.identified_meter}")
print(f"Rhyme Scheme: {analysis.structural.rhyme_pattern}")
```
 
### Analyze PDF
```python
from app2 import TajikPoemAnalyzer, AnalysisConfig
from pdf_handler import read_file_with_pdf_support
 
# Read PDF (supports normal and scanned PDFs)
text = read_file_with_pdf_support('my_poem.pdf')
 
# Initialize analyzer
config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
analyzer = TajikPoemAnalyzer(config=config)
 
# Analyze poems
poems = text.split('*****')  # Or other separator
for poem_text in poems:
    if len(poem_text.strip()) > 20:
        analysis = analyzer.analyze_poem(poem_text)
        print(f"Meter: {analysis.structural.aruz_analysis.identified_meter}")
```
 
## Option 3: Run Example Script
 
```bash
python3 example_usage.py
```
 
This analyzes example poems in `data/poems.txt` and shows complete results.
 
## Run Tests
 
### Workflow Test
```bash
python3 test_complete_workflow.py
```
 
**Expected Output:**
```
======================================================================
TEST: Complete Workflow
======================================================================
 
Test 1: Initialize analyzer...
  Analyzer loaded
 
Test 2: Read text file...
  6945 characters read from data/poems.txt
  10 poems found
 
Test 3: Analyze poem...
  Lines: 14
  Avg syllables: 15.6
  Aruz meter: ramal
  Confidence: medium
 
WORKFLOW SUCCESSFUL
```
 
## Troubleshooting
 
### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit
```
 
### "ModuleNotFoundError: No module named 'app2'"
Make sure you're in the correct directory:
```bash
cd /Users/moritzschlenstedt/tajik-poem-analyzer
```
 
### OCR not working
Install system dependencies:
 
**macOS:**
```bash
brew install poppler tesseract tesseract-lang
```
 
**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-fas
```
 
### "Lexicon not found"
Check if data files exist:
```bash
ls -lh data/
```
 
Should show:
```
tajik_lexicon.json    1.4M
tajik_corpus.txt      404M
poems.txt             12K
```
 
## Directory Structure
 
```
tajik-poem-analyzer/
├── ui.py                      # Web UI (START HERE!)
├── app2.py                    # Main analyzer
├── pdf_handler.py             # PDF/OCR integration
├── example_usage.py           # Example script
├── test_complete_workflow.py  # Test
├── data/
│   ├── tajik_lexicon.json     # 1.4MB - CRITICAL
│   ├── tajik_corpus.txt       # 404MB - CRITICAL
│   └── poems.txt              # Examples
├── README.md                  # Complete documentation
└── HOW_TO_RUN.md              # This file
```
 
## Important
 
### API Usage
```python
# CORRECT
from app2 import TajikPoemAnalyzer, AnalysisConfig
config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
analyzer = TajikPoemAnalyzer(config=config)
analysis = analyzer.analyze_poem("Poem as string")
 
# WRONG
from enhanced_tajik_analyzer import ...  # Wrong module
poem = PoemData(content="...")
analysis = analyzer.analyze_poem(poem)  # Expects string, not PoemData!
```
 
## Next Steps
 
1. **Start the UI**: `streamlit run ui.py`
2. **Upload a PDF** with Tajik/Persian poems
3. **Analyze** and view the results
4. For questions see README.md
