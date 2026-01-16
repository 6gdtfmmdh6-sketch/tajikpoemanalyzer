#!/usr/bin/env python3
"""
Configuration settings for Tajik Poetry Analyzer
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CORPORA_DIR = BASE_DIR / "tajik_corpora"
LIBRARY_DIR = BASE_DIR / "tajik_poetry_library"

# Analysis settings
LEXICON_PATH = DATA_DIR / "tajik_lexicon.json"
MIN_POEM_LENGTH = 50  # characters
MAX_POEM_LENGTH = 10000  # characters

# Corpus settings
CORPUS_TYPES = {
    "linguistic": {
        "description": "For OCR/KI training",
        "min_metadata": True,
        "store_raw_text": True
    },
    "literary": {
        "description": "For literary analysis",
        "min_metadata": False,
        "store_analysis": True
    }
}

# Literary periods for Tajik poetry
LITERARY_PERIODS = [
    {"name": "Classical (pre-1920)", "range": (900, 1920)},
    {"name": "Soviet Early (1920-1940)", "range": (1920, 1940)},
    {"name": "Soviet Mid (1940-1970)", "range": (1940, 1970)},
    {"name": "Soviet Late (1970-1991)", "range": (1970, 1991)},
    {"name": "Independence (1991-2000)", "range": (1991, 2000)},
    {"name": "Contemporary (2000-present)", "range": (2000, 2100)}
]

# Ensure directories exist
for directory in [DATA_DIR, CORPORA_DIR, LIBRARY_DIR]:
    directory.mkdir(exist_ok=True, parents=True)
