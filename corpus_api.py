#!/usr/bin/env python3
"""
Minimal API für Korpus-Zugriff vom Browser aus
"""
from flask import Flask, jsonify
from extended_corpus_manager import TajikLibraryManager
import json

app = Flask(__name__)
library = TajikLibraryManager()

@app.route('/api/corpus/stats')
def get_stats():
    return jsonify(library.get_statistics())

@app.route('/api/corpus/timeline')
def get_timeline():
    corpus = library.load_corpus()
    return jsonify(corpus.get("timeline", {}))

# Weitere Endpoints für Filterung, Suche, etc.
