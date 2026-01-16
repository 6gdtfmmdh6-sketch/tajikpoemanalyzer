# Daten-Dateien

## Kritische Dateien

Das Projekt benötigt zwei große Daten-Dateien, die für die Analyse essentiell sind:

### 1. tajik_lexicon.json (1.4 MB)
**Zweck:** Wörterbuch tadschikischer Wörter für Wort-Analyse
- Neologismus-Erkennung
- Archaismus-Erkennung
- Wortstamm-Analyse

**Status:** ✅ Kann in Git hochgeladen werden (1.4 MB)

### 2. tajik_corpus.txt (404 MB)
**Zweck:** Großer Korpus tadschikischer Texte für statistische Analyse
- Häufigkeitsanalyse
- Kollokations-Erkennung
- Korpus-basierte Validierung

**Status:** ⚠️ ZU GROSS für Git (404 MB)

## Optionen für tajik_corpus.txt

### Option 1: Git LFS (Empfohlen)
```bash
# Git LFS installieren
brew install git-lfs  # macOS
# sudo apt-get install git-lfs  # Linux

# Git LFS aktivieren
git lfs install

# Korpus-Datei mit LFS tracken
git lfs track "data/tajik_corpus.txt"
git add .gitattributes
git add data/tajik_corpus.txt
git commit -m "Add corpus with Git LFS"
```

### Option 2: Externe Speicherung
Lade `tajik_corpus.txt` auf einen Cloud-Service hoch:
- Google Drive
- Dropbox
- OneDrive

Erstelle dann eine `data/DOWNLOAD_CORPUS.md` mit Download-Link.

### Option 3: Ohne Korpus arbeiten
Der Analyzer funktioniert auch ohne Korpus, aber mit eingeschränkten Features:
- ✅ Aruz-Metrik-Analyse funktioniert
- ✅ Reimschema-Erkennung funktioniert
- ✅ Strukturanalyse funktioniert
- ⚠️ Häufigkeits-Scores nicht verfügbar

## Für deinen Freund

Wenn dein Freund das Projekt klont:

**Mit Korpus (Git LFS):**
```bash
git clone <dein-repo-url>
cd tajik-poem-analyzer
git lfs pull  # Lädt große Dateien herunter
pip install -r requirements.txt
```

**Ohne Korpus:**
```bash
git clone <dein-repo-url>
cd tajik-poem-analyzer
# Korpus manuell herunterladen von [Link]
# Korpus nach data/tajik_corpus.txt kopieren
pip install -r requirements.txt
```

## Aktueller Status

- ✅ `data/tajik_lexicon.json` (1.4 MB) - Kann direkt commitet werden
- ⚠️ `data/tajik_corpus.txt` (404 MB) - Entscheide dich für eine Option oben
- ✅ `data/poems.txt` (12 KB) - Beispieldateien, können commitet werden
