#!/usr/bin/env python3
"""
Export-Skript für urheberrechtskonforme GitHub-Veröffentlichung

Dieses Skript erstellt bereinigte Kopien der Contribution-Dateien:
- ENTFERNT: raw_text, normalized_text (urheberrechtlich geschützt)
- BEHÄLT: title, first_line, alle Analysen, Metadaten

Verwendung:
    python export_for_git.py

Die bereinigten Dateien werden in tajik_corpus/contributions_public/ gespeichert.
Die Originaldateien in tajik_corpus/contributions/ bleiben unverändert.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))

import json
import os
from pathlib import Path
from datetime import datetime
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pfade
SCRIPT_DIR = Path(__file__).parent.parent
CONTRIBUTIONS_LOCAL = SCRIPT_DIR / "tajik_corpus" / "contributions"
CONTRIBUTIONS_PUBLIC = SCRIPT_DIR / "tajik_corpus" / "contributions_public"

# Felder, die aus urheberrechtlichen Gründen entfernt werden
FIELDS_TO_REMOVE = ["raw_text", "normalized_text"]

# Felder, die hinzugefügt werden (für Identifikation)
def extract_first_line(text: str) -> str:
    """Extrahiert die erste nicht-leere Zeile (Incipit) für wissenschaftliche Identifikation"""
    if not text:
        return ""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Überspringe Titel (erste Zeile), nimm zweite
    if len(lines) > 1:
        return lines[1][:80]  # Max 80 Zeichen
    elif len(lines) == 1:
        return lines[0][:80]
    return ""


def sanitize_contribution(data: dict) -> dict:
    """
    Bereinigt eine Contribution für die öffentliche Veröffentlichung.
    
    - Entfernt urheberrechtlich geschützte Volltexte
    - Behält alle Analysen und Metadaten
    - Fügt first_line für wissenschaftliche Identifikation hinzu
    """
    # Kopie erstellen (nicht das Original verändern)
    sanitized = {}
    
    for key, value in data.items():
        if key in FIELDS_TO_REMOVE:
            # Diese Felder werden nicht kopiert
            continue
        else:
            sanitized[key] = value
    
    # First Line hinzufügen (aus raw_text extrahieren, falls vorhanden)
    if "raw_text" in data and "first_line" not in sanitized:
        sanitized["first_line"] = extract_first_line(data.get("raw_text", ""))
    
    # Markierung hinzufügen
    sanitized["_sanitized"] = {
        "sanitized_at": datetime.now().isoformat(),
        "fields_removed": FIELDS_TO_REMOVE,
        "reason": "copyright_protection"
    }
    
    return sanitized


def export_contributions():
    """Exportiert alle Contributions in den öffentlichen Ordner"""
    
    if not CONTRIBUTIONS_LOCAL.exists():
        logger.error(f"Lokaler Contributions-Ordner nicht gefunden: {CONTRIBUTIONS_LOCAL}")
        return False
    
    # Öffentlichen Ordner erstellen falls nötig
    CONTRIBUTIONS_PUBLIC.mkdir(parents=True, exist_ok=True)
    
    # Statistiken
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0
    }
    
    # Alle JSON-Dateien verarbeiten
    json_files = list(CONTRIBUTIONS_LOCAL.glob("*.json"))
    logger.info(f"Gefunden: {len(json_files)} Contribution-Dateien")
    
    for json_file in json_files:
        try:
            # Original laden
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Bereinigen
            sanitized = sanitize_contribution(data)
            
            # In öffentlichen Ordner speichern
            output_path = CONTRIBUTIONS_PUBLIC / json_file.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized, f, ensure_ascii=False, indent=2)
            
            stats["processed"] += 1
            
        except Exception as e:
            logger.error(f"Fehler bei {json_file.name}: {e}")
            stats["errors"] += 1
    
    # Zusammenfassung
    logger.info("=" * 60)
    logger.info("EXPORT ABGESCHLOSSEN")
    logger.info(f"  Verarbeitet: {stats['processed']}")
    logger.info(f"  Fehler: {stats['errors']}")
    logger.info(f"  Zielordner: {CONTRIBUTIONS_PUBLIC}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Die bereinigten Dateien können jetzt gepusht werden:")
    logger.info("  git add tajik_corpus/contributions_public/")
    logger.info("  git commit -m 'Add sanitized contributions (no full texts)'")
    logger.info("  git push")
    
    return stats["errors"] == 0


def verify_sanitization():
    """Überprüft, ob alle öffentlichen Dateien korrekt bereinigt wurden"""
    
    if not CONTRIBUTIONS_PUBLIC.exists():
        logger.error("Öffentlicher Ordner existiert nicht. Führe zuerst export_contributions() aus.")
        return False
    
    issues = []
    
    for json_file in CONTRIBUTIONS_PUBLIC.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for field in FIELDS_TO_REMOVE:
            if field in data:
                issues.append(f"{json_file.name}: enthält noch '{field}'")
    
    if issues:
        logger.error("PROBLEME GEFUNDEN:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("✓ Alle öffentlichen Dateien sind korrekt bereinigt")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("TAJIK POETRY ANALYZER - Export für GitHub")
    print("=" * 60)
    print()
    print("Dieses Skript erstellt urheberrechtskonforme Kopien der")
    print("Contribution-Dateien (ohne Volltexte).")
    print()
    
    success = export_contributions()
    
    if success:
        print()
        verify_sanitization()
