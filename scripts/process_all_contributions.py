#!/usr/bin/env python3
"""
Batch-Processing: Fügt Batch-Informationen zu allen Contributions hinzu
und erstellt anschließend die bereinigten öffentlichen Versionen.

Dieses Skript wird einmalig ausgeführt.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))

import json
from pathlib import Path
from datetime import datetime
import shutil

# Pfade
SCRIPT_DIR = Path(__file__).parent
CONTRIBUTIONS_DIR = SCRIPT_DIR / "tajik_corpus" / "contributions"
CONTRIBUTIONS_PUBLIC = SCRIPT_DIR / "tajik_corpus" / "contributions_public"

# Batch-Mapping basierend auf Timestamps
BATCH_MAPPING = {
    "030110": {
        "source_filename": "dilorom_soliboeva_tufonhoi_sokit.txt",
        "upload_batch_id": "batch_tufonhoi_sokit_2008",
        "volume_title": "Тӯфонҳои сокит",
        "volume_year": 2008
    },
    "030947": {
        "source_filename": "dilorom_bo_anore_khamhun.txt", 
        "upload_batch_id": "batch_bo_anore_khamhun_2019",
        "volume_title": "Бо аноре хамхун...",
        "volume_year": 2019
    },
    "030948": {
        "source_filename": "dilorom_bo_anore_khamhun.txt",
        "upload_batch_id": "batch_bo_anore_khamhun_2019",
        "volume_title": "Бо аноре хамхун...",
        "volume_year": 2019
    }
}

# Felder für öffentliche Version entfernen
FIELDS_TO_REMOVE = ["raw_text", "normalized_text"]


def extract_timestamp_suffix(filename: str) -> str:
    """Extrahiert den Timestamp-Suffix aus dem Dateinamen"""
    parts = filename.replace('.json', '').split('_')
    if len(parts) >= 3:
        return parts[-1]
    return ""


def extract_first_line(text: str) -> str:
    """Extrahiert die erste Zeile nach dem Titel"""
    if not text:
        return ""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) > 1:
        return lines[1][:80]
    elif len(lines) == 1:
        return lines[0][:80]
    return ""


def process_contribution(filepath: Path) -> dict:
    """
    Verarbeitet eine Contribution:
    1. Fügt Batch-Info hinzu (Original)
    2. Erstellt bereinigte öffentliche Version
    """
    result = {
        "file": filepath.name,
        "batch_added": False,
        "public_created": False,
        "error": None
    }
    
    try:
        # Original laden
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_data = data.copy()
        
        # === SCHRITT 1: Batch-Info zum Original hinzufügen ===
        timestamp_suffix = extract_timestamp_suffix(filepath.name)
        
        if timestamp_suffix in BATCH_MAPPING:
            batch_info = BATCH_MAPPING[timestamp_suffix]
            fields_added = []
            
            # Neue Felder hinzufügen (nicht überschreiben)
            if "source_filename" not in data:
                data["source_filename"] = batch_info["source_filename"]
                fields_added.append("source_filename")
            
            if "upload_batch_id" not in data:
                data["upload_batch_id"] = batch_info["upload_batch_id"]
                fields_added.append("upload_batch_id")
            
            # Metadata ergänzen
            if "metadata" in data:
                if "volume_title" not in data["metadata"]:
                    data["metadata"]["volume_title"] = batch_info["volume_title"]
                    fields_added.append("metadata.volume_title")
                if "volume_year" not in data["metadata"]:
                    data["metadata"]["volume_year"] = batch_info["volume_year"]
                    fields_added.append("metadata.volume_year")
            
            # Änderungsprotokoll
            if fields_added:
                data["_batch_assignment"] = {
                    "assigned_at": datetime.now().isoformat(),
                    "fields_added": fields_added,
                    "script_version": "1.0"
                }
                
                # Original mit Batch-Info speichern
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                result["batch_added"] = True
                result["batch_id"] = batch_info["upload_batch_id"]
        
        # === SCHRITT 2: Öffentliche Version erstellen (ohne Volltexte) ===
        public_data = {}
        
        for key, value in data.items():
            if key not in FIELDS_TO_REMOVE:
                public_data[key] = value
        
        # First Line für Identifikation hinzufügen
        if "raw_text" in original_data and "first_line" not in public_data:
            public_data["first_line"] = extract_first_line(original_data.get("raw_text", ""))
        
        # Sanitization-Marker
        public_data["_sanitized"] = {
            "sanitized_at": datetime.now().isoformat(),
            "fields_removed": FIELDS_TO_REMOVE,
            "reason": "copyright_protection"
        }
        
        # Öffentliche Version speichern
        CONTRIBUTIONS_PUBLIC.mkdir(parents=True, exist_ok=True)
        public_path = CONTRIBUTIONS_PUBLIC / filepath.name
        
        with open(public_path, 'w', encoding='utf-8') as f:
            json.dump(public_data, f, ensure_ascii=False, indent=2)
        
        result["public_created"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    """Hauptfunktion"""
    print("=" * 70)
    print("BATCH-VERARBEITUNG: Batch-Info + Öffentliche Versionen")
    print("=" * 70)
    print()
    
    if not CONTRIBUTIONS_DIR.exists():
        print(f"FEHLER: Contributions-Ordner nicht gefunden: {CONTRIBUTIONS_DIR}")
        return
    
    json_files = list(CONTRIBUTIONS_DIR.glob("*.json"))
    print(f"Gefunden: {len(json_files)} Contribution-Dateien")
    print()
    
    stats = {
        "batch_added": 0,
        "public_created": 0,
        "errors": 0
    }
    
    batch_counts = {}
    
    for i, json_file in enumerate(sorted(json_files), 1):
        result = process_contribution(json_file)
        
        print(f"[{i:3}/{len(json_files)}] {result['file'][:40]:40}", end=" ")
        
        if result["error"]:
            print(f"❌ FEHLER: {result['error']}")
            stats["errors"] += 1
        else:
            status = []
            if result["batch_added"]:
                status.append("Batch✓")
                stats["batch_added"] += 1
                batch_id = result.get("batch_id", "unknown")
                batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1
            if result["public_created"]:
                status.append("Public✓")
                stats["public_created"] += 1
            print(" ".join(status) if status else "Übersprungen")
    
    # Zusammenfassung
    print()
    print("=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    print(f"  Batch-Info hinzugefügt: {stats['batch_added']}")
    print(f"  Öffentliche Versionen:  {stats['public_created']}")
    print(f"  Fehler:                 {stats['errors']}")
    print()
    print("Batch-Verteilung:")
    for batch_id, count in sorted(batch_counts.items()):
        print(f"  {batch_id}: {count} Gedichte")
    print()
    print(f"Öffentliche Dateien in: {CONTRIBUTIONS_PUBLIC}")
    print()
    print("Nächste Schritte:")
    print("  git add tajik_corpus/contributions_public/")
    print("  git commit -m 'Add sanitized contributions (copyright-safe)'")
    print("  git push")


if __name__ == "__main__":
    main()
