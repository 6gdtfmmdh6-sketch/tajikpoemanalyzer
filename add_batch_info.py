#!/usr/bin/env python3
"""
Batch-Zuordnungs-Skript für bestehende Contributions

Dieses Skript ergänzt die bestehenden Contribution-Dateien um:
- source_filename: Name der Quelldatei
- upload_batch_id: Eindeutige Batch-ID

NICHT-DESTRUKTIV: Bestehende Felder werden NICHT verändert, nur neue hinzugefügt.
"""

import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pfade
SCRIPT_DIR = Path(__file__).parent
CONTRIBUTIONS_DIR = SCRIPT_DIR / "tajik_corpus" / "contributions"

# Batch-Mapping basierend auf Timestamps in den Dateinamen
# Format: *_YYYYMMDD_HHMMSS.json
BATCH_MAPPING = {
    "030110": {
        "source_filename": "dilorom_soliboeva_tufonhoi_sokit.txt",
        "upload_batch_id": "batch_tufonhoi_sokit_2008",
        "volume_title": "Тӯфонҳои сокит",
        "volume_year": 2008
    },
    "030947": {
        "source_filename": "dilorom_bo anore khamhun.txt", 
        "upload_batch_id": "batch_bo_anore_khamhun_2019",
        "volume_title": "Бо аноре хамхун...",
        "volume_year": 2019
    },
    "030948": {  # Gleicher Batch wie 030947, nur Sekunden später
        "source_filename": "dilorom_bo anore khamhun.txt",
        "upload_batch_id": "batch_bo_anore_khamhun_2019",
        "volume_title": "Бо аноре хамхун...",
        "volume_year": 2019
    }
}


def extract_timestamp_suffix(filename: str) -> str:
    """Extrahiert den Timestamp-Suffix aus dem Dateinamen"""
    # Format: hash_YYYYMMDD_HHMMSS.json
    parts = filename.replace('.json', '').split('_')
    if len(parts) >= 3:
        return parts[-1]  # HHMMSS
    return ""


def add_batch_info_to_contribution(filepath: Path, dry_run: bool = False) -> dict:
    """
    Fügt Batch-Informationen zu einer Contribution hinzu.
    
    Returns:
        dict mit Status-Informationen
    """
    result = {
        "file": filepath.name,
        "action": None,
        "batch_id": None,
        "error": None
    }
    
    try:
        # Datei laden
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Prüfen ob bereits Batch-Info vorhanden
        if "source_filename" in data and "upload_batch_id" in data:
            result["action"] = "skipped (already has batch info)"
            result["batch_id"] = data.get("upload_batch_id")
            return result
        
        # Timestamp extrahieren
        timestamp_suffix = extract_timestamp_suffix(filepath.name)
        
        if timestamp_suffix not in BATCH_MAPPING:
            result["action"] = "skipped (unknown timestamp)"
            result["error"] = f"Unknown timestamp: {timestamp_suffix}"
            return result
        
        batch_info = BATCH_MAPPING[timestamp_suffix]
        
        # Neue Felder hinzufügen (NICHT bestehende überschreiben)
        fields_added = []
        
        if "source_filename" not in data:
            data["source_filename"] = batch_info["source_filename"]
            fields_added.append("source_filename")
        
        if "upload_batch_id" not in data:
            data["upload_batch_id"] = batch_info["upload_batch_id"]
            fields_added.append("upload_batch_id")
        
        # Volume-Metadaten in metadata ergänzen (falls nicht vorhanden)
        if "metadata" in data:
            if "volume_title" not in data["metadata"]:
                data["metadata"]["volume_title"] = batch_info["volume_title"]
                fields_added.append("metadata.volume_title")
            if "volume_year" not in data["metadata"]:
                data["metadata"]["volume_year"] = batch_info["volume_year"]
                fields_added.append("metadata.volume_year")
        
        # Änderungsprotokoll hinzufügen
        if "_batch_assignment" not in data:
            data["_batch_assignment"] = {
                "assigned_at": datetime.now().isoformat(),
                "fields_added": fields_added,
                "script_version": "1.0"
            }
        
        if not dry_run:
            # Datei speichern
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        result["action"] = "updated" if not dry_run else "would update"
        result["batch_id"] = batch_info["upload_batch_id"]
        result["fields_added"] = fields_added
        
    except Exception as e:
        result["action"] = "error"
        result["error"] = str(e)
    
    return result


def process_all_contributions(dry_run: bool = True):
    """
    Verarbeitet alle Contributions und fügt Batch-Informationen hinzu.
    
    Args:
        dry_run: Wenn True, werden keine Änderungen gespeichert (nur Vorschau)
    """
    if not CONTRIBUTIONS_DIR.exists():
        logger.error(f"Contributions-Ordner nicht gefunden: {CONTRIBUTIONS_DIR}")
        return
    
    json_files = list(CONTRIBUTIONS_DIR.glob("*.json"))
    logger.info(f"Gefunden: {len(json_files)} Contribution-Dateien")
    
    if dry_run:
        logger.info("=== DRY RUN - Keine Änderungen werden gespeichert ===")
    
    stats = {
        "updated": 0,
        "skipped": 0,
        "errors": 0
    }
    
    batch_counts = {}
    
    for json_file in sorted(json_files):
        result = add_batch_info_to_contribution(json_file, dry_run=dry_run)
        
        if "updated" in (result["action"] or ""):
            stats["updated"] += 1
            batch_id = result.get("batch_id", "unknown")
            batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1
            logger.info(f"  ✓ {result['file']} -> {batch_id}")
        elif "skipped" in (result["action"] or ""):
            stats["skipped"] += 1
            logger.debug(f"  - {result['file']}: {result['action']}")
        else:
            stats["errors"] += 1
            logger.error(f"  ✗ {result['file']}: {result.get('error', 'unknown error')}")
    
    # Zusammenfassung
    print()
    print("=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"  Aktualisiert: {stats['updated']}")
    print(f"  Übersprungen: {stats['skipped']}")
    print(f"  Fehler: {stats['errors']}")
    print()
    print("Batch-Verteilung:")
    for batch_id, count in sorted(batch_counts.items()):
        print(f"  {batch_id}: {count} Gedichte")
    print()
    
    if dry_run:
        print("Dies war ein DRY RUN. Um die Änderungen anzuwenden:")
        print("  python add_batch_info.py --apply")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("BATCH-ZUORDNUNG FÜR BESTEHENDE CONTRIBUTIONS")
    print("=" * 60)
    print()
    
    dry_run = "--apply" not in sys.argv
    
    if dry_run:
        print("Modus: DRY RUN (Vorschau ohne Änderungen)")
        print("Verwende --apply um Änderungen zu speichern")
    else:
        print("Modus: ANWENDEN (Änderungen werden gespeichert)")
        response = input("Fortfahren? (j/n): ")
        if response.lower() != 'j':
            print("Abgebrochen.")
            sys.exit(0)
    
    print()
    process_all_contributions(dry_run=dry_run)
