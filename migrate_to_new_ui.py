#!/usr/bin/env python3
"""
Migration script to move from old UI to new Streamlit app
"""

import json
import shutil
from pathlib import Path
import sys

def migrate_corpus_data():
    """Migrate existing corpus data to new structure"""
    
    # Paths
    old_corpus_path = Path("./tajik_corpus")
    new_corpus_path = Path("./tajik_corpora")
    
    if not old_corpus_path.exists():
        print("No existing corpus found to migrate.")
        return
    
    print("Migrating corpus data...")
    
    # Create new structure
    new_corpus_path.mkdir(exist_ok=True)
    
    # Migrate master corpus
    old_master = old_corpus_path / "corpus" / "master.json"
    if old_master.exists():
        # Create literary corpus directory
        literary_path = new_corpus_path / "literary"
        literary_path.mkdir(exist_ok=True)
        
        # Copy master corpus
        new_master = literary_path / "master.json"
        shutil.copy2(old_master, new_master)
        print(f"Migrated master corpus to {new_master}")
    
    # Migrate contributions
    old_contributions = old_corpus_path / "contributions"
    if old_contributions.exists():
        # Create linguistic corpus directory for raw texts
        linguistic_path = new_corpus_path / "linguistic"
        linguistic_path.mkdir(exist_ok=True)
        
        # Process each contribution
        for json_file in old_contributions.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract raw text
                raw_text = data.get('raw_text', '')
                if raw_text:
                    # Save as text file for linguistic corpus
                    text_file = linguistic_path / f"{json_file.stem}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(raw_text)
                    
                    # Keep JSON for literary corpus
                    json_dest = literary_path / json_file.name
                    shutil.copy2(json_file, json_dest)
                    
            except Exception as e:
                print(f"Error migrating {json_file}: {e}")
        
        print(f"Migrated {len(list(old_contributions.glob('*.json')))} contributions")
    
    print("Migration completed.")

def migrate_library_data():
    """Migrate existing library data to new structure"""
    
    old_library_path = Path("./tajik_poetry_library")
    
    if not old_library_path.exists():
        print("No existing library found to migrate.")
        return
    
    print("Note: Extended library data structure remains compatible.")
    print("No migration needed for library data.")

if __name__ == "__main__":
    print("Tajik Poetry Analyzer - Data Migration")
    print("=" * 50)
    
    migrate_corpus_data()
    migrate_library_data()
    
    print("\nMigration complete. You can now run the new Streamlit app.")
