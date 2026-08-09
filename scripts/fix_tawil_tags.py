#!/usr/bin/env python3
"""
Fix Tawil False Positives in Existing Contributions

This script updates the tags in existing contribution files to:
1. If a poem has both 'meter:ṭawīl' AND 'form:free_verse', change meter to free_verse
2. This fixes the historical misclassification issue
"""

import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_tawil_tags():
    """Fix Tawil false positives in all contribution files"""
    
    contrib_dir = Path("./tajik_corpus/contributions")
    
    if not contrib_dir.exists():
        logger.error(f"Directory not found: {contrib_dir}")
        return
    
    fixed_count = 0
    total_count = 0
    
    for file in sorted(contrib_dir.glob("*.json")):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_count += 1
            tags = data.get('tags', [])
            
            # Check if this needs fixing
            has_tawil = 'meter:ṭawīl' in tags
            has_free_verse = 'form:free_verse' in tags
            
            if has_tawil and has_free_verse:
                # Remove the ṭawīl meter tag
                tags.remove('meter:ṭawīl')
                
                # Add free_verse meter if not already there
                if 'meter:free_verse' not in tags:
                    tags.append('meter:free_verse')
                
                data['tags'] = tags
                
                # Add fix metadata
                data['_tawil_fix'] = {
                    'fixed_at': datetime.now().isoformat(),
                    'original_meter': 'ṭawīl',
                    'new_meter': 'free_verse',
                    'reason': 'form:free_verse was present, indicating misclassification'
                }
                
                # Save
                with open(file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                fixed_count += 1
                logger.info(f"Fixed: {file.name} - {data.get('title', 'Untitled')}")
            
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total files: {total_count}")
    logger.info(f"Fixed: {fixed_count}")
    logger.info(f"Unchanged: {total_count - fixed_count}")


if __name__ == "__main__":
    fix_tawil_tags()
