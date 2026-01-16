#!/usr/bin/env python3
"""
Tajik Poetry Corpus Manager
Git-based corpus manager for collaborative poetry analysis contributions

Features:
- Local contribution storage
- Git export preparation
- Deduplication via content hashing
- Corpus statistics
- License management
"""

import json
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)


class TajikCorpusManager:
    """
    Git-based corpus manager for Tajik poetry analysis.
    Enables decentralized contributions and synchronization.
    """
    
    def __init__(self, local_repo_path: str = "./tajik_corpus"):
        self.local_repo = Path(local_repo_path)
        self.contributions_dir = self.local_repo / "contributions"
        self.master_corpus = self.local_repo / "corpus" / "master.json"
        self.initialize_structure()
        
        # GitHub API Endpoint (can be configured later)
        self.remote_url = "https://api.github.com/repos/username/tajik-poetry-corpus"
        
    def initialize_structure(self):
        """Initialize local corpus structure"""
        if not self.local_repo.exists():
            self.local_repo.mkdir(parents=True)
            self.contributions_dir.mkdir()
            (self.local_repo / "corpus").mkdir()
            (self.local_repo / "exports").mkdir()
            self.create_initial_corpus()
            logger.info(f"Corpus structure initialized at {self.local_repo}")
            
    def create_initial_corpus(self):
        """Create initial corpus schema"""
        corpus_schema = {
            "metadata": {
                "version": "1.0.0",
                "created": datetime.now().isoformat(),
                "license": "CC-BY-NC-SA-4.0",
                "language": "tg",
                "script": "Cyrillic",
                "description": "Tajik Poetry Corpus - Scientific Research Collection"
            },
            "statistics": {
                "total_poems": 0,
                "total_lines": 0,
                "total_words": 0,
                "unique_words": 0,
                "contributors": 0
            },
            "contributors": {},
            "poems": [],
            "vocabulary": {},
            "phoneme_inventory": {},
            "aruz_distribution": {},
            "theme_distribution": {},
            "radif_collection": []  # NEW: Track poems with Radīf
        }
        
        self.save_corpus(corpus_schema)
        
    def prepare_contribution(self, analysis_result: Dict, raw_text: str, 
                           user_info: Optional[Dict] = None) -> Dict:
        """
        Prepare a contribution for the corpus.
        
        Args:
            analysis_result: Complete analysis from TajikPoemAnalyzer
            raw_text: Raw poem text
            user_info: Optional user information (Git username, email)
            
        Returns:
            Contribution dictionary for local storage
        """
        poem_id = analysis_result.get("poem_id", f"poem_{int(datetime.now().timestamp())}")
        
        # Calculate hash for deduplication
        content_hash = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()[:16]
        
        # Serialize analysis objects
        analysis_data = self._serialize_analysis_result(analysis_result)
        
        contribution = {
            "contribution_id": f"{content_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "poem_id": poem_id,
            "title": analysis_result.get("title", "Untitled"),
            "metadata": {
                "submission_date": datetime.now().isoformat(),
                "contributor": user_info or {"anonymous": True},
                "software_version": "2.1.0",
                "content_hash": content_hash,
                "source_language": "tg",
                "license_accepted": True
            },
            "raw_text": raw_text,
            "normalized_text": self.normalize_text(raw_text),
            "analysis": analysis_data,
            "tags": self._extract_tags(analysis_result)
        }
        
        return contribution
        
    def _serialize_analysis_result(self, analysis_result: Dict) -> Dict:
        """Serialize analysis objects for JSON storage"""
        analysis = analysis_result.get("analysis")
        validation = analysis_result.get("validation", {})
        
        if analysis is None:
            return {"error": "No analysis data"}
        
        # Handle dataclass objects
        def serialize_obj(obj):
            if is_dataclass(obj):
                return asdict(obj)
            elif hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if is_dataclass(value):
                        result[key] = asdict(value)
                    elif isinstance(value, list):
                        result[key] = [serialize_obj(item) for item in value]
                    elif hasattr(value, 'value'):  # Enum
                        result[key] = value.value
                    else:
                        result[key] = value
                return result
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            return obj
        
        serialized = {
            "structural": serialize_obj(analysis.structural),
            "content": serialize_obj(analysis.content),
            "literary": serialize_obj(analysis.literary),
            "quality_metrics": validation
        }
        
        return serialized
        
    def save_contribution(self, contribution: Dict) -> Path:
        """
        Save contribution locally in contributions directory.
        
        Returns:
            Path to saved contribution file
        """
        # Individual contribution file
        filename = f"{contribution['contribution_id']}.json"
        filepath = self.contributions_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(contribution, f, ensure_ascii=False, indent=2, default=str)
            
        # Also merge to master corpus
        self._merge_to_local_corpus(contribution)
        
        logger.info(f"Contribution saved: {filename}")
        return filepath
        
    def _merge_to_local_corpus(self, contribution: Dict):
        """Merge contribution into local corpus"""
        corpus = self.load_corpus()
        
        # Check for duplicates
        content_hash = contribution["metadata"]["content_hash"]
        existing_hashes = [p.get("metadata", {}).get("content_hash", "") 
                          for p in corpus.get("poems", [])]
        
        if content_hash not in existing_hashes:
            # Extract analysis data
            analysis = contribution.get("analysis", {})
            structural = analysis.get("structural", {})
            content_analysis = analysis.get("content", {})
            
            # Add poem entry
            poem_entry = {
                "id": contribution["poem_id"],
                "title": contribution.get("title", ""),
                "metadata": contribution["metadata"],
                "text": contribution["normalized_text"],
                "analysis_summary": {
                    "meter": structural.get("aruz_analysis", {}).get("identified_meter", "unknown"),
                    "lines": structural.get("lines", 0),
                    "words": content_analysis.get("total_words", 0),
                    "stanza_form": structural.get("stanza_structure", ""),
                    "rhyme_pattern": structural.get("rhyme_pattern", ""),
                    "has_radif": self._check_for_radif(structural)
                }
            }
            
            corpus.setdefault("poems", []).append(poem_entry)
            
            # Update statistics
            stats = corpus.setdefault("statistics", {})
            stats["total_poems"] = stats.get("total_poems", 0) + 1
            stats["total_lines"] = stats.get("total_lines", 0) + structural.get("lines", 0)
            stats["total_words"] = stats.get("total_words", 0) + content_analysis.get("total_words", 0)
            
            # Update unique words
            if "normalized_text" in contribution:
                words = set(re.findall(r'[\wӣӯ]+', contribution["normalized_text"].lower()))
                existing_vocab = set(corpus.get("vocabulary", {}).keys())
                new_words = words - existing_vocab
                stats["unique_words"] = stats.get("unique_words", 0) + len(new_words)
                
                # Add to vocabulary
                vocab = corpus.setdefault("vocabulary", {})
                for word in words:
                    vocab[word] = vocab.get(word, 0) + 1
            
            # Update ʿArūḍ distribution
            meter = structural.get("aruz_analysis", {}).get("identified_meter", "unknown")
            aruz_dist = corpus.setdefault("aruz_distribution", {})
            aruz_dist[meter] = aruz_dist.get(meter, 0) + 1
            
            # Update theme distribution
            themes = content_analysis.get("theme_distribution", {})
            theme_dist = corpus.setdefault("theme_distribution", {})
            for theme, count in themes.items():
                if count > 0:
                    theme_dist[theme] = theme_dist.get(theme, 0) + count
            
            # Track Radīf poems
            if self._check_for_radif(structural):
                radif_list = corpus.setdefault("radif_collection", [])
                radif_list.append({
                    "poem_id": contribution["poem_id"],
                    "radif": self._get_radif_text(structural)
                })
            
            # Update contributor list
            contributor = contribution["metadata"]["contributor"]
            if contributor.get("username"):
                contributors = corpus.setdefault("contributors", {})
                contributors[contributor["username"]] = contributors.get(
                    contributor["username"], 0) + 1
                stats["contributors"] = len(contributors)
                
            self.save_corpus(corpus)
            logger.info(f"Poem {contribution['poem_id']} merged to corpus")
    
    def _check_for_radif(self, structural: Dict) -> bool:
        """Check if poem has a global Radīf"""
        rhyme_scheme = structural.get("rhyme_scheme", [])
        if not rhyme_scheme:
            return False
        
        radif_values = [r.get("radif", "") for r in rhyme_scheme if r.get("radif")]
        return len(radif_values) > 0 and len(set(radif_values)) == 1
    
    def _get_radif_text(self, structural: Dict) -> str:
        """Extract Radīf text from structural analysis"""
        rhyme_scheme = structural.get("rhyme_scheme", [])
        for r in rhyme_scheme:
            if r.get("radif"):
                return r["radif"]
        return ""
            
    def export_contributions_for_git(self) -> Path:
        """
        Export all contributions for Git push.
        
        Returns:
            Path to exported file
        """
        # Collect all contributions
        all_contributions = []
        
        for file in self.contributions_dir.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                all_contributions.append(json.load(f))
                
        # Create export file
        export_data = {
            "export_version": "1.0",
            "export_date": datetime.now().isoformat(),
            "total_contributions": len(all_contributions),
            "contributions": all_contributions
        }
        
        export_dir = self.local_repo / "exports"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"contributions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
        # Create README for Git
        readme_path = export_dir / "README.md"
        readme_content = f"""# Tajik Poetry Corpus - Contributions

## Export from {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Statistics
- Total Contributions: {len(all_contributions)}
- Contained Poems: {len(all_contributions)}

### Instructions for Repository Maintainers
1. Copy this file to the main corpus repository
2. Review and validate contributions
3. Integrate into master corpus
4. Update statistics

### License
All contributions are under the CC-BY-NC-SA 4.0 license.
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
            
        logger.info(f"Export prepared: {export_path}")
        return export_path
        
    def generate_git_commands(self) -> str:
        """Generate Git commands for pushing contributions"""
        export_dir = self.local_repo / "exports"
        
        try:
            latest_export = max(export_dir.glob("*.json"), 
                              key=lambda x: x.stat().st_mtime, 
                              default=None)
        except ValueError:
            return "No contributions found for export."
            
        if not latest_export:
            return "No contributions found for export."
            
        commands = f"""
# 1. Navigate to corpus repository
cd /path/to/tajik-poetry-corpus

# 2. Create new branch
git checkout -b contributions-{datetime.now().strftime('%Y%m%d')}

# 3. Copy exported file
cp "{latest_export}" ./contributions/

# 4. Copy README
cp "{export_dir / 'README.md'}" ./

# 5. Add and commit changes
git add contributions/ README.md
git commit -m "New poetry contributions from {datetime.now().strftime('%Y-%m-%d')}"

# 6. Push to GitHub
git push origin contributions-{datetime.now().strftime('%Y%m%d')}

# 7. Create Pull Request on GitHub
# Visit: https://github.com/username/tajik-poetry-corpus/pulls
"""
        return commands
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent storage"""
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Standardize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Reduce multiple empty lines
        lines = []
        for line in text.split('\n'):
            if line.strip() or (lines and lines[-1].strip()):
                lines.append(line)
        
        return '\n'.join(lines)
        
    def _extract_tags(self, analysis_result: Dict) -> List[str]:
        """Extract tags for search functionality"""
        tags = []
        analysis = analysis_result.get("analysis")
        
        if analysis is None:
            return tags
        
        # Meter tags
        try:
            meter = analysis.structural.aruz_analysis.identified_meter
            if meter and meter != "unknown":
                tags.append(f"meter:{meter}")
        except AttributeError:
            pass
            
        # Form tags
        try:
            form = analysis.structural.stanza_structure
            if form:
                tags.append(f"form:{form}")
        except AttributeError:
            pass
            
        # Theme tags
        try:
            for theme, count in analysis.content.theme_distribution.items():
                if count > 0:
                    tags.append(f"theme:{theme.lower()}")
        except AttributeError:
            pass
                
        # Register tag
        try:
            register = analysis.content.stylistic_register
            if register:
                tags.append(f"register:{register}")
        except AttributeError:
            pass
        
        # Free verse tag
        try:
            if hasattr(analysis.structural, 'is_free_verse') and analysis.structural.is_free_verse:
                tags.append("form:free_verse")
        except AttributeError:
            pass
        
        # Radīf tag
        try:
            rhyme_scheme = analysis.structural.rhyme_scheme
            radif_values = [r.radif for r in rhyme_scheme if r.radif]
            if radif_values and len(set(radif_values)) == 1:
                tags.append(f"radif:{radif_values[0]}")
        except AttributeError:
            pass
        
        return tags
        
    def load_corpus(self) -> Dict:
        """Load the local corpus"""
        if self.master_corpus.exists():
            with open(self.master_corpus, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._get_empty_corpus()
    
    def _get_empty_corpus(self) -> Dict:
        """Return empty corpus structure"""
        return {
            "metadata": {},
            "statistics": {
                "total_poems": 0,
                "total_lines": 0,
                "total_words": 0,
                "unique_words": 0,
                "contributors": 0
            },
            "contributors": {},
            "poems": [],
            "vocabulary": {},
            "aruz_distribution": {},
            "theme_distribution": {},
            "radif_collection": []
        }
        
    def save_corpus(self, corpus: Dict):
        """Save the corpus"""
        # Ensure directory exists
        self.master_corpus.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.master_corpus, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2, default=str)
            
    def get_corpus_statistics(self) -> Dict:
        """Get current corpus statistics"""
        corpus = self.load_corpus()
        stats = corpus.get("statistics", {})
        
        # Add calculated fields
        total_words = stats.get("total_words", 0)
        unique_words = stats.get("unique_words", 0)
        
        stats["lexical_diversity"] = (unique_words / total_words * 100) if total_words > 0 else 0
        stats["aruz_distribution"] = corpus.get("aruz_distribution", {})
        stats["theme_distribution"] = corpus.get("theme_distribution", {})
        stats["radif_count"] = len(corpus.get("radif_collection", []))
        
        return stats
    
    def search_poems(self, query: str = None, tags: List[str] = None) -> List[Dict]:
        """Search poems in corpus by text or tags"""
        corpus = self.load_corpus()
        poems = corpus.get("poems", [])
        results = []
        
        for poem in poems:
            match = True
            
            # Text search
            if query:
                text = poem.get("text", "").lower()
                title = poem.get("title", "").lower()
                if query.lower() not in text and query.lower() not in title:
                    match = False
            
            # Tag search (would need to store tags in poem entry)
            # For now, search in analysis_summary
            if tags and match:
                summary = poem.get("analysis_summary", {})
                for tag in tags:
                    if ":" in tag:
                        key, value = tag.split(":", 1)
                        if key == "meter" and summary.get("meter") != value:
                            match = False
                        elif key == "form" and summary.get("stanza_form") != value:
                            match = False
            
            if match:
                results.append(poem)
        
        return results


# Convenience function for quick corpus access
def get_corpus_manager(path: str = "./tajik_corpus") -> TajikCorpusManager:
    """Get or create a corpus manager instance"""
    return TajikCorpusManager(local_repo_path=path)
