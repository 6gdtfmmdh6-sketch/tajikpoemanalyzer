#!/usr/bin/env python3
"""
Extended Corpus Manager for Tajik Poetry Library
Scientific library system with chronological analysis
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import Counter
import re
import unicodedata

logger = logging.getLogger(__name__)


class Period(Enum):
    """Historical periods of Tajik literature"""
    CLASSICAL = "Classical (pre-1920)"
    SOVIET_EARLY = "Early Soviet (1920-1940)"
    SOVIET_MID = "Mid Soviet (1940-1970)"
    SOVIET_LATE = "Late Soviet (1970-1991)"
    INDEPENDENCE = "Independence (1991-2000)"
    CONTEMPORARY = "Contemporary (2000-present)"


class Genre(Enum):
    """Literary genres"""
    GHAZAL = "Ghazal"
    QASIDA = "Qasida"
    RUBAIYAT = "Rubaiyat"
    MASNAVI = "Masnavi"
    FREE_VERSE = "Free Verse"
    PROSE_POEM = "Prose Poem"
    MODERNIST = "Modernist"
    FOLK = "Folk Poetry"
    LYRIC = "Lyric"
    EPIC = "Epic"
    SATIRE = "Satire"
    RELIGIOUS = "Religious"


@dataclass
class VolumeMetadata:
    """Metadata for a poetry volume"""
    # REQUIRED FIELDS (no default values)
    author_name: str
    volume_title: str
    publication_year: int
    
    # OPTIONAL FIELDS (with default values)
    author_birth_year: Optional[int] = None
    author_death_year: Optional[int] = None
    publisher: Optional[str] = None
    city: Optional[str] = None
    original_language: str = "tg"
    script: str = "Cyrillic"
    period: Optional[Period] = None
    genres: List[Genre] = field(default_factory=list)
    isbn: Optional[str] = None
    pages: Optional[int] = None
    translator: Optional[str] = None
    edition: Optional[str] = None
    source_type: str = "printed"
    notes: Optional[str] = None
    
    def __post_init__(self):
        if not self.period and self.publication_year:
            self.period = self._infer_period()
    
    def _infer_period(self) -> Period:
        """Infer historical period from publication year"""
        if self.publication_year < 1920:
            return Period.CLASSICAL
        elif 1920 <= self.publication_year < 1940:
            return Period.SOVIET_EARLY
        elif 1940 <= self.publication_year < 1970:
            return Period.SOVIET_MID
        elif 1970 <= self.publication_year < 1991:
            return Period.SOVIET_LATE
        elif 1991 <= self.publication_year < 2000:
            return Period.INDEPENDENCE
        else:
            return Period.CONTEMPORARY
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['period'] = self.period.value if self.period else None
        data['genres'] = [g.value for g in self.genres]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VolumeMetadata':
        """Create from dictionary"""
        # Extract required fields first
        required_fields = {
            'author_name': data['author_name'],
            'volume_title': data['volume_title'],
            'publication_year': data['publication_year']
        }
        
        # Handle period
        period_str = data.get('period')
        period = None
        if period_str:
            for p in Period:
                if p.value == period_str:
                    period = p
                    break
        
        # Handle genres
        genres = []
        for genre_str in data.get('genres', []):
            for g in Genre:
                if g.value == genre_str:
                    genres.append(g)
                    break
        
        # Create instance with required fields
        instance = cls(**required_fields)
        
        # Set optional fields
        for field_name in ['author_birth_year', 'author_death_year', 'publisher', 
                          'city', 'original_language', 'script', 'isbn', 'pages', 
                          'translator', 'edition', 'source_type', 'notes']:
            if field_name in data:
                setattr(instance, field_name, data[field_name])
        
        instance.period = period
        instance.genres = genres
        
        return instance


class TajikLibraryManager:
    """
    Enhanced corpus manager with library functions for chronological analysis
    """
    
    def __init__(self, library_path: str = "./tajik_poetry_library"):
        self.library_path = Path(library_path)
        self.volumes_dir = self.library_path / "volumes"
        self.authors_dir = self.library_path / "authors"
        self.poems_dir = self.library_path / "poems"
        self.corpus_file = self.library_path / "corpus.json"
        self.stats_file = self.library_path / "statistics.json"
        self.initialize_library()
    
    def initialize_library(self):
        """Initialize library directory structure"""
        directories = [
            self.library_path,
            self.volumes_dir,
            self.authors_dir,
            self.poems_dir,
            self.library_path / "exports",
            self.library_path / "analysis",
            self.library_path / "timeline"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize corpus if not exists
        if not self.corpus_file.exists():
            self.create_initial_corpus()
    
    def create_initial_corpus(self):
        """Create initial corpus structure"""
        corpus = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "2.0.0",
                "description": "Tajik Poetry Digital Library",
                "license": "CC-BY-NC-SA-4.0",
                "total_volumes": 0,
                "total_poems": 0
            },
            "timeline": {
                "by_year": {},
                "by_period": {},
                "by_decade": {}
            },
            "authors": {},
            "volumes": [],
            "poems": [],
            "statistics": self._empty_statistics()
        }
        
        self.save_corpus(corpus)
    
    def _empty_statistics(self) -> Dict:
        """Create empty statistics structure"""
        return {
            "total_volumes": 0,
            "total_poems": 0,
            "total_lines": 0,
            "total_words": 0,
            "unique_words": 0,
            "authors_count": 0,
            "publication_years": {
                "min": None,
                "max": None,
                "distribution": {}
            },
            "genre_distribution": {},
            "period_distribution": {},
            "meter_distribution": {},
            "theme_distribution": {},
            "lexical_diversity_by_period": {},
            "stylistic_evolution": {}
        }
    
    def register_volume(self, volume_metadata: VolumeMetadata, poems_data: List[Dict]) -> str:
        """
        Register a complete volume with all its poems
        
        Args:
            volume_metadata: Metadata about the volume
            poems_data: List of poem analysis results
            
        Returns:
            Volume ID
        """
        # Generate volume ID
        author_slug = self._slugify(volume_metadata.author_name)
        title_slug = self._slugify(volume_metadata.volume_title)
        volume_id = f"{author_slug}_{title_slug}_{volume_metadata.publication_year}"
        
        # Create volume record
        volume_record = {
            "volume_id": volume_id,
            "metadata": volume_metadata.to_dict(),
            "poems_count": len(poems_data),
            "registered_date": datetime.now().isoformat(),
            "poem_ids": []
        }
        
        # Register each poem
        poem_ids = []
        for i, poem_data in enumerate(poems_data):
            poem_id = self._register_poem(poem_data, volume_id, volume_metadata, i)
            poem_ids.append(poem_id)
        
        volume_record["poem_ids"] = poem_ids
        
        # Update corpus
        corpus = self.load_corpus()
        
        # Add volume
        corpus["volumes"].append(volume_record)
        corpus["metadata"]["total_volumes"] = len(corpus["volumes"])
        corpus["metadata"]["total_poems"] += len(poems_data)
        
        # Update author record
        self._update_author_record(corpus, volume_metadata)
        
        # Update timeline
        self._update_timeline(corpus, volume_metadata, poems_data)
        
        # Update statistics
        self._update_statistics(corpus, volume_metadata, poems_data)
        
        # Save corpus
        self.save_corpus(corpus)
        
        # Save volume-specific file
        self._save_volume_file(volume_id, volume_record, poems_data)
        
        logger.info(f"Volume registered: {volume_id}")
        return volume_id
    
    def _register_poem(self, poem_data: Dict, volume_id: str, 
                      volume_metadata: VolumeMetadata, index: int) -> str:
        """Register a single poem"""
        # Generate poem ID
        poem_hash = hashlib.sha256(
            poem_data.get("content", "").encode('utf-8')
        ).hexdigest()[:12]
        
        poem_id = f"{volume_id}_poem_{index:03d}_{poem_hash}"
        
        # Extract analysis data
        analysis = poem_data.get("analysis", {})
        
        # Handle different analysis formats
        if hasattr(analysis, 'structural'):
            # EnhancedComprehensiveAnalysis object
            structural = analysis.structural
            content = analysis.content
            meter = structural.aruz_analysis.identified_meter if hasattr(structural.aruz_analysis, 'identified_meter') else "unknown"
            lines = structural.lines if hasattr(structural, 'lines') else 0
            words = content.total_words if hasattr(content, 'total_words') else 0
            lexical_diversity = content.lexical_diversity if hasattr(content, 'lexical_diversity') else 0
            is_free_verse = getattr(structural, 'is_free_verse', False) if hasattr(structural, 'is_free_verse') else False
            stanza_form = getattr(structural, 'stanza_structure', 'unknown')
            rhyme_pattern = getattr(structural, 'rhyme_pattern', '')
            themes = getattr(content, 'theme_distribution', {})
        elif isinstance(analysis, dict):
            # Dictionary format
            structural = analysis.get("structural", {})
            content = analysis.get("content", {})
            meter = structural.get("aruz_analysis", {}).get("identified_meter", "unknown")
            lines = structural.get("lines", 0)
            words = content.get("total_words", 0)
            lexical_diversity = content.get("lexical_diversity", 0)
            is_free_verse = structural.get("is_free_verse", False)
            stanza_form = structural.get("stanza_structure", "unknown")
            rhyme_pattern = structural.get("rhyme_pattern", "")
            themes = content.get("theme_distribution", {})
        else:
            # Unknown format
            meter = "unknown"
            lines = 0
            words = 0
            lexical_diversity = 0
            is_free_verse = False
            stanza_form = "unknown"
            rhyme_pattern = ""
            themes = {}
        
        # Extract themes list
        theme_list = [k for k, v in themes.items() if v > 0] if isinstance(themes, dict) else []
        
        poem_record = {
            "poem_id": poem_id,
            "volume_id": volume_id,
            "metadata": {
                "author": volume_metadata.author_name,
                "publication_year": volume_metadata.publication_year,
                "period": volume_metadata.period.value if volume_metadata.period else None,
                "genres": [g.value for g in volume_metadata.genres],
                "registration_date": datetime.now().isoformat()
            },
            "analysis_summary": {
                "meter": meter,
                "lines": lines,
                "words": words,
                "lexical_diversity": lexical_diversity,
                "is_free_verse": is_free_verse,
                "stanza_form": stanza_form,
                "rhyme_pattern": rhyme_pattern,
                "themes": theme_list
            },
            "text": {
                "original": poem_data.get("content", ""),
                "normalized": self._normalize_text(poem_data.get("content", ""))
            }
        }
        
        # Save poem record
        poem_file = self.poems_dir / f"{poem_id}.json"
        with open(poem_file, 'w', encoding='utf-8') as f:
            json.dump(poem_record, f, ensure_ascii=False, indent=2)
        
        # Update corpus poems list
        corpus = self.load_corpus()
        corpus["poems"].append({
            "poem_id": poem_id,
            "volume_id": volume_id,
            "title": f"Poem {index + 1} from {volume_metadata.volume_title}",
            "author": volume_metadata.author_name,
            "year": volume_metadata.publication_year
        })
        self.save_corpus(corpus)
        
        return poem_id
    
    def _update_author_record(self, corpus: Dict, volume_metadata: VolumeMetadata):
        """Update or create author record"""
        author_name = volume_metadata.author_name
        
        if author_name not in corpus["authors"]:
            corpus["authors"][author_name] = {
                "name": author_name,
                "birth_year": volume_metadata.author_birth_year,
                "death_year": volume_metadata.author_death_year,
                "volumes": [],
                "total_poems": 0,
                "periods": set(),
                "genres": set(),
                "first_publication": volume_metadata.publication_year,
                "last_publication": volume_metadata.publication_year
            }
        
        author = corpus["authors"][author_name]
        author["volumes"].append({
            "title": volume_metadata.volume_title,
            "year": volume_metadata.publication_year,
            "publisher": volume_metadata.publisher,
            "volume_id": f"{self._slugify(author_name)}_{self._slugify(volume_metadata.volume_title)}_{volume_metadata.publication_year}"
        })
        
        if volume_metadata.period:
            author["periods"].add(volume_metadata.period.value)
        
        for genre in volume_metadata.genres:
            author["genres"].add(genre.value)
        
        author["total_poems"] += corpus["metadata"]["total_poems"]
        
        # Update publication range
        if volume_metadata.publication_year < author["first_publication"]:
            author["first_publication"] = volume_metadata.publication_year
        if volume_metadata.publication_year > author["last_publication"]:
            author["last_publication"] = volume_metadata.publication_year
        
        # Convert sets to lists for JSON
        author["periods"] = list(author["periods"])
        author["genres"] = list(author["genres"])
    
    def _update_timeline(self, corpus: Dict, volume_metadata: VolumeMetadata,
                        poems_data: List[Dict]):
        """Update timeline data"""
        year = volume_metadata.publication_year
        period = volume_metadata.period.value if volume_metadata.period else "unknown"
        decade = f"{year // 10 * 10}s"
        
        # Initialize if not exists
        if str(year) not in corpus["timeline"]["by_year"]:
            corpus["timeline"]["by_year"][str(year)] = {
                "volumes": 0,
                "poems": 0,
                "authors": set(),
                "meters": {},
                "themes": {}
            }
        
        if period not in corpus["timeline"]["by_period"]:
            corpus["timeline"]["by_period"][period] = {
                "volumes": 0,
                "poems": 0,
                "years": set()
            }
        
        if decade not in corpus["timeline"]["by_decade"]:
            corpus["timeline"]["by_decade"][decade] = {
                "volumes": 0,
                "poems": 0,
                "years": set()
            }
        
        # Update counts
        year_data = corpus["timeline"]["by_year"][str(year)]
        year_data["volumes"] += 1
        year_data["poems"] += len(poems_data)
        year_data["authors"].add(volume_metadata.author_name)
        
        period_data = corpus["timeline"]["by_period"][period]
        period_data["volumes"] += 1
        period_data["poems"] += len(poems_data)
        period_data["years"].add(year)
        
        decade_data = corpus["timeline"]["by_decade"][decade]
        decade_data["volumes"] += 1
        decade_data["poems"] += len(poems_data)
        decade_data["years"].add(year)
        
        # Convert sets to lists
        year_data["authors"] = list(year_data["authors"])
        period_data["years"] = list(period_data["years"])
        decade_data["years"] = list(decade_data["years"])
    
    def _update_statistics(self, corpus: Dict, volume_metadata: VolumeMetadata,
                          poems_data: List[Dict]):
        """Update comprehensive statistics"""
        stats = corpus.setdefault("statistics", self._empty_statistics())
        
        # Basic counts
        stats["total_volumes"] = len(corpus["volumes"])
        stats["total_poems"] = corpus["metadata"]["total_poems"]
        stats["authors_count"] = len(corpus["authors"])
        
        # Publication years
        year = volume_metadata.publication_year
        if stats["publication_years"]["min"] is None or year < stats["publication_years"]["min"]:
            stats["publication_years"]["min"] = year
        if stats["publication_years"]["max"] is None or year > stats["publication_years"]["max"]:
            stats["publication_years"]["max"] = year
        
        # Update distribution
        stats["publication_years"]["distribution"][str(year)] = \
            stats["publication_years"]["distribution"].get(str(year), 0) + 1
        
        # Genre distribution
        for genre in volume_metadata.genres:
            genre_name = genre.value
            stats["genre_distribution"][genre_name] = \
                stats["genre_distribution"].get(genre_name, 0) + 1
        
        # Period distribution
        if volume_metadata.period:
            period_name = volume_metadata.period.value
            stats["period_distribution"][period_name] = \
                stats["period_distribution"].get(period_name, 0) + 1
        
        # Process poem-level statistics
        total_lines = 0
        total_words = 0
        unique_words_set = set()
        
        for poem_data in poems_data:
            analysis = poem_data.get("analysis", {})
            
            # Extract data based on format
            if hasattr(analysis, 'structural'):
                # EnhancedComprehensiveAnalysis object
                structural = analysis.structural
                content = analysis.content
                lines = structural.lines if hasattr(structural, 'lines') else 0
                words = content.total_words if hasattr(content, 'total_words') else 0
                meter = structural.aruz_analysis.identified_meter if hasattr(structural.aruz_analysis, 'identified_meter') else "unknown"
                themes = content.theme_distribution if hasattr(content, 'theme_distribution') else {}
            elif isinstance(analysis, dict):
                # Dictionary format
                structural = analysis.get("structural", {})
                content = analysis.get("content", {})
                lines = structural.get("lines", 0)
                words = content.get("total_words", 0)
                meter = structural.get("aruz_analysis", {}).get("identified_meter", "unknown")
                themes = content.get("theme_distribution", {})
            else:
                lines = 0
                words = 0
                meter = "unknown"
                themes = {}
            
            # Accumulate totals
            total_lines += lines
            total_words += words
            
            # Meter distribution
            stats["meter_distribution"][meter] = stats["meter_distribution"].get(meter, 0) + 1
            
            # Theme distribution
            for theme, count in themes.items():
                if count > 0:
                    stats["theme_distribution"][theme] = \
                        stats["theme_distribution"].get(theme, 0) + 1
            
            # Extract words for unique words count (simplified)
            if isinstance(poem_data.get("content"), str):
                words_list = re.findall(r'\b[\wӣӯ]+\b', poem_data["content"].lower())
                unique_words_set.update(words_list)
        
        # Update totals
        stats["total_lines"] = total_lines
        stats["total_words"] = total_words
        stats["unique_words"] = len(unique_words_set)
        
        # Calculate lexical diversity
        if total_words > 0:
            stats["lexical_diversity_by_period"][volume_metadata.period.value if volume_metadata.period else "unknown"] = \
                len(unique_words_set) / total_words
    
    def _save_volume_file(self, volume_id: str, volume_record: Dict, 
                         poems_data: List[Dict]):
        """Save volume data to separate file"""
        volume_file = self.volumes_dir / f"{volume_id}.json"
        
        volume_data = {
            "volume": volume_record,
            "poems": poems_data,
            "export_date": datetime.now().isoformat()
        }
        
        with open(volume_file, 'w', encoding='utf-8') as f:
            json.dump(volume_data, f, ensure_ascii=False, indent=2)
    
    def generate_timeline_report(self, output_format: str = "html") -> str:
        """Generate timeline visualization report"""
        corpus = self.load_corpus()
        
        if output_format == "html":
            return self._generate_html_timeline(corpus)
        elif output_format == "json":
            return json.dumps(corpus["timeline"], ensure_ascii=False, indent=2)
        else:
            return self._generate_text_timeline(corpus)
    
    def _generate_html_timeline(self, corpus: Dict) -> str:
        """Generate interactive HTML timeline"""
        
        # Prepare data
        years = sorted([int(y) for y in corpus["timeline"]["by_year"].keys() if y.isdigit()])
        counts = [corpus["timeline"]["by_year"][str(y)]["poems"] for y in years]
        
        periods = list(corpus["timeline"]["by_period"].keys())
        period_counts = [corpus["timeline"]["by_period"][p]["poems"] for p in periods]
        
        # Simple HTML template
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Tajik Poetry Timeline</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .stat { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .year-bar { background: #3498db; color: white; padding: 5px 10px; margin: 5px 0; border-radius: 3px; }
        .period { background: #2ecc71; color: white; padding: 8px 15px; margin: 8px 0; border-radius: 5px; }
        .meter-item { background: #9b59b6; color: white; padding: 5px 10px; margin: 3px; border-radius: 3px; display: inline-block; }
        .theme-item { background: #e74c3c; color: white; padding: 5px 10px; margin: 3px; border-radius: 3px; display: inline-block; }
        .summary { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Tajik Poetry Digital Library - Timeline Report</h1>
        
        <div class="summary">
            <h2>📊 Overall Statistics</h2>
            <p><strong>Total Volumes:</strong> {total_volumes}</p>
            <p><strong>Total Poems:</strong> {total_poems}</p>
            <p><strong>Total Authors:</strong> {total_authors}</p>
            <p><strong>Publication Range:</strong> {year_min} - {year_max}</p>
        </div>
        
        <div class="grid">
            <div class="stat">
                <h2>📅 Poems by Year</h2>
                {year_distribution}
            </div>
            
            <div class="stat">
                <h2>🕰️ Distribution by Historical Period</h2>
                {period_distribution}
            </div>
        </div>
        
        <div class="grid">
            <div class="stat">
                <h2>🎵 Meter Distribution</h2>
                {meter_distribution}
            </div>
            
            <div class="stat">
                <h2>🎭 Theme Distribution</h2>
                {theme_distribution}
            </div>
        </div>
        
        <div class="stat">
            <h2>👥 Authors in Library</h2>
            <p>{authors_list}</p>
        </div>
        
        <div class="stat">
            <h2>📈 Timeline Analysis</h2>
            <p>This library contains Tajik poetry from {period_count} historical periods, 
            spanning {year_count} years from {year_min} to {year_max}.</p>
            <p>The most productive period was <strong>{most_productive_period}</strong> with {max_poems} poems.</p>
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em;">
            <p>Generated on: {generation_date}</p>
            <p>Library Version: {library_version}</p>
            <p>License: CC-BY-NC-SA-4.0</p>
        </div>
    </div>
</body>
</html>"""
        
        # Prepare data for template
        stats = corpus.get("statistics", {})
        total_volumes = stats.get("total_volumes", 0)
        total_poems = stats.get("total_poems", 0)
        total_authors = len(corpus.get("authors", {}))
        year_min = stats.get("publication_years", {}).get("min", "N/A")
        year_max = stats.get("publication_years", {}).get("max", "N/A")
        
        # Year distribution
        year_distribution_html = ""
        if years and counts:
            max_count = max(counts) if counts else 1
            for year, count in zip(years, counts):
                width = (count / max_count) * 100
                year_distribution_html += f'<div class="year-bar" style="width: {width}%;">{year}: {count} poems</div>'
        
        # Period distribution
        period_distribution_html = ""
        if periods and period_counts:
            for period, count in zip(periods, period_counts):
                period_distribution_html += f'<div class="period">{period}: {count} poems</div>'
        
        # Meter distribution
        meter_distribution_html = ""
        meter_dist = stats.get("meter_distribution", {})
        for meter, count in sorted(meter_dist.items(), key=lambda x: -x[1])[:10]:
            if meter != "unknown":
                meter_distribution_html += f'<span class="meter-item">{meter}: {count}</span> '
        
        # Theme distribution
        theme_distribution_html = ""
        theme_dist = stats.get("theme_distribution", {})
        for theme, count in sorted(theme_dist.items(), key=lambda x: -x[1])[:8]:
            theme_distribution_html += f'<span class="theme-item">{theme}: {count}</span> '
        
        # Authors list
        authors = list(corpus.get("authors", {}).keys())
        authors_list = ", ".join(authors[:10])
        if len(authors) > 10:
            authors_list += f" and {len(authors) - 10} more"
        
        # Find most productive period
        most_productive_period = "N/A"
        max_poems = 0
        for period, data in corpus["timeline"]["by_period"].items():
            if data["poems"] > max_poems:
                max_poems = data["poems"]
                most_productive_period = period
        
        # Fill template
        return html_template.format(
            total_volumes=total_volumes,
            total_poems=total_poems,
            total_authors=total_authors,
            year_min=year_min,
            year_max=year_max,
            year_distribution=year_distribution_html,
            period_distribution=period_distribution_html,
            meter_distribution=meter_distribution_html,
            theme_distribution=theme_distribution_html,
            authors_list=authors_list,
            period_count=len(periods),
            year_count=len(years),
            most_productive_period=most_productive_period,
            max_poems=max_poems,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            library_version=corpus.get("metadata", {}).get("version", "1.0")
        )
    
    def _generate_text_timeline(self, corpus: Dict) -> str:
        """Generate text timeline report"""
        stats = corpus.get("statistics", {})
        
        report = [
            "Tajik Poetry Digital Library - Timeline Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Volumes: {stats.get('total_volumes', 0)}",
            f"Total Poems: {stats.get('total_poems', 0)}",
            f"Total Authors: {len(corpus.get('authors', {}))}",
            "",
            "Publication Year Range:",
            f"  From: {stats.get('publication_years', {}).get('min', 'N/A')}",
            f"  To: {stats.get('publication_years', {}).get('max', 'N/A')}",
            ""
        ]
        
        # Add period distribution
        report.append("Distribution by Historical Period:")
        for period, data in corpus["timeline"]["by_period"].items():
            report.append(f"  {period}: {data['poems']} poems, {data['volumes']} volumes")
        
        # Add meter distribution
        report.append("\nTop Meters:")
        meter_dist = stats.get("meter_distribution", {})
        for meter, count in sorted(meter_dist.items(), key=lambda x: -x[1])[:5]:
            report.append(f"  {meter}: {count}")
        
        return "\n".join(report)
    
    def export_contributions_for_git(self) -> Path:
        """
        Export all contributions for Git push
        
        Returns:
            Path to exported file
        """
        # Collect all volumes
        all_volumes = []
        
        for file in self.volumes_dir.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                all_volumes.append(json.load(f))
        
        # Create export data
        export_data = {
            "export_version": "2.0",
            "export_date": datetime.now().isoformat(),
            "library_metadata": self.load_corpus()["metadata"],
            "total_volumes": len(all_volumes),
            "volumes": all_volumes
        }
        
        export_dir = self.library_path / "exports"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"library_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        # Create README
        readme_path = export_path.parent / f"README_{datetime.now().strftime('%Y%m%d')}.md"
        readme_content = f"""# Tajik Poetry Library Export

## Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Statistics
- Volumes: {len(all_volumes)}
- Total Poems: {export_data['library_metadata']['total_poems']}
- Library Version: {export_data['library_metadata']['version']}

### Contents
This export contains complete metadata and analysis data for Tajik poetry volumes.

### Usage
1. Import this JSON file into your Tajik Poetry Analyzer
2. Use for research, visualization, or further analysis
3. Share with research collaborators

### License
CC-BY-NC-SA-4.0 - Attribution-NonCommercial-ShareAlike 4.0 International
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"Library export prepared: {export_path}")
        return export_path
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for storage"""
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text
    
    def load_corpus(self) -> Dict:
        """Load corpus data"""
        if self.corpus_file.exists():
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_corpus(self, corpus: Dict):
        """Save corpus data"""
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> Dict:
        """Get library statistics"""
        corpus = self.load_corpus()
        return corpus.get("statistics", {})
