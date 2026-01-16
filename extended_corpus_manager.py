# extended_corpus_manager.py
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class Period(Enum):
    """Historische Perioden der tadschikischen Literatur"""
    CLASSICAL = "klassisch (vor 1920)"
    SOVIET_EARLY = "früh-sowjetisch (1920-1940)"
    SOVIET_MID = "mittel-sowjetisch (1940-1970)"
    SOVIET_LATE = "spät-sowjetisch (1970-1991)"
    INDEPENDENCE = "Unabhängigkeit (1991-2000)"
    CONTEMPORARY = "zeitgenössisch (2000-heute)"


class Genre(Enum):
    """Literarische Gattungen"""
    GHAZAL = "Ghazal"
    QASIDA = "Qasida"
    RUBAIYAT = "Rubaiyat"
    MASNAVI = "Masnavi"
    FREE_VERSE = "Freier Vers"
    PROSE_POEM = "Prosagedicht"
    MODERNIST = "Modernistisch"
    FOLK = "Volksdichtung"


@dataclass
class VolumeMetadata:
    """Metadaten für einen Gedichtband"""
    author_name: str
    author_birth_year: Optional[int] = None
    author_death_year: Optional[int] = None
    volume_title: str
    publication_year: int
    publisher: Optional[str] = None
    city: Optional[str] = None
    original_language: str = "tg"
    script: str = "Cyrillic"
    period: Optional[Period] = None
    genres: List[Genre] = None
    isbn: Optional[str] = None
    pages: Optional[int] = None
    translator: Optional[str] = None
    edition: Optional[str] = None
    source_type: str = "printed"  # printed, manuscript, digital
    
    def __post_init__(self):
        if self.genres is None:
            self.genres = []
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


class TajikLibraryManager:
    """
    Enhanced corpus manager with library functions for chronological analysis
    """
    
    def __init__(self, library_path: str = "./tajik_poetry_library"):
        self.library_path = Path(library_path)
        self.volumes_dir = self.library_path / "volumes"
        self.authors_dir = self.library_path / "authors"
        self.corpus_file = self.library_path / "corpus.json"
        self.stats_file = self.library_path / "statistics.json"
        self.initialize_library()
    
    def initialize_library(self):
        """Initialize library directory structure"""
        directories = [
            self.library_path,
            self.volumes_dir,
            self.authors_dir,
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
        for poem_data in poems_data:
            poem_id = self._register_poem(poem_data, volume_id, volume_metadata)
            volume_record["poem_ids"].append(poem_id)
        
        # Update corpus
        corpus = self.load_corpus()
        
        # Add volume
        corpus["volumes"].append(volume_record)
        corpus["metadata"]["total_volumes"] += 1
        
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
                      volume_metadata: VolumeMetadata) -> str:
        """Register a single poem"""
        # Generate poem ID
        poem_hash = hashlib.sha256(
            poem_data.get("content", "").encode('utf-8')
        ).hexdigest()[:12]
        
        poem_id = f"{volume_id}_poem_{poem_hash}"
        
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
            "analysis": poem_data.get("analysis", {}),
            "text": {
                "original": poem_data.get("content", ""),
                "normalized": self._normalize_text(poem_data.get("content", ""))
            },
            "structural_features": self._extract_structural_features(
                poem_data.get("analysis", {})
            )
        }
        
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
            "publisher": volume_metadata.publisher
        })
        
        if volume_metadata.period:
            author["periods"].add(volume_metadata.period.value)
        
        for genre in volume_metadata.genres:
            author["genres"].add(genre.value)
        
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
        if year not in corpus["timeline"]["by_year"]:
            corpus["timeline"]["by_year"][year] = {
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
        year_data = corpus["timeline"]["by_year"][year]
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
        stats["total_poems"] += len(poems_data)
        
        # Publication years
        year = volume_metadata.publication_year
        if stats["publication_years"]["min"] is None or year < stats["publication_years"]["min"]:
            stats["publication_years"]["min"] = year
        if stats["publication_years"]["max"] is None or year > stats["publication_years"]["max"]:
            stats["publication_years"]["max"] = year
        
        # Update distribution
        stats["publication_years"]["distribution"][year] = \
            stats["publication_years"]["distribution"].get(year, 0) + 1
        
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
        for poem_data in poems_data:
            analysis = poem_data.get("analysis", {})
            
            # Meter distribution
            meter = analysis.get("structural", {}).get("aruz_analysis", {}).get("identified_meter", "unknown")
            stats["meter_distribution"][meter] = stats["meter_distribution"].get(meter, 0) + 1
            
            # Theme distribution
            themes = analysis.get("content", {}).get("theme_distribution", {})
            for theme, count in themes.items():
                if count > 0:
                    stats["theme_distribution"][theme] = \
                        stats["theme_distribution"].get(theme, 0) + 1
            
            # Line and word counts
            stats["total_lines"] += analysis.get("structural", {}).get("lines", 0)
            stats["total_words"] += analysis.get("content", {}).get("total_words", 0)
    
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
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tajik Poetry Timeline</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .chart {{ margin: 20px 0; }}
                h2 {{ color: #2c3e50; }}
                .period {{ 
                    background: #f8f9fa; 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-left: 4px solid #3498db;
                }}
            </style>
        </head>
        <body>
            <h1>Tajik Poetry Digital Library Timeline</h1>
            
            <div class="chart">
                <h2>Poems by Publication Year</h2>
                <div id="yearChart"></div>
            </div>
            
            <div class="chart">
                <h2>Distribution by Historical Period</h2>
                <div id="periodChart"></div>
            </div>
            
            <div class="chart">
                <h2>Evolution of Poetic Forms</h2>
                <div id="meterChart"></div>
            </div>
            
            <script>
                // Year distribution data
                var yearData = {{
                    x: {years},
                    y: {counts},
                    type: 'bar',
                    name: 'Poems',
                    marker: {{color: '#3498db'}}
                }};
                
                // Period distribution data
                var periodData = [{{
                    labels: {period_labels},
                    values: {period_values},
                    type: 'pie',
                    hole: .4,
                    marker: {{colors: ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']}}
                }}];
                
                // Layout configurations
                var yearLayout = {{
                    title: 'Poems Published per Year',
                    xaxis: {{title: 'Publication Year'}},
                    yaxis: {{title: 'Number of Poems'}}
                }};
                
                var periodLayout = {{
                    title: 'Distribution by Historical Period'
                }};
                
                // Render charts
                Plotly.newPlot('yearChart', [yearData], yearLayout);
                Plotly.newPlot('periodChart', periodData, periodLayout);
            </script>
        </body>
        </html>
        """
        
        # Prepare data
        years = sorted(corpus["timeline"]["by_year"].keys())
        counts = [corpus["timeline"]["by_year"][y]["poems"] for y in years]
        
        periods = list(corpus["timeline"]["by_period"].keys())
        period_counts = [corpus["timeline"]["by_period"][p]["poems"] for p in periods]
        
        return html_template.format(
            years=json.dumps(years),
            counts=json.dumps(counts),
            period_labels=json.dumps(periods),
            period_values=json.dumps(period_counts)
        )
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug"""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for storage"""
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text
    
    def _extract_structural_features(self, analysis: Dict) -> Dict:
        """Extract key structural features for indexing"""
        structural = analysis.get("structural", {})
        content = analysis.get("content", {})
        
        return {
            "meter": structural.get("aruz_analysis", {}).get("identified_meter", "unknown"),
            "lines": structural.get("lines", 0),
            "syllables_per_line": structural.get("avg_syllables", 0),
            "rhyme_pattern": structural.get("rhyme_pattern", ""),
            "stanza_form": structural.get("stanza_structure", ""),
            "lexical_diversity": content.get("lexical_diversity", 0),
            "register": content.get("stylistic_register", "unknown"),
            "is_free_verse": structural.get("is_free_verse", False) if isinstance(structural, dict) else False
        }
    
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
