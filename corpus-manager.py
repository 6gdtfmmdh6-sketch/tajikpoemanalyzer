# corpus_manager.py
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

class TajikCorpusManager:
    """
    Git-basierter Korpus-Manager für Tadschikische Poesie
    Ermöglicht dezentralen Beitrag und Synchronisierung
    """
    
    def __init__(self, local_repo_path: str = "./tajik_corpus"):
        self.local_repo = Path(local_repo_path)
        self.contributions_dir = self.local_repo / "contributions"
        self.master_corpus = self.local_repo / "corpus" / "master.json"
        self.initialize_structure()
        
        # GitHub API Endpoint (kann später konfiguriert werden)
        self.remote_url = "https://api.github.com/repos/username/tajik-poetry-corpus"
        
    def initialize_structure(self):
        """Initialisiert lokale Korpus-Struktur"""
        if not self.local_repo.exists():
            self.local_repo.mkdir(parents=True)
            self.contributions_dir.mkdir()
            (self.local_repo / "corpus").mkdir()
            
            # Initiales Korpus-Schema
            self.create_initial_corpus()
            
    def create_initial_corpus(self):
        """Erstellt initiales Korpus-Schema"""
        corpus_schema = {
            "metadata": {
                "version": "1.0.0",
                "created": datetime.now().isoformat(),
                "license": "CC-BY-NC-SA-4.0",
                "language": "tg",
                "script": "Cyrillic"
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
            "theme_distribution": {}
        }
        
        self.save_corpus(corpus_schema)
        
    def prepare_contribution(self, analysis_result: Dict, raw_text: str, 
                           user_info: Optional[Dict] = None) -> Dict:
        """
        Bereitet einen Beitrag für den Korpus vor
        
        Args:
            analysis_result: Vollständige Analyse aus TajikPoemAnalyzer
            raw_text: Roher Gedichttext
            user_info: Optionale Benutzerinformationen (Git username, email)
            
        Returns:
            Beitrags-Dictionary für lokale Speicherung
        """
        poem_id = analysis_result.get("poem_id", f"poem_{int(datetime.now().timestamp())}")
        
        # Berechne Hash für Deduplizierung
        content_hash = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()[:16]
        
        contribution = {
            "contribution_id": f"{content_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "poem_id": poem_id,
            "metadata": {
                "submission_date": datetime.now().isoformat(),
                "contributor": user_info or {"anonymous": True},
                "software_version": "2.0.0",
                "content_hash": content_hash,
                "source_language": "tg",
                "license_accepted": True
            },
            "raw_text": raw_text,
            "normalized_text": self.normalize_text(raw_text),
            "analysis": {
                "structural": self._serialize_analysis(analysis_result["analysis"].structural),
                "content": self._serialize_analysis(analysis_result["analysis"].content),
                "literary": self._serialize_analysis(analysis_result["analysis"].literary),
                "quality_metrics": analysis_result["validation"]
            },
            "tags": self._extract_tags(analysis_result)
        }
        
        return contribution
        
    def save_contribution(self, contribution: Dict):
        """
        Speichert Beitrag lokal im contributions-Verzeichnis
        """
        # Einzelne Beitragsdatei
        filename = f"{contribution['contribution_id']}.json"
        filepath = self.contributions_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(contribution, f, ensure_ascii=False, indent=2)
            
        # Auch zur Hauptdatei hinzufügen (lokales Merge)
        self._merge_to_local_corpus(contribution)
        
        logger.info(f"Beitrag gespeichert: {filename}")
        return filepath
        
    def _merge_to_local_corpus(self, contribution: Dict):
        """
        Mergt Beitrag ins lokale Korpus
        """
        corpus = self.load_corpus()
        
        # Prüfe auf Duplikate
        content_hash = contribution["metadata"]["content_hash"]
        existing_hashes = [p.get("metadata", {}).get("content_hash", "") 
                          for p in corpus["poems"]]
        
        if content_hash not in existing_hashes:
            # Füge Gedicht hinzu
            corpus["poems"].append({
                "id": contribution["poem_id"],
                "metadata": contribution["metadata"],
                "text": contribution["normalized_text"],
                "analysis_summary": {
                    "meter": contribution["analysis"]["structural"].get("aruz_analysis", {}).get("identified_meter"),
                    "lines": contribution["analysis"]["structural"].get("lines", 0),
                    "words": contribution["analysis"]["content"].get("total_words", 0)
                }
            })
            
            # Update Statistiken
            corpus["statistics"]["total_poems"] += 1
            corpus["statistics"]["total_lines"] += contribution["analysis"]["structural"].get("lines", 0)
            corpus["statistics"]["total_words"] += contribution["analysis"]["content"].get("total_words", 0)
            
            # Update Contributor-Liste
            contributor = contribution["metadata"]["contributor"]
            if contributor.get("username"):
                corpus["contributors"][contributor["username"]] = corpus["contributors"].get(
                    contributor["username"], 0) + 1
                
            self.save_corpus(corpus)
            
    def export_contributions_for_git(self) -> Path:
        """
        Exportiert alle Beiträge für Git-Push
        
        Returns:
            Pfad zur exportierten Datei
        """
        # Sammle alle Beiträge
        all_contributions = []
        
        for file in self.contributions_dir.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                all_contributions.append(json.load(f))
                
        # Erstelle Export-Datei
        export_data = {
            "export_version": "1.0",
            "export_date": datetime.now().isoformat(),
            "total_contributions": len(all_contributions),
            "contributions": all_contributions
        }
        
        export_path = self.local_repo / "exports" / f"contributions_{datetime.now().strftime('%Y%m%d')}.json"
        export_path.parent.mkdir(exist_ok=True)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        # Erstelle README für Git
        readme_path = export_path.parent / "README.md"
        readme_content = f"""# Tadschikische Poesie Korpus - Beiträge

## Export vom {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Statistiken
- Anzahl Beiträge: {len(all_contributions)}
- Enthaltene Gedichte: {len(all_contributions)}

### Anleitung für Repository-Maintainer
1. Diese Datei in das Haupt-Korpus-Repository kopieren
2. Beiträge prüfen und validieren
3. In das Master-Korpus integrieren
4. Statistiken aktualisieren

### Lizenz
Alle Beiträge stehen unter der CC-BY-NC-SA 4.0 Lizenz.
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
            
        logger.info(f"Export vorbereitet: {export_path}")
        return export_path
        
    def generate_git_commands(self):
        """
        Generiert Git-Befehle für den Nutzer zum Pushen
        """
        export_dir = self.local_repo / "exports"
        latest_export = max(export_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, default=None)
        
        if not latest_export:
            return "Keine Beiträge zum Exportieren gefunden."
            
        commands = f"""
# 1. Zum Korpus-Repository navigieren
cd /pfad/zum/tajik-poetry-corpus

# 2. Neuen Branch erstellen
git checkout -b contributions-{datetime.now().strftime('%Y%m%d')}

# 3. Exportierte Datei kopieren
cp "{latest_export}" ./contributions/

# 4. README kopieren
cp "{export_dir / 'README.md'}" ./

# 5. Änderungen hinzufügen und committen
git add contributions/ README.md
git commit -m "Neue Gedichtbeiträge vom {datetime.now().strftime('%Y-%m-%d')}"

# 6. Zu GitHub pushen
git push origin contributions-{datetime.now().strftime('%Y%m%d')}

# 7. Pull Request auf GitHub erstellen
# Gehe zu: https://github.com/username/tajik-poetry-corpus/pulls
"""
        return commands
        
    def sync_from_remote(self, remote_url: Optional[str] = None):
        """
        Synchronisiert lokales Korpus mit Remote (vereinfachte Version)
        """
        # In einer vollständigen Implementierung würde dies über Git oder API geschehen
        # Hier zeigen wir das Konzept
        try:
            # Beispiel: JSON von URL laden
            if remote_url:
                response = requests.get(f"{remote_url}/master.json", timeout=10)
                remote_corpus = response.json()
                
                # Merge-Strategie
                self._merge_remote_corpus(remote_corpus)
                logger.info("Korpus erfolgreich synchronisiert")
                
        except Exception as e:
            logger.error(f"Sync fehlgeschlagen: {e}")
            
    def normalize_text(self, text: str) -> str:
        """Normalisiert Text für konsistente Speicherung"""
        # Unicode normalisieren
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Standard-Zeilenumbrüche
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Mehrfache Leerzeilen reduzieren
        lines = []
        for line in text.split('\n'):
            if line.strip() or (lines and lines[-1].strip()):
                lines.append(line)
        
        return '\n'.join(lines)
        
    def _serialize_analysis(self, analysis_obj):
        """Serialisiert Analyse-Objekte für JSON"""
        if hasattr(analysis_obj, '__dict__'):
            return analysis_obj.__dict__
        return str(analysis_obj)
        
    def _extract_tags(self, analysis_result: Dict) -> List[str]:
        """Extrahiert Tags für die Suche"""
        tags = []
        analysis = analysis_result["analysis"]
        
        # Metrik-Tags
        meter = analysis.structural.aruz_analysis.identified_meter
        if meter != "unknown":
            tags.append(f"meter:{meter}")
            
        # Form-Tags
        form = analysis.structural.stanza_structure
        tags.append(f"form:{form}")
        
        # Thema-Tags
        for theme, count in analysis.content.theme_distribution.items():
            if count > 0:
                tags.append(f"theme:{theme.lower()}")
                
        # Register-Tag
        tags.append(f"register:{analysis.content.stylistic_register}")
        
        return tags
        
    def load_corpus(self) -> Dict:
        """Lädt das lokale Korpus"""
        if self.master_corpus.exists():
            with open(self.master_corpus, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def save_corpus(self, corpus: Dict):
        """Speichert das Korpus"""
        with open(self.master_corpus, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
