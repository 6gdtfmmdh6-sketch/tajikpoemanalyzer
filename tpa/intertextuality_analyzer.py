#!/usr/bin/env python3
"""
Diatextuelle Analyse für tadschikische/persische Poesie
Erkennung von Bezügen zu klassischen Werken
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
import re
import json
import logging
from difflib import SequenceMatcher
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class ClassicalReference:
    """Datenstruktur für klassische Referenzen"""
    source_author: str
    source_work: str
    source_text: str
    matched_text: str
    similarity_score: float
    reference_type: str  # 'direct_quote', 'allusion', 'theme', 'motif'
    confidence: float
    contextual_notes: List[str] = field(default_factory=list)


class ClassicalCorpus:
    """Datenbank klassischer persischer/tadschikischer Poesie"""

    def __init__(self, db_path: str = "data/corpus/classical_poetry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialisiere Datenbank mit klassischen Werken"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabelle für klassische Autoren
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classical_authors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                name_farsi TEXT,
                birth_year INTEGER,
                death_year INTEGER,
                region TEXT,
                literary_period TEXT,
                bio_summary TEXT
            )
        """)

        # Tabelle für klassische Werke
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classical_works (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                author_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                title_farsi TEXT,
                genre TEXT,
                meter TEXT,
                language TEXT,
                century INTEGER,
                line_count INTEGER,
                FOREIGN KEY(author_id) REFERENCES classical_authors(id)
            )
        """)

        # Tabelle für Verse (Bait)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                work_id INTEGER NOT NULL,
                line_number INTEGER,
                misra1 TEXT NOT NULL,
                misra2 TEXT,
                complete_bait TEXT,
                phonetic_pattern TEXT,
                keywords TEXT,
                themes TEXT,
                FOREIGN KEY(work_id) REFERENCES classical_works(id)
            )
        """)

        # Tabelle für bekannte Motive/Topoi
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS literary_motifs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                motif_name TEXT NOT NULL,
                motif_name_farsi TEXT,
                description TEXT,
                example_verses TEXT,
                thematic_category TEXT
            )
        """)

        # Indizes für schnelle Suche
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_verses_text ON verses(misra1, misra2)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_verses_keywords ON verses(keywords)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_works_author ON classical_works(author_id)")

        conn.commit()
        conn.close()

        # Standard-Daten einfügen (wenn leer)
        self._populate_with_classics()

    def _populate_with_classics(self):
        """Füge Standard-Klassiker ein"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Prüfe ob bereits Daten existieren
        cursor.execute("SELECT COUNT(*) FROM classical_authors")
        if cursor.fetchone()[0] > 0:
            conn.close()
            logger.info("Classical corpus already populated")
            return

        # Klassische persische/tadschikische Dichter
        classical_authors = [
            # Persische Klassiker
            ("Rudaki", "رودکی", 858, 941, "Samaniden", "Early Persian", "Vater der persischen Poesie"),
            ("Ferdowsi", "فردوسی", 940, 1020, "Ghaznaviden", "Epic", "Shahnameh-Autor"),
            ("Omar Khayyam", "عمر خیام", 1048, 1131, "Seldschuken", "Philosophical", "Rubaiyat-Autor"),
            ("Attar", "عطار", 1145, 1221, "Persien", "Sufi", "Mantiq al-Tayr"),
            ("Rumi", "مولانا", 1207, 1273, "Rum-Sultanat", "Sufi", "Mathnawi, Diwan-e Shams"),
            ("Saadi", "سعدی", 1210, 1291, "Persien", "Ethical", "Gulistan, Bustan"),
            ("Hafez", "حافظ", 1315, 1390, "Shiraz", "Lyric", "Ghazal-Meister"),
            ("Jami", "جامی", 1414, 1492, "Timuridenreich", "Sufi", "Letzter großer klassischer Dichter"),

            # Tadschikische/sowjetische Dichter
            ("Sadriddin Ayni", "Садриддин Айнӣ", 1878, 1954, "Tadschikistan", "Modern", "Begründer der modernen tadschikischen Literatur"),
            ("Mirzo Tursunzoda", "Мирзо Турсунзода", 1911, 1977, "Tadschikistan", "Soviet", "Nationaldichter"),
            ("Bozor Sobir", "Бозор Собир", 1938, 2018, "Tadschikistan", "Modern", "Widerstandsdichter"),
            ("Gulruchsor", "Гулрухсор", 1947, None, "Tadschikistan", "Contemporary", "Führende Dichterin"),
            ("Loiq Sherali", "Лоиқ Шералӣ", 1941, 2000, "Tadschikistan", "Soviet", "Populärer Lyriker"),

            # Arabische Einflüsse
            ("Al-Mutanabbi", "المتنبي", 915, 965, "Arabien", "Classical Arabic", "Einfluss auf persische Poesie"),
            ("Abu Nuwas", "أبو نواس", 756, 814, "Abbasiden", "Classical Arabic", "Hedonistische Poesie"),
        ]

        for author in classical_authors:
            cursor.execute("""
                INSERT OR IGNORE INTO classical_authors
                (name, name_farsi, birth_year, death_year, region, literary_period, bio_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, author)

        conn.commit()

        # Füge einige bekannte Werke hinzu
        self._add_classical_works(cursor)

        # Füge literarische Motive hinzu
        self._add_literary_motifs(cursor)

        conn.commit()
        conn.close()
        logger.info(f"Populated database with {len(classical_authors)} classical authors")

    def _add_classical_works(self, cursor):
        """Füge klassische Werke zur Datenbank hinzu"""
        # Hole Autor-IDs
        cursor.execute("SELECT id, name FROM classical_authors")
        authors = {name: id for id, name in cursor.fetchall()}

        classical_works = [
            (authors.get("Ferdowsi"), "Shahnameh", "شاهنامه", "epic", "mutaqarib", "persian", 11, 60000),
            (authors.get("Rumi"), "Mathnawi", "مثنوی معنوی", "mystical", "various", "persian", 13, 25000),
            (authors.get("Hafez"), "Divan-e Hafez", "دیوان حافظ", "ghazal", "various", "persian", 14, 5000),
            (authors.get("Saadi"), "Gulistan", "گلستان", "ethical", "prose", "persian", 13, 1000),
            (authors.get("Saadi"), "Bustan", "بوستان", "ethical", "mathnawi", "persian", 13, 4000),
            (authors.get("Omar Khayyam"), "Rubaiyat", "رباعیات", "philosophical", "rubai", "persian", 12, 1000),
            (authors.get("Attar"), "Mantiq al-Tayr", "منطق الطیر", "mystical", "mathnawi", "persian", 12, 4500),
        ]

        for work in classical_works:
            if work[0]:  # Nur wenn Autor existiert
                cursor.execute("""
                    INSERT OR IGNORE INTO classical_works
                    (author_id, title, title_farsi, genre, meter, language, century, line_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, work)

        # Füge einige berühmte Verse hinzu
        self._add_famous_verses(cursor)

    def _add_famous_verses(self, cursor):
        """Füge berühmte Verse zur Datenbank hinzu"""
        # Saadi's berühmter Vers aus Gulistan
        cursor.execute("SELECT id FROM classical_authors WHERE name = ?", ("Saadi",))
        saadi_id = cursor.fetchone()
        if saadi_id:
            cursor.execute("SELECT id FROM classical_works WHERE author_id = ? AND title = ?",
                         (saadi_id[0], "Gulistan"))
            gulistan_id = cursor.fetchone()
            if gulistan_id:
                famous_verses = [
                    (gulistan_id[0], 1,
                     "بنی آدم اعضای یک پیکرند",
                     "که در آفرینش ز یک گوهرند",
                     "بنی آدم اعضای یک پیکرند که در آفرینش ز یک گوهرند",
                     None,
                     json.dumps(["انسانیت", "برادری", "همدردی"]),
                     json.dumps(["humanity", "brotherhood", "empathy"])),

                    (gulistan_id[0], 2,
                     "چو عضوی به درد آورد روزگار",
                     "دگر عضوها را نماند قرار",
                     "چو عضوی به درد آورد روزگار دگر عضوها را نماند قرار",
                     None,
                     json.dumps(["همدردی", "انسانیت"]),
                     json.dumps(["empathy", "solidarity"])),
                ]

                for verse in famous_verses:
                    cursor.execute("""
                        INSERT OR IGNORE INTO verses
                        (work_id, line_number, misra1, misra2, complete_bait, phonetic_pattern, keywords, themes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, verse)

    def _add_literary_motifs(self, cursor):
        """Füge literarische Motive zur Datenbank hinzu"""
        persian_motifs = [
            ("nightingale_rose", "بلبل و گل", "Nachtigall und Rose - Symbol für Liebenden und Geliebte",
             None, "love"),
            ("wine_tavern", "می و میخانه", "Wein und Taverne - Mystische Trunkenheit und spirituelle Suche",
             None, "mysticism"),
            ("beloved_tresses", "زلف یار", "Locken der Geliebten - Verwirrung und Verstrickung des Liebenden",
             None, "love"),
            ("spring_garden", "باغ بهار", "Frühlingsgarten - Erneuerung und Schönheit",
             None, "nature"),
            ("separation", "فراق", "Trennung vom Geliebten - Schmerz und Sehnsucht",
             None, "love"),
            ("union", "وصال", "Vereinigung mit dem Geliebten - Spirituelle Erfüllung",
             None, "mysticism"),
        ]

        for motif in persian_motifs:
            cursor.execute("""
                INSERT OR IGNORE INTO literary_motifs
                (motif_name, motif_name_farsi, description, example_verses, thematic_category)
                VALUES (?, ?, ?, ?, ?)
            """, motif)

    def find_intertextual_references(self, text: str, min_similarity: float = 0.7) -> List[ClassicalReference]:
        """Finde intertextuelle Bezüge in gegebenem Text"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        references = []

        try:
            # Suche nach direkten Zitaten
            references.extend(self._find_direct_quotes(cursor, text, min_similarity))

            # Suche nach thematischen Ähnlichkeiten
            references.extend(self._find_thematic_allusions(cursor, text))

            # Suche nach motivischen Übereinstimmungen
            references.extend(self._find_motif_references(cursor, text))

        except Exception as e:
            logger.error(f"Error finding intertextual references: {e}")
        finally:
            conn.close()

        # Dedupliziere und sortiere nach Konfidenz
        references = self._deduplicate_references(references)
        references.sort(key=lambda r: r.confidence, reverse=True)

        return references

    def _find_direct_quotes(self, cursor, text: str, min_similarity: float) -> List[ClassicalReference]:
        """Finde direkte Zitate aus klassischen Werken"""
        references = []

        # Zerlege Text in Zeilen
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) >= 10]

        for line in lines[:50]:  # Limitiere auf erste 50 Zeilen für Performance
            # Suche nach ähnlichen Versen in der Datenbank
            cursor.execute("""
                SELECT v.misra1, v.misra2, v.complete_bait,
                       a.name, w.title, v.themes, w.genre
                FROM verses v
                JOIN classical_works w ON v.work_id = w.id
                JOIN classical_authors a ON w.author_id = a.id
                LIMIT 100
            """)

            for misra1, misra2, bait, author, work, themes, genre in cursor.fetchall():
                # Vergleiche mit beiden Halbversen und komplettem Vers
                texts_to_compare = [t for t in [misra1, misra2, bait] if t]

                for source_text in texts_to_compare:
                    similarity = self._calculate_similarity(line, source_text)

                    if similarity >= min_similarity:
                        references.append(ClassicalReference(
                            source_author=author,
                            source_work=work,
                            source_text=source_text[:200],  # Begrenze Länge
                            matched_text=line,
                            similarity_score=similarity,
                            reference_type='direct_quote' if similarity > 0.9 else 'allusion',
                            confidence=similarity,
                            contextual_notes=[f"Genre: {genre}", f"Themes: {themes}"]
                        ))

        return references

    def _find_thematic_allusions(self, cursor, text: str) -> List[ClassicalReference]:
        """Finde thematische Anspielungen"""
        references = []

        # Extrahiere Schlüsselwörter aus Text
        keywords = self._extract_keywords(text)

        if not keywords:
            return references

        # Suche nach Versen mit ähnlichen Keywords
        for keyword in keywords[:15]:  # Top 15 Keywords
            cursor.execute("""
                SELECT v.complete_bait, a.name, w.title, v.themes, v.keywords
                FROM verses v
                JOIN classical_works w ON v.work_id = w.id
                JOIN classical_authors a ON w.author_id = a.id
                WHERE v.keywords LIKE ? OR v.themes LIKE ?
                LIMIT 5
            """, (f"%{keyword}%", f"%{keyword}%"))

            for bait, author, work, themes, verse_keywords in cursor.fetchall():
                theme_similarity = self._calculate_theme_similarity(keyword, themes, verse_keywords)

                if theme_similarity > 0.5:
                    references.append(ClassicalReference(
                        source_author=author,
                        source_work=work,
                        source_text=bait[:150] if bait else "",
                        matched_text=keyword,
                        similarity_score=theme_similarity,
                        reference_type='thematic_allusion',
                        confidence=theme_similarity * 0.7,  # Lower confidence for thematic matches
                        contextual_notes=[f"Keyword: {keyword}", f"Themes: {themes}"]
                    ))

        return references

    def _find_motif_references(self, cursor, text: str) -> List[ClassicalReference]:
        """Finde literarische Motive"""
        references = []

        # Bekannte literarische Motive mit ihren persischen/arabischen Begriffen
        motif_patterns = {
            "nightingale": ["بلبل", "bolbol", "булбул"],
            "rose": ["گل", "gul", "гул"],
            "wine": ["می", "may", "май", "шароб", "sharab"],
            "beloved": ["یار", "معشوق", "дилбар", "ёр"],
            "tresses": ["زلف", "zulf", "зулф"],
            "spring": ["بهار", "bahar", "баҳор"],
            "garden": ["باغ", "bagh", "бог"],
            "tavern": ["میخانه", "خرابات", "майхона"],
        }

        text_lower = text.lower()

        for motif_name, patterns in motif_patterns.items():
            for pattern in patterns:
                if pattern in text_lower or re.search(pattern, text, re.IGNORECASE):
                    # Finde Beispiele dieses Motivs in der Datenbank
                    cursor.execute("""
                        SELECT motif_name_farsi, description, thematic_category
                        FROM literary_motifs
                        WHERE motif_name LIKE ?
                        LIMIT 1
                    """, (f"%{motif_name}%",))

                    result = cursor.fetchone()
                    if result:
                        farsi_name, description, category = result
                        references.append(ClassicalReference(
                            source_author="Classical Tradition",
                            source_work="Persian Literary Motifs",
                            source_text=description or f"Motif: {farsi_name}",
                            matched_text=pattern,
                            similarity_score=1.0,
                            reference_type='motif',
                            confidence=0.8,
                            contextual_notes=[
                                f"Motif: {motif_name}",
                                f"Category: {category}",
                                f"Pattern matched: {pattern}"
                            ]
                        ))
                    break  # Nur einmal pro Motiv

        return references

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrahiere bedeutende Schlüsselwörter aus Text"""
        # Stoppwörter (persisch/tadschikisch/arabisch)
        stopwords = {
            # Persisch
            'و', 'در', 'به', 'از', 'که', 'را', 'این', 'با', 'است',
            'برای', 'آن', 'یک', 'تا', 'بر', 'هم', 'یا', 'اما',
            # Tadschikisch (Kyrillisch)
            'ва', 'дар', 'ба', 'аз', 'ки', 'ро', 'ин', 'бо', 'аст',
            'барои', 'он', 'як', 'то', 'бар', 'ҳам', 'ё', 'аммо',
            # Russisch
            'и', 'в', 'на', 'с', 'от', 'по', 'для', 'что', 'как', 'это',
        }

        # Tokenisierung - unterstützt Kyrillisch, Arabisch, Persisch
        words = re.findall(r'[\w\u0600-\u06FF\u0400-\u04FF]+', text)

        # Filtere Stoppwörter und kurze Wörter
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]

        # Zähle Häufigkeit
        word_freq = Counter(keywords)

        # Gebe Top Keywords zurück
        return [word for word, _ in word_freq.most_common(30)]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Berechne Ähnlichkeit zwischen zwei Texten"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _calculate_theme_similarity(self, keyword: str, themes_json: Optional[str],
                                   keywords_json: Optional[str]) -> float:
        """Berechne thematische Ähnlichkeit"""
        try:
            themes = json.loads(themes_json) if themes_json else []
            keywords_list = json.loads(keywords_json) if keywords_json else []

            keyword_lower = keyword.lower()

            # Exakte Übereinstimmung
            if keyword_lower in [t.lower() for t in themes]:
                return 1.0
            if keyword_lower in [k.lower() for k in keywords_list]:
                return 0.9

            # Teilweise Übereinstimmung
            for theme in themes:
                if keyword_lower in theme.lower() or theme.lower() in keyword_lower:
                    return 0.7

            for kw in keywords_list:
                if keyword_lower in kw.lower() or kw.lower() in keyword_lower:
                    return 0.6

        except (json.JSONDecodeError, TypeError):
            pass

        return 0.0

    def _deduplicate_references(self, references: List[ClassicalReference]) -> List[ClassicalReference]:
        """Entferne Duplikate basierend auf Quelle und matched_text"""
        seen = set()
        unique_refs = []

        for ref in references:
            key = (ref.source_author, ref.source_work, ref.matched_text[:50])
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)

        return unique_refs

    def get_corpus_statistics(self) -> Dict[str, int]:
        """Hole Statistiken über das Korpus"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM classical_authors")
        stats['authors'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM classical_works")
        stats['works'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM verses")
        stats['verses'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM literary_motifs")
        stats['motifs'] = cursor.fetchone()[0]

        conn.close()
        return stats
