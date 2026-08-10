#!/usr/bin/env python3
"""
corpus_core — the single corpus layer.

Replaces the split between corpus_manager.TajikCorpusManager (analysis path,
thin schema, no author) and extended_corpus_manager.TajikLibraryManager (rich
schema, never filled). Both wrote to different files; the analysis path won,
so every poem ended up without an author, and the timeline stayed empty.

Model
-----
    Author   — a person who writes. Name variants, life dates, notes.
    Work     — a bounded textual source: a printed volume, a samizdat
               notebook, a manuscript draft, a periodical issue, a social
               media account. Carries BOTH composition and publication year,
               because for samizdat these differ by decades.
    Poem     — a text in a Work by an Author, plus its analysis.

Every poem therefore always has an author_id and a work_id. Analysis records
are stored in full (`analysis`) plus a flat, queryable `features` block.

Design rules
------------
* Enum values are stored as plain lowercase strings, never as "StanzaForm.X".
* Titles have a canonical form plus a `variants` list — the old corpus held
  one volume under four spellings.
* `meter_status` distinguishes identified / free_verse / detection_failed.
  "unknown" conflated "this poem has no meter by design" with "the detector
  gave up", which are opposite findings.
* Nothing here writes poem texts to a public file; see knowledge_export.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "3.0.0"

SOURCE_TYPES = [
    "printed",       # published by a press
    "samizdat",      # self-produced, circulated outside publishing
    "manuscript",    # unpublished draft or fair copy
    "periodical",    # journal, newspaper, almanac
    "online",        # social media, blog, web magazine
    "oral",          # recorded performance / recitation
    "unknown",
]

PERIODS = [
    ("classical", None, 1920),
    ("soviet_early", 1920, 1940),
    ("soviet_mid", 1940, 1970),
    ("soviet_late", 1970, 1991),
    ("independence", 1991, 2000),
    ("contemporary", 2000, None),
]


def infer_period(year: Optional[int]) -> Optional[str]:
    if not year:
        return None
    for name, lo, hi in PERIODS:
        if (lo is None or year >= lo) and (hi is None or year < hi):
            return name
    return None


def normalize_text(text: str) -> str:
    """NFC + collapsed whitespace. Used for hashing and comparison."""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    s = unicodedata.normalize("NFKD", text or "").lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    return re.sub(r"[-\s]+", "-", s).strip("-")[:60] or "unnamed"


def clean_enum(value: Any) -> Any:
    """'StanzaForm.FREE_VERSE' -> 'free_verse'. Idempotent."""
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str) and "." in value:
        head, _, tail = value.partition(".")
        if head[:1].isupper() and tail.isupper() or (head[:1].isupper() and "_" in tail):
            return tail.lower()
    return value.lower() if isinstance(value, str) else value


def title_key(title: str) -> str:
    """Match 'Tūfonhoi Sokit' and 'Tūfonhoi sokit' and 'TŪFONHOI SOKIT'."""
    s = unicodedata.normalize("NFKD", title or "").lower()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\(.*?\)", " ", s)              # drop "(DRAFT)" etc.
    return re.sub(r"[^\w]+", "", s, flags=re.UNICODE)


@dataclass
class Author:
    id: str
    name: str
    name_variants: List[str] = field(default_factory=list)
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    notes: Optional[str] = None

    @staticmethod
    def make_id(name: str) -> str:
        return f"A_{slugify(name)}"


@dataclass
class Work:
    """A bounded textual source. 'Volume' was too narrow a word."""
    id: str
    author_id: str
    title: str
    title_variants: List[str] = field(default_factory=list)
    source_type: str = "unknown"
    publication_year: Optional[int] = None
    composition_year: Optional[int] = None       # differs for samizdat
    composition_year_uncertain: bool = False
    publisher: Optional[str] = None
    city: Optional[str] = None
    is_draft: bool = False
    witness_of: Optional[str] = None             # work_id this is a draft of
    collected_from: Optional[str] = None         # archive, person, fieldwork
    collected_date: Optional[str] = None
    notes: Optional[str] = None

    @property
    def period(self) -> Optional[str]:
        return infer_period(self.publication_year or self.composition_year)

    @property
    def timeline_year(self) -> Optional[int]:
        """Year to place this on a timeline: when it was written if known."""
        return self.composition_year or self.publication_year

    @staticmethod
    def make_id(author_id: str, title: str, year: Optional[int]) -> str:
        return f"W_{author_id[2:]}_{slugify(title)}_{year or 'nd'}"


class Corpus:
    """Single entry point for reading and writing the corpus."""

    def __init__(self, root: str | Path = "./tajik_corpus"):
        self.root = Path(root)
        self.path = self.root / "corpus" / "corpus.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        (self.root / "exports").mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    # ---------------------------------------------------------------- io

    def _empty(self) -> Dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "created": datetime.now().isoformat(timespec="seconds"),
                "language": "tg",
                "script": "Cyrillic",
                "license_data": "CC-BY-4.0 (derived data only)",
                "description": "Tajik poetry research corpus",
            },
            "authors": {},
            "works": {},
            "poems": [],
            "vocabulary": {},
        }

    def _load(self) -> Dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return self._empty()

    def save(self) -> None:
        self.data["metadata"]["updated"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        tmp.replace(self.path)

    # ----------------------------------------------------------- authors

    def upsert_author(self, name: str, **kw) -> str:
        aid = Author.make_id(name)
        existing = self.data["authors"].get(aid)
        if existing:
            for variant in kw.pop("name_variants", []):
                if variant not in existing["name_variants"]:
                    existing["name_variants"].append(variant)
            existing.update({k: v for k, v in kw.items() if v is not None})
        else:
            self.data["authors"][aid] = asdict(Author(id=aid, name=name, **kw))
        return aid

    def find_author_by_name(self, name: str) -> Optional[str]:
        key = title_key(name)
        for aid, a in self.data["authors"].items():
            if title_key(a["name"]) == key:
                return aid
            if any(title_key(v) == key for v in a.get("name_variants", [])):
                return aid
        return None

    # ------------------------------------------------------------- works

    def upsert_work(self, author_id: str, title: str, **kw) -> str:
        """Match on normalized title + author, so spelling variants merge."""
        key = title_key(title)
        for wid, w in self.data["works"].items():
            if w["author_id"] != author_id:
                continue
            known = [title_key(w["title"])] + [title_key(v) for v in w.get("title_variants", [])]
            if key in known and bool(kw.get("is_draft")) == bool(w.get("is_draft")):
                if title != w["title"] and title not in w["title_variants"]:
                    w["title_variants"].append(title)
                for k, v in kw.items():
                    if v is not None and not w.get(k):
                        w[k] = v
                return wid
        wid = Work.make_id(author_id, title, kw.get("publication_year"))
        self.data["works"][wid] = asdict(Work(id=wid, author_id=author_id, title=title, **kw))
        return wid

    # ------------------------------------------------------------- poems

    def add_poem(self, *, title: str, text: str, author: str, work_title: str,
                 analysis: Optional[Dict] = None, author_kw: Optional[Dict] = None,
                 work_kw: Optional[Dict] = None, contributor: Optional[Dict] = None,
                 poem_id: Optional[str] = None) -> Optional[str]:
        """Single ingestion path. Every poem gets an author and a work."""
        h = text_hash(text)
        aid = self.upsert_author(author, **(author_kw or {}))
        wid = self.upsert_work(aid, work_title, **(work_kw or {}))

        # The same text attested in a second work is NOT a duplicate to be
        # dropped: for textual criticism, which witnesses carry a poem is
        # itself the finding. Record an attestation and keep one record.
        for existing in self.data["poems"]:
            if existing["text_sha256"] == h:
                if not any(a["work_id"] == wid for a in existing.setdefault("attestations", [])):
                    existing["attestations"].append({
                        "work_id": wid, "title_as_given": title,
                        "added": datetime.now().isoformat(timespec="seconds"),
                    })
                    logger.info("Poem '%s' also attested in %s", title, wid)
                return existing["id"]
        pid = poem_id or f"P{len(self.data['poems']) + 1:04d}"

        entry = {
            "id": pid,
            "title": title,
            "author_id": aid,
            "work_id": wid,
            "text": text,
            "text_sha256": h,
            "incipit": self._incipit(text),
            "added": datetime.now().isoformat(timespec="seconds"),
            "contributor": contributor or {"anonymous": True},
            "attestations": [{"work_id": wid, "title_as_given": title,
                              "added": datetime.now().isoformat(timespec="seconds")}],
            "features": extract_features(analysis or {}),
            "analysis": analysis or {},
        }
        self.data["poems"].append(entry)
        self._update_vocabulary(text)
        return pid

    @staticmethod
    def _incipit(text: str, max_len: int = 60) -> str:
        lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
        if not lines:
            return ""
        return (lines[1] if len(lines) > 1 else lines[0])[:max_len]

    def _update_vocabulary(self, text: str) -> None:
        vocab = self.data.setdefault("vocabulary", {})
        for w in re.findall(r"[\w\u0400-\u04FF]+", (text or "").lower()):
            vocab[w] = vocab.get(w, 0) + 1

    # -------------------------------------------------------- statistics

    def statistics(self) -> Dict:
        poems = self.data["poems"]
        return {
            "authors": len(self.data["authors"]),
            "works": len(self.data["works"]),
            "poems": len(poems),
            "unique_words": len(self.data.get("vocabulary", {})),
            "total_words": sum(p["features"].get("words") or 0 for p in poems),
            "by_source_type": self._count(lambda p: self.work_of(p).get("source_type")),
            "by_form": self._count(lambda p: p["features"].get("stanza_form")),
            "by_meter_status": self._count(lambda p: p["features"].get("meter_status")),
            "by_author": self._count(lambda p: self.data["authors"][p["author_id"]]["name"]),
            "multiply_attested": sum(1 for p in poems if len(p.get("attestations", [])) > 1),
        }

    def _count(self, key) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for p in self.data["poems"]:
            k = str(key(p) or "unknown")
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def work_of(self, poem: Dict) -> Dict:
        return self.data["works"].get(poem.get("work_id"), {})

    # ---------------------------------------------------------- timeline

    def timeline(self) -> List[Dict]:
        """Chronological spine of the corpus, by year of composition where
        known and publication otherwise — the distinction that matters for
        samizdat, where circulation long precedes print."""
        buckets: Dict[int, Dict] = {}
        for wid, w in self.data["works"].items():
            work = Work(**{k: v for k, v in w.items() if k in Work.__dataclass_fields__})
            year = work.timeline_year
            if not year:
                continue
            b = buckets.setdefault(year, {
                "year": year, "period": infer_period(year),
                "works": [], "poem_count": 0, "authors": set(),
            })
            b["works"].append({
                "id": wid, "title": w["title"], "source_type": w.get("source_type"),
                "is_draft": w.get("is_draft", False),
                "dated_by": "composition" if w.get("composition_year") else "publication",
            })
            n = sum(1 for p in self.data["poems"]
                    if any(a["work_id"] == wid for a in p.get("attestations",
                                                              [{"work_id": p["work_id"]}])))
            b["poem_count"] += n
            b["authors"].add(self.data["authors"].get(w["author_id"], {}).get("name", "?"))
        out = []
        for year in sorted(buckets):
            b = buckets[year]
            b["authors"] = sorted(b["authors"])
            out.append(b)
        return out


# ------------------------------------------------------------------ features

def extract_features(analysis: Dict) -> Dict:
    """Flatten the analysis into a queryable feature row.

    The old corpus stored five fields and dropped everything else, including
    MTLD — the one metric the MA thesis actually argues from. Everything here
    is already computed by the analyzer; it was simply never persisted.
    """
    structural = analysis.get("structural", {}) or {}
    content = analysis.get("content", {}) or {}
    quality = analysis.get("quality_metrics", {}) or {}

    fv = quality.get("free_verse_analysis") or {}
    aruz = structural.get("aruz_analysis", {}) or {}
    radif = structural.get("radif_analysis", {}) or {}
    syllables = structural.get("syllables_per_line", []) or []

    meter = clean_enum(aruz.get("identified_meter") or "unknown")
    form = clean_enum(structural.get("stanza_structure") or "unknown")

    if meter and meter != "unknown":
        meter_status = "identified"
    elif form in ("free_verse", "free verse"):
        meter_status = "free_verse"
    else:
        meter_status = "detection_failed"

    mean_syl = sum(syllables) / len(syllables) if syllables else None
    if syllables and len(syllables) > 1:
        var = sum((s - mean_syl) ** 2 for s in syllables) / len(syllables)
    else:
        var = None

    themes = {k: v for k, v in (content.get("theme_distribution") or {}).items() if v}

    return {
        "lines": structural.get("lines"),
        "words": content.get("total_words"),
        "unique_words": content.get("unique_words"),
        # The analyzer stores MTLD under the misleading name
        # `lexical_diversity` (its docstring even says "type-token ratio",
        # which it is not — it is McCarthy & Jarvis MTLD). Persist it under
        # the correct name so the metric the thesis argues from is queryable.
        "mtld": content.get("lexical_diversity"),
        "stanza_form": form,
        "meter": meter,
        "meter_status": meter_status,
        "meter_confidence": clean_enum(aruz.get("confidence")),
        "meter_accuracy": aruz.get("pattern_accuracy"),
        "rhyme_pattern": structural.get("rhyme_pattern"),
        "has_radif": bool(radif.get("radif_present")),
        "radif_text": radif.get("radif_text") or None,
        "syllables_mean": round(mean_syl, 2) if mean_syl is not None else None,
        "syllables_variance": round(var, 2) if var is not None else None,
        "line_length_min": min(syllables) if syllables else None,
        "line_length_max": max(syllables) if syllables else None,
        "enjambment_score": fv.get("enjambement_score"),
        "free_verse_confidence": fv.get("confidence"),
        "prose_poetry_score": fv.get("prose_poetry_score"),
        "line_variation_score": fv.get("line_variation_score"),
        "primary_theme": content.get("primary_theme"),
        "themes": themes,
        "neologism_count": len(content.get("neologisms") or []),
    }
