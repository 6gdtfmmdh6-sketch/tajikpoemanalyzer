#!/usr/bin/env python3
"""
One-off migration: old master.json (thin, authorless) -> corpus.json (v3).

What it repairs, per poem:
  * assigns an author (the old schema had no author field at all)
  * merges the four spellings of two volume titles into two Works, and
    separates the draft witness as its own Work linked via `witness_of`
  * re-runs the analyzer so MTLD, syllable profile, themes and enjambment
    are persisted instead of being recomputed and thrown away
  * normalizes enums and replaces "unknown" meter with an explicit
    meter_status

Usage:
    python scripts/migrate_corpus.py [--master PATH] [--author NAME] [--dry-run]

The old master.json is left untouched; the result is written to
tajik_corpus/corpus/corpus.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from corpus_core import Corpus, title_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("migrate")

# Known source characteristics, keyed by normalized title.
# Extend this table as further works enter the corpus.
# Explicit alias table. Cyrillic and Latin-transliterated titles cannot be
# matched automatically without a transliteration engine, and guessing across
# scripts would silently merge distinct works. Every canonical work lists the
# normalized title keys that belong to it — auditable and extensible.
WORK_PROFILES = {
    "Tūfonhoi sokit": {
        "aliases": ["tufonhoisokit"],
        "source_type": "printed",
        "publication_year": 2008,
    },
    "Бо аноре ҳамхун": {
        "aliases": ["boanorehamhun", "боанореҳамхун", "боанорехамхун"],
        "source_type": "printed",
        "publication_year": 2019,
    },
    "Bā anore ḥamḫūn (Entwurf)": {
        "aliases": ["baanorehamhun"],
        "source_type": "manuscript",
        "is_draft": True,
        "witness_alias": "Бо аноре ҳамхун",
    },
}

# key -> (canonical title, profile)
ALIAS_INDEX = {}
for _canon, _prof in WORK_PROFILES.items():
    for _a in _prof["aliases"]:
        ALIAS_INDEX[_a] = (_canon, _prof)
    ALIAS_INDEX.setdefault(title_key(_canon), (_canon, _prof))

DRAFT_MARKERS = ("draft", "entwurf", "черновик")


def looks_like_draft(title: str) -> bool:
    return any(m in (title or "").lower() for m in DRAFT_MARKERS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="tajik_corpus/corpus/master.json")
    ap.add_argument("--author", default="Dilorom Soliboeva",
                    help="Author for poems whose source carries none")
    ap.add_argument("--author-variants", default="Дилором Солибоева")
    ap.add_argument("--no-reanalyze", action="store_true",
                    help="Skip re-running the analyzer (keeps thin features)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    master_path = Path(args.master)
    if not master_path.exists():
        logger.error("Not found: %s", master_path)
        return 1

    master = json.loads(master_path.read_text(encoding="utf-8"))
    poems = master.get("poems", [])
    logger.info("Read %d poems from %s", len(poems), master_path)

    analyzer = None
    if not args.no_reanalyze:
        try:
            from analyzer import EnhancedTajikPoemAnalyzer
            analyzer = EnhancedTajikPoemAnalyzer()
            logger.info("Analyzer loaded — recomputing full analyses")
        except Exception as e:                                   # pragma: no cover
            logger.warning("Analyzer unavailable (%s); keeping stored summaries", e)

    corpus = Corpus()
    variants = [v.strip() for v in args.author_variants.split(",") if v.strip()]

    added = skipped = 0
    for poem in poems:
        text = poem.get("text") or ""
        if not text.strip():
            skipped += 1
            continue

        meta = poem.get("metadata", {}) or {}
        raw_title = meta.get("volume_title") or "Unknown source"
        key = title_key(raw_title)

        canonical, prof = ALIAS_INDEX.get(key, (raw_title, {}))
        profile = {k: v for k, v in prof.items()
                   if k not in ("aliases", "witness_alias")}
        if raw_title != canonical:
            profile["title_variants"] = [raw_title]

        if profile.get("is_draft") or looks_like_draft(raw_title):
            profile["is_draft"] = True
            profile.setdefault("source_type", "manuscript")
            profile["publication_year"] = None
            profile["composition_year"] = meta.get("volume_year")
            profile["composition_year_uncertain"] = True
            profile["notes"] = ("Draft witness supplied by the author; year "
                                "inherited from the printed edition and "
                                "therefore provisional.")
        else:
            profile.setdefault("publication_year", meta.get("volume_year"))

        analysis = {}
        if analyzer is not None:
            try:
                from dataclasses import asdict as _asdict
                result = analyzer.analyze_poem(text)
                analysis = json.loads(json.dumps(_asdict(result), default=str))
            except Exception as e:
                logger.warning("Analysis failed for %s: %s", poem.get("title"), e)

        if not analysis:
            analysis = {"structural": {}, "content": {},
                        "legacy_summary": poem.get("analysis_summary", {})}

        pid = corpus.add_poem(
            title=poem.get("title") or "Untitled",
            text=text,
            author=args.author,
            author_kw={"name_variants": variants},
            work_title=canonical,
            work_kw=profile,
            analysis=analysis,
            contributor=meta.get("contributor"),
            poem_id=poem.get("id"),
        )
        if pid:
            added += 1
        else:
            skipped += 1

    # Link draft witnesses to their printed counterparts via the alias table.
    works = corpus.data["works"]
    by_title = {w["title"]: wid for wid, w in works.items()}
    for canon, prof in WORK_PROFILES.items():
        target = prof.get("witness_alias")
        if not target:
            continue
        draft_id, printed_id = by_title.get(canon), by_title.get(target)
        if draft_id and printed_id:
            works[draft_id]["witness_of"] = printed_id
            logger.info("Draft '%s' linked as witness of '%s'", canon, target)

    stats = corpus.statistics()
    logger.info("Migrated: %d poems added, %d skipped", added, skipped)
    logger.info("Authors: %d | Works: %d", stats["authors"], stats["works"])
    for wid, w in works.items():
        logger.info("  %s | %s | %s | variants=%s",
                    w["title"], w.get("source_type"),
                    w.get("publication_year") or w.get("composition_year") or "n.d.",
                    w.get("title_variants") or "-")
    logger.info("Forms: %s", stats["by_form"])
    logger.info("Meter status: %s", stats["by_meter_status"])

    if args.dry_run:
        logger.info("Dry run — nothing written")
        return 0

    corpus.save()
    logger.info("Written: %s", corpus.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
