#!/usr/bin/env python3
"""
Knowledge export for the Tajik Poetry Analyzer.

Two-layer data policy:

  LAYER 1 — PRIVATE (never online):
    Full poem texts: master.json with `text` fields, data/*.txt volumes,
    tajik_corpus/contributions/, the poetry library. These stay local and
    are covered by .gitignore and .dockerignore. The complete research
    state can be exported as a portable snapshot archive
    (`export_snapshot`) for backup or hand-over to collaborators within a
    defined research group (§ 60d UrhG scope) — by file, not by repo.

  LAYER 2 — PUBLIC (shareable):
    Derived features only (`export_public_features`): stable SHA-256 text
    hashes, incipit (first line, max. 60 chars, for scholarly
    identification), bibliographic metadata and the full analysis layer
    (meter, rhyme, radif words, syllable counts, MTLD, themes, ...).
    Texts cannot be reconstructed from this export.

Usage (CLI):
    python knowledge_export.py features   # -> tajik_corpus/exports/features_public.json
    python knowledge_export.py snapshot   # -> snapshots/knowledge_snapshot_<date>.tar.gz
    python knowledge_export.py restore snapshots/knowledge_snapshot_2026-08-10.tar.gz
"""

import hashlib
import json
import logging
import sys
import tarfile
import unicodedata
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("knowledge_export")

ROOT = Path(__file__).parent
MASTER = ROOT / "tajik_corpus" / "corpus" / "master.json"
EXPORT_DIR = ROOT / "tajik_corpus" / "exports"
SNAPSHOT_DIR = ROOT / "snapshots"

# Everything that constitutes the private research state.
SNAPSHOT_PATHS = [
    "tajik_corpus",
    "tajik_poetry_library",
    "data/dilorom_*.txt",   # all textual witnesses (T, B, E, K)
    "data/poems.txt",
    "data/shahnama.txt",
    "data/themes.json",
    "exports",
]

# Excluded from snapshots by default: derivable or huge (word list ~400 MB).
SNAPSHOT_EXCLUDE_LARGE = ["data/tajik_corpus.txt", "data/tajik_lexicon.json"]

# Fields that may never appear in a public export.
PRIVATE_POEM_FIELDS = {"text", "raw_text", "normalized_text", "lines", "body"}


def _normalize(text: str) -> str:
    """Normalization used for stable hashing: NFC, collapsed whitespace."""
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.split())


def text_hash(text: str) -> str:
    """Stable identifier of a poem text without revealing it."""
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _incipit(text: str, max_len: int = 60) -> str:
    """First non-title line, truncated — standard scholarly identification."""
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    if not lines:
        return ""
    body = lines[1] if len(lines) > 1 else lines[0]
    return body[:max_len]


def _strip_private(obj):
    """Recursively remove private text fields from nested structures."""
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items()
                if k not in PRIVATE_POEM_FIELDS}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def export_public_features(master_path: Path = MASTER,
                           out_path: Optional[Path] = None) -> Path:
    """Create the copyright-safe public feature export from master.json."""
    if not master_path.exists():
        raise FileNotFoundError(f"{master_path} not found")

    master = json.loads(master_path.read_text(encoding="utf-8"))
    poems_out = []
    for poem in master.get("poems", []):
        full_text = poem.get("text", "")
        clean = _strip_private(poem)
        clean["text_sha256"] = text_hash(full_text) if full_text else None
        clean["incipit"] = _incipit(full_text)
        poems_out.append(clean)

    public = {
        "metadata": {
            **master.get("metadata", {}),
            "export_type": "public_features",
            "export_date": datetime.now().isoformat(timespec="seconds"),
            "note": ("Derived data only. Poem texts are identified by "
                     "SHA-256 hash and incipit; full texts are not part "
                     "of this export."),
            "license": "CC-BY-4.0 (derived data)",
        },
        "statistics": master.get("statistics", {}),
        "vocabulary": master.get("vocabulary", {}),
        "aruz_distribution": master.get("aruz_distribution", {}),
        "theme_distribution": master.get("theme_distribution", {}),
        "radif_collection": master.get("radif_collection", {}),
        "poems": poems_out,
    }

    out_path = out_path or EXPORT_DIR / "features_public.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(public, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    # Hard guarantee: no private field survived.
    dumped = out_path.read_text(encoding="utf-8")
    parsed = json.loads(dumped)
    for poem in parsed["poems"]:
        leaked = PRIVATE_POEM_FIELDS & set(poem)
        if leaked:
            out_path.unlink()
            raise RuntimeError(f"Private fields leaked into export: {leaked}")

    logger.info("Public feature export: %s (%d poems, no full texts)",
                out_path, len(poems_out))
    return out_path


def export_snapshot(out_dir: Path = SNAPSHOT_DIR) -> Path:
    """Bundle the complete private research state into one archive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    out_path = out_dir / f"knowledge_snapshot_{stamp}.tar.gz"

    manifest = {"created": datetime.now().isoformat(timespec="seconds"),
                "contents": [], "tool": "knowledge_export.py"}

    with tarfile.open(out_path, "w:gz") as tar:
        for pattern in SNAPSHOT_PATHS:
            matches = sorted(ROOT.glob(pattern)) if "*" in pattern else [ROOT / pattern]
            if "*" in pattern and not matches:
                logger.warning("No match for snapshot pattern: %s", pattern)
            for p in matches:
                if not p.exists():
                    logger.warning("Missing from snapshot: %s", p.name)
                    continue
                rel_name = str(p.relative_to(ROOT))
                tar.add(p, arcname=rel_name)
                manifest["contents"].append(rel_name)
        manifest_bytes = json.dumps(manifest, indent=1).encode("utf-8")
        info = tarfile.TarInfo("SNAPSHOT_MANIFEST.json")
        info.size = len(manifest_bytes)
        import io
        tar.addfile(info, io.BytesIO(manifest_bytes))

    logger.info("Knowledge snapshot: %s (%.1f MB)", out_path,
                out_path.stat().st_size / 1e6)
    return out_path


def restore_snapshot(archive: Path, target: Path = ROOT) -> None:
    """Restore a snapshot archive into the working directory."""
    archive = Path(archive)
    if not archive.exists():
        raise FileNotFoundError(archive)
    with tarfile.open(archive, "r:gz") as tar:
        # Refuse path traversal.
        for member in tar.getmembers():
            member_path = (target / member.name).resolve()
            if not str(member_path).startswith(str(target.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(target)
    logger.info("Snapshot restored into %s", target)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "features"
    if cmd == "features":
        export_public_features()
    elif cmd == "snapshot":
        export_snapshot()
    elif cmd == "restore" and len(sys.argv) > 2:
        restore_snapshot(Path(sys.argv[2]))
    else:
        print(__doc__)
        sys.exit(1)
