#!/usr/bin/env python3
"""
corpus_analysis — comparative and diachronic analysis over the corpus.

Replaces analysis_tool.py, which was a skeleton: three functions returning
empty dicts, `analyze_stylistic_evolution` defined twice (the second shadowed
the first), `Counter` used without being imported, and a data path
(`corpus["volumes"]`) that pointed at the library schema that was never
populated. Nothing imported it.

These are the operations a corpus is for: comparing groups and watching
features move over time. They work on the flat `features` rows written by
corpus_core, so nothing here re-runs the analyzer.

Caveat on statistics: with a handful of works by one author these are
descriptive, not inferential. The significance tests below are only
meaningful once the corpus holds several authors and enough poems per group;
`compare_groups` therefore reports group sizes alongside every result and
refuses tests below a minimum n.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional

from corpus_core import Corpus

logger = logging.getLogger(__name__)

MIN_GROUP_N = 5   # below this, report descriptives only

NUMERIC_FEATURES = [
    "lines", "words", "unique_words", "mtld",
    "syllables_mean", "syllables_variance",
    "line_length_min", "line_length_max",
    "enjambment_score", "prose_poetry_score", "line_variation_score",
    "neologism_count",
]


# ----------------------------------------------------------------- helpers

def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _stdev(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _median(xs: List[float]) -> Optional[float]:
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def describe(values: List[float]) -> Dict[str, Any]:
    return {"n": len([v for v in values if isinstance(v, (int, float))]),
            "mean": _mean(values), "median": _median(values),
            "sd": _stdev(values),
            "min": min((v for v in values if isinstance(v, (int, float))), default=None),
            "max": max((v for v in values if isinstance(v, (int, float))), default=None)}


def welch_t(a: List[float], b: List[float]) -> Optional[Dict[str, float]]:
    """Welch's t-test. Returns None when either group is too small.

    Deliberately dependency-free: scipy is not in requirements, and the old
    skeleton imported it anyway, which would have raised on first call.
    """
    a = [x for x in a if isinstance(x, (int, float))]
    b = [x for x in b if isinstance(x, (int, float))]
    if len(a) < MIN_GROUP_N or len(b) < MIN_GROUP_N:
        return None
    ma, mb = _mean(a), _mean(b)
    va, vb = (_stdev(a) or 0) ** 2, (_stdev(b) or 0) ** 2
    denom = math.sqrt(va / len(a) + vb / len(b))
    if denom == 0:
        return None
    t = (ma - mb) / denom
    # Welch–Satterthwaite degrees of freedom
    num = (va / len(a) + vb / len(b)) ** 2
    den = ((va / len(a)) ** 2 / (len(a) - 1)) + ((vb / len(b)) ** 2 / (len(b) - 1))
    df = num / den if den else float("nan")
    # Cohen's d with pooled sd — the effect size matters more than the p-value
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    d = (ma - mb) / pooled if pooled else None
    return {"t": round(t, 3), "df": round(df, 1),
            "mean_a": round(ma, 3), "mean_b": round(mb, 3),
            "cohens_d": round(d, 3) if d is not None else None,
            "n_a": len(a), "n_b": len(b)}


# ------------------------------------------------------------- group logic

def group_poems(corpus: Corpus, by: str) -> Dict[str, List[Dict]]:
    """Group poems by 'author', 'work', 'source_type', 'period', 'decade',
    'year' or 'draft_status'."""
    keyfns: Dict[str, Callable[[Dict], Any]] = {
        "author": lambda p: corpus.data["authors"].get(p["author_id"], {}).get("name"),
        "work": lambda p: corpus.work_of(p).get("title"),
        "source_type": lambda p: corpus.work_of(p).get("source_type"),
        "draft_status": lambda p: "draft" if corpus.work_of(p).get("is_draft") else "final",
        "year": lambda p: _work_year(corpus.work_of(p)),
        "decade": lambda p: (lambda y: f"{y // 10 * 10}s" if y else None)(_work_year(corpus.work_of(p))),
        "period": lambda p: corpus.work_of(p).get("period") or _period_of(corpus.work_of(p)),
    }
    if by not in keyfns:
        raise ValueError(f"Unknown grouping '{by}'. Options: {sorted(keyfns)}")
    out: Dict[str, List[Dict]] = defaultdict(list)
    for p in corpus.data["poems"]:
        key = keyfns[by](p)
        if key is not None:
            out[str(key)].append(p)
    return dict(out)


def _work_year(work: Dict) -> Optional[int]:
    return work.get("composition_year") or work.get("publication_year")


def _period_of(work: Dict) -> Optional[str]:
    from corpus_core import infer_period
    return infer_period(_work_year(work))


# -------------------------------------------------------------- operations

def profile_groups(corpus: Corpus, by: str = "work") -> Dict[str, Dict]:
    """Descriptive profile of every group: form counts, metre status and the
    numeric features. This is the workhorse — everything else builds on it."""
    result = {}
    for name, poems in sorted(group_poems(corpus, by).items()):
        feats = [p["features"] for p in poems]
        result[name] = {
            "n_poems": len(poems),
            "forms": dict(Counter(f.get("stanza_form") for f in feats)),
            "meter_status": dict(Counter(f.get("meter_status") for f in feats)),
            "meters": dict(Counter(f.get("meter") for f in feats
                                   if f.get("meter") and f["meter"] != "unknown")),
            "themes": dict(Counter(t for f in feats for t in (f.get("themes") or {}))),
            "radif_share": round(sum(1 for f in feats if f.get("has_radif")) / len(feats), 3),
            "numeric": {k: describe([f.get(k) for f in feats]) for k in NUMERIC_FEATURES},
        }
    return result


def compare_groups(corpus: Corpus, by: str, a: str, b: str) -> Dict[str, Any]:
    """Compare two groups feature by feature.

    This is the operation the thesis performs by hand between the draft and
    the printed edition, and the one a dissertation needs between authors.
    """
    groups = group_poems(corpus, by)
    for name in (a, b):
        if name not in groups:
            raise ValueError(f"No group '{name}' for grouping '{by}'. "
                             f"Available: {sorted(groups)}")
    fa = [p["features"] for p in groups[a]]
    fb = [p["features"] for p in groups[b]]

    out: Dict[str, Any] = {
        "grouping": by, "a": a, "b": b,
        "n_a": len(fa), "n_b": len(fb),
        "underpowered": len(fa) < MIN_GROUP_N or len(fb) < MIN_GROUP_N,
        "features": {},
        "form_shift": {
            a: dict(Counter(f.get("stanza_form") for f in fa)),
            b: dict(Counter(f.get("stanza_form") for f in fb)),
        },
    }
    for key in NUMERIC_FEATURES:
        va = [f.get(key) for f in fa]
        vb = [f.get(key) for f in fb]
        entry = {"a": describe(va), "b": describe(vb)}
        test = welch_t(va, vb)
        if test:
            entry["test"] = test
        out["features"][key] = entry
    return out


def evolution(corpus: Corpus, by: str = "year",
              features: Optional[List[str]] = None) -> List[Dict]:
    """Feature trajectories over time — the diachronic view.

    Returns one row per time bucket, ordered. With a single author and two
    dates this is a line between two points; it becomes meaningful as the
    corpus grows, which is the point of building it this way now.
    """
    features = features or ["mtld", "syllables_mean", "syllables_variance",
                            "enjambment_score", "lines"]
    rows = []
    for key, poems in group_poems(corpus, by).items():
        feats = [p["features"] for p in poems]
        row = {"bucket": key, "n_poems": len(poems)}
        for f in features:
            row[f] = _mean([x.get(f) for x in feats])
        row["free_verse_share"] = round(
            sum(1 for x in feats if x.get("meter_status") == "free_verse") / len(feats), 3)
        rows.append(row)
    rows.sort(key=lambda r: r["bucket"])
    return rows


def witness_divergence(corpus: Corpus) -> List[Dict]:
    """For every draft work, compare it against the printed work it witnesses.

    This is the editorial question of the MA thesis made repeatable: what
    changes between a draft and its published form, measured rather than
    asserted.
    """
    out = []
    for wid, work in corpus.data["works"].items():
        target = work.get("witness_of")
        if not target or target not in corpus.data["works"]:
            continue
        printed = corpus.data["works"][target]
        try:
            comparison = compare_groups(corpus, "work", work["title"], printed["title"])
        except ValueError as e:
            logger.warning("Skipping %s: %s", wid, e)
            continue
        shared = _shared_texts(corpus, wid, target)
        aligned = align_by_title(corpus, wid, target)
        comparison["shared_texts"] = len(shared)
        comparison["aligned_by_title"] = len(aligned)
        comparison["revised_pairs"] = [a for a in aligned if not a["identical_text"]]
        comparison["draft"] = work["title"]
        comparison["printed"] = printed["title"]
        out.append(comparison)
    return out


def align_by_title(corpus: Corpus, wid_a: str, wid_b: str) -> List[Dict]:
    """Pair poems across two works by normalized title.

    Attestation matching is exact-hash and therefore finds nothing between a
    draft and its edited printing — which is the whole point of the pair: the
    wording changed. Title alignment gives the poem-by-poem pairs an editorial
    comparison actually needs.
    """
    from corpus_core import title_key
    by_title_a, by_title_b = {}, {}
    for p in corpus.data["poems"]:
        works = {a["work_id"] for a in p.get("attestations", [])} or {p["work_id"]}
        key = title_key(p.get("title", ""))
        if not key:
            continue
        if wid_a in works:
            by_title_a.setdefault(key, p)
        if wid_b in works:
            by_title_b.setdefault(key, p)

    pairs = []
    for key in sorted(set(by_title_a) & set(by_title_b)):
        pa, pb = by_title_a[key], by_title_b[key]
        deltas = {}
        for f in NUMERIC_FEATURES:
            va, vb = pa["features"].get(f), pb["features"].get(f)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                deltas[f] = round(vb - va, 3)
        pairs.append({
            "title": pa.get("title"),
            "id_a": pa["id"], "id_b": pb["id"],
            "identical_text": pa["text_sha256"] == pb["text_sha256"],
            "delta": deltas,
        })
    return pairs


def _shared_texts(corpus: Corpus, wid_a: str, wid_b: str) -> List[str]:
    """Poem ids attested in both works."""
    shared = []
    for p in corpus.data["poems"]:
        works = {att["work_id"] for att in p.get("attestations", [])}
        if wid_a in works and wid_b in works:
            shared.append(p["id"])
    return shared


def attestation_report(corpus: Corpus) -> List[Dict]:
    """Poems carried by more than one source — the textual-criticism view."""
    out = []
    for p in corpus.data["poems"]:
        atts = p.get("attestations", [])
        if len(atts) > 1:
            out.append({
                "id": p["id"], "title": p.get("title"),
                "works": [corpus.data["works"].get(a["work_id"], {}).get("title", a["work_id"])
                          for a in atts],
                "titles_as_given": sorted({a.get("title_as_given") for a in atts}),
            })
    return out


if __name__ == "__main__":
    import json
    c = Corpus()
    print("== Profile by work ==")
    prof = profile_groups(c, "work")
    for name, d in prof.items():
        num = d["numeric"]
        print(f"  {name}: {d['n_poems']} poems | forms={d['forms']} | "
              f"MTLD median={num['mtld']['median']}")
    print("\n== Witness divergence ==")
    for comp in witness_divergence(c):
        print(f"  {comp['draft']} vs {comp['printed']} "
              f"(n={comp['n_a']}/{comp['n_b']}, identical={comp['shared_texts']}, "
              f"title-aligned={comp['aligned_by_title']}, "
              f"revised={len(comp['revised_pairs'])})")
        for k, v in comp["features"].items():
            if "test" in v:
                t = v["test"]
                print(f"    {k}: {t['mean_a']} vs {t['mean_b']} "
                      f"(d={t['cohens_d']})")
    print("\n== Multiply attested ==")
    for row in attestation_report(c)[:10]:
        print(f"  {row['title']}: {row['works']}")
