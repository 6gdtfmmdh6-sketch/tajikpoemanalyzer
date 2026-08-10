#!/usr/bin/env python3
"""Analysis — comparative and diachronic views over the corpus."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from corpus_core import Corpus
from corpus_analysis import (MIN_GROUP_N, align_by_title, attestation_report,
                             compare_groups, evolution, group_poems,
                             profile_groups, witness_divergence)

st.set_page_config(page_title="Analysis", page_icon="A", layout="wide")
st.title("Corpus analysis")

corpus = Corpus()
if not corpus.data["poems"]:
    st.info("Corpus is empty. Add poems on the Analyze page or run "
            "scripts/migrate_corpus.py.")
    st.stop()

tab_profile, tab_compare, tab_time, tab_witness = st.tabs(
    ["Profiles", "Compare", "Over time", "Witnesses"])

with tab_profile:
    by = st.selectbox("Group by", ["work", "author", "source_type", "period",
                                   "decade", "draft_status"], key="prof_by")
    prof = profile_groups(corpus, by)
    rows = []
    for name, d in prof.items():
        rows.append({
            "Group": name, "Poems": d["n_poems"],
            "Free verse": d["forms"].get("free_verse", 0),
            "With radīf": f"{d['radif_share']:.0%}",
            "MTLD (median)": (d["numeric"]["mtld"]["median"] or 0) and
                             round(d["numeric"]["mtld"]["median"], 1),
            "Syllables/line": (d["numeric"]["syllables_mean"]["mean"] or 0) and
                              round(d["numeric"]["syllables_mean"]["mean"], 1),
            "Lines (median)": d["numeric"]["lines"]["median"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with st.expander("Full numeric profile"):
        st.json(prof)

with tab_compare:
    by = st.selectbox("Group by", ["work", "author", "source_type",
                                   "draft_status", "period"], key="cmp_by")
    groups = sorted(group_poems(corpus, by))
    if len(groups) < 2:
        st.info(f"Need at least two groups for '{by}'. Currently: {groups}")
    else:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Group A", groups, index=0, key="cmp_a")
        b = c2.selectbox("Group B", groups, index=1, key="cmp_b")
        if a != b:
            res = compare_groups(corpus, by, a, b)
            if res["underpowered"]:
                st.warning(f"One group has fewer than {MIN_GROUP_N} poems — "
                           "descriptive figures only, no tests.")
            rows = []
            for feat, v in res["features"].items():
                t = v.get("test") or {}
                rows.append({
                    "Feature": feat,
                    f"{a} (mean)": round(v['a']['mean'], 2) if v['a']['mean'] else None,
                    f"{b} (mean)": round(v['b']['mean'], 2) if v['b']['mean'] else None,
                    "Cohen's d": t.get("cohens_d"),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Cohen's d is the effect size: roughly 0.2 small, 0.5 "
                       "medium, 0.8 large. With one author and few works, read "
                       "these as descriptive, not as evidence.")

with tab_time:
    by = st.selectbox("Time axis", ["year", "decade", "period"], key="evo_by")
    rows = evolution(corpus, by)
    df = pd.DataFrame(rows).set_index("bucket")
    st.dataframe(df, width="stretch")
    numeric = [c for c in df.columns if c != "n_poems" and df[c].notna().any()]
    if numeric:
        pick = st.multiselect("Plot", numeric,
                              default=[c for c in ("mtld", "syllables_mean") if c in numeric])
        if pick:
            st.line_chart(df[pick])
    st.caption("Works are placed by year of composition where recorded, "
               "otherwise by publication.")

with tab_witness:
    divergences = witness_divergence(corpus)
    if not divergences:
        st.info("No draft/printed pairs in the corpus. Mark a source as a draft "
                "on the Analyze page and link it via witness_of.")
    for comp in divergences:
        st.subheader(f"{comp['draft']} → {comp['printed']}")
        st.caption(f"{comp['n_a']} vs {comp['n_b']} poems · "
                   f"{comp['aligned_by_title']} paired by title · "
                   f"{len(comp['revised_pairs'])} of those differ in wording · "
                   f"{comp['shared_texts']} textually identical")
        pairs = comp["revised_pairs"]
        if pairs:
            st.dataframe(pd.DataFrame([
                {"Poem": p["title"],
                 "Δ lines": p["delta"].get("lines"),
                 "Δ MTLD": p["delta"].get("mtld"),
                 "Δ words": p["delta"].get("words"),
                 "Δ syllables/line": p["delta"].get("syllables_mean")}
                for p in pairs
            ]), width="stretch", hide_index=True)
            st.caption("Positive means the printed version is larger or more "
                       "diverse than the draft.")

    report = attestation_report(corpus)
    if report:
        with st.expander(f"Poems attested in more than one source ({len(report)})"):
            st.dataframe(pd.DataFrame([
                {"Poem": r["title"], "Sources": ", ".join(r["works"])} for r in report
            ]), width="stretch", hide_index=True)
