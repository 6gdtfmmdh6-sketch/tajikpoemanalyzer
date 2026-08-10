#!/usr/bin/env python3
"""Timeline — the corpus arranged chronologically.

Places every work on the year it was written where that is known, and on the
year it was published otherwise. The distinction matters: for samizdat and
manuscript circulation the two can lie decades apart, and a timeline built on
publication dates alone misrepresents when the writing happened.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from corpus_core import Corpus

st.set_page_config(page_title="Timeline", page_icon="T", layout="wide")
st.title("Timeline")

corpus = Corpus()
timeline = corpus.timeline()
stats = corpus.statistics()

if not timeline:
    st.info("No dated works in the corpus yet. Add poems with an author and a "
            "source on the Analyze page, or run scripts/migrate_corpus.py.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Authors", stats["authors"])
c2.metric("Works", stats["works"])
c3.metric("Poems", stats["poems"])
c4.metric("Attested in >1 work", stats["multiply_attested"],
          help="Same text carried by more than one source — a textual-criticism "
               "finding the old corpus discarded as a duplicate.")

st.subheader("Chronology")
rows = []
for bucket in timeline:
    for work in bucket["works"]:
        rows.append({
            "Year": bucket["year"],
            "Period": bucket["period"],
            "Work": work["title"] + (" (draft)" if work["is_draft"] else ""),
            "Source type": work["source_type"],
            "Dated by": work["dated_by"],
            "Authors": ", ".join(bucket["authors"]),
        })
df = pd.DataFrame(rows)
st.dataframe(df, width="stretch", hide_index=True)

st.caption("‘Dated by: composition’ means the year of writing is recorded; "
           "‘publication’ means only the print date is known and the writing "
           "may be earlier.")

st.subheader("Poems per year")
counts = pd.DataFrame(
    [{"Year": b["year"], "Poems": b["poem_count"]} for b in timeline]
).set_index("Year")
st.bar_chart(counts)

st.subheader("Distribution")
d1, d2 = st.columns(2)
with d1:
    st.write("**By source type**")
    st.dataframe(pd.DataFrame(stats["by_source_type"].items(),
                              columns=["Source type", "Poems"]),
                 width="stretch", hide_index=True)
with d2:
    st.write("**By meter status**")
    st.dataframe(pd.DataFrame(stats["by_meter_status"].items(),
                              columns=["Status", "Poems"]),
                 width="stretch", hide_index=True)
    st.caption("‘free_verse’ = no metre by design; ‘detection_failed’ = the "
               "detector gave up. The old corpus called both ‘unknown’.")

with st.expander("Works in detail"):
    for wid, w in corpus.data["works"].items():
        st.markdown(f"**{w['title']}** · `{wid}`")
        bits = [f"type: {w.get('source_type')}"]
        if w.get("publication_year"):
            bits.append(f"published {w['publication_year']}")
        if w.get("composition_year"):
            bits.append(f"written {w['composition_year']}"
                        + (" (uncertain)" if w.get("composition_year_uncertain") else ""))
        if w.get("witness_of"):
            bits.append(f"draft witness of `{w['witness_of']}`")
        if w.get("title_variants"):
            bits.append("also recorded as: " + ", ".join(w["title_variants"]))
        st.caption(" · ".join(bits))
        if w.get("notes"):
            st.caption(w["notes"])
