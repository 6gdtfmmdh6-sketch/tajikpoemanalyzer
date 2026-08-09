#!/usr/bin/env python3
"""Export page — knowledge snapshot (private) and feature export (public)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from knowledge_export import (export_public_features, export_snapshot,
                              restore_snapshot, MASTER)

st.set_page_config(page_title="Export", page_icon="T", layout="wide")
st.title("Export")

st.markdown(
    """
Zwei getrennte Wege, bewusst getrennt gehalten:

| | Enthält Volltexte? | Bestimmung |
|---|---|---|
| **Feature-Export** | Nein — nur SHA-256, Incipit, Analysen | Öffentlich teilbar (GitHub, Zenodo, Kolleg:innen) |
| **Wissens-Snapshot** | **Ja — kompletter Forschungsstand** | Nur lokal / Backup / geschlossener Forschungskreis |
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature-Export (öffentlich)")
    st.caption("Abgeleitete Daten ohne Gedichttexte — urheberrechtsfrei teilbar.")
    if st.button("Feature-Export erzeugen"):
        if not MASTER.exists():
            st.error("Kein master.json gefunden — Korpus ist leer.")
        else:
            out = export_public_features()
            st.success(f"Erzeugt: {out.name}")
            st.download_button(
                "features_public.json herunterladen",
                data=out.read_bytes(),
                file_name=out.name,
                mime="application/json",
            )

with col2:
    st.subheader("Wissens-Snapshot (privat)")
    st.warning(
        "Enthält sämtliche Volltexte. Nicht hochladen, nicht ins Git, "
        "nicht öffentlich ablegen."
    )
    if st.button("Snapshot erzeugen"):
        out = export_snapshot()
        st.success(f"Erzeugt: {out.name} ({out.stat().st_size/1e6:.1f} MB)")
        st.download_button(
            "Snapshot herunterladen",
            data=out.read_bytes(),
            file_name=out.name,
            mime="application/gzip",
        )

st.divider()
st.subheader("Snapshot wiederherstellen")
st.caption("Umzug auf einen anderen Rechner: Snapshot hochladen, Zustand wird übernommen.")
uploaded = st.file_uploader("knowledge_snapshot_*.tar.gz", type=["gz"])
if uploaded is not None and st.button("Wiederherstellen"):
    tmp = Path("uploads") / uploaded.name
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_bytes(uploaded.getbuffer())
    try:
        restore_snapshot(tmp)
        st.success("Snapshot wiederhergestellt. Seite neu laden.")
    except Exception as e:
        st.error(f"Fehler: {e}")
