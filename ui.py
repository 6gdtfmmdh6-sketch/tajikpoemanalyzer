#!/usr/bin/env python3
"""
Schlichte Web-UI für Tajik Poetry Analyzer
Unterstützt PDF-Upload und Analyse
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil
from app2 import TajikPoemAnalyzer, AnalysisConfig, PoemData
from pdf_handler import read_file_with_pdf_support
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seiten-Konfiguration
st.set_page_config(
    page_title="Tajik Poetry Analyzer",
    page_icon="📖",
    layout="wide"
)

# CSS für schlichtes Design
st.markdown("""
<style>
    .main {max-width: 1200px; margin: 0 auto;}
    h1 {text-align: center; color: #2c3e50;}
    .stButton>button {width: 100%;}
    .metric-box {
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer():
    """Analyzer initialisieren (cached)"""
    config = AnalysisConfig(lexicon_path='data/tajik_lexicon.json')
    return TajikPoemAnalyzer(config=config)


def split_poems(text: str) -> list:
    """Text in Gedichte aufteilen"""
    # Nach ***** oder mehrfachen Leerzeilen trennen
    if '*****' in text:
        poems = [p.strip() for p in text.split('*****')]
    elif '\n\n\n' in text:
        poems = [p.strip() for p in text.split('\n\n\n')]
    else:
        poems = [p.strip() for p in text.split('\n\n')]

    return [p for p in poems if len(p) > 50]


def main():
    st.title("Tajik Poetry Analyzer")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Info")
        st.write("Wissenschaftliche Analyse tadschikischer/persischer Poesie")
        st.markdown("---")
        st.write("**Features:**")
        st.write("- Aruz-Metrik-Analyse")
        st.write("- Reimschema-Erkennung")
        st.write("- Phonetische Transkription")
        st.write("- Thematische Analyse")
        st.write("- PDF & OCR Unterstützung")

    # Hauptbereich
    st.header("Datei hochladen")

    uploaded_file = st.file_uploader(
        "PDF oder TXT hochladen",
        type=['pdf', 'txt'],
        help="Unterstützt normale und gescannte PDFs"
    )

    if uploaded_file is not None:
        # Temporäre Datei speichern
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        try:
            # Text extrahieren
            with st.spinner("Extrahiere Text aus Datei..."):
                text = read_file_with_pdf_support(tmp_path)
                st.success(f"Text extrahiert: {len(text)} Zeichen")

            # Text anzeigen
            with st.expander("Extrahierter Text anzeigen"):
                st.text_area("Inhalt", text, height=200)

            # Gedichte aufteilen
            poems = split_poems(text)
            st.info(f"Gefunden: {len(poems)} Gedichte")

            if st.button("Analyse starten", type="primary"):
                analyzer = load_analyzer()

                # Progress Bar
                progress_bar = st.progress(0)
                results_container = st.container()

                all_results = []

                for i, poem_text in enumerate(poems):
                    progress_bar.progress((i + 1) / len(poems))

                    try:
                        analysis = analyzer.analyze_poem(poem_text)
                        all_results.append({
                            'poem_text': poem_text,
                            'poem_num': i+1,
                            'analysis': analysis,
                            'success': True
                        })
                    except Exception as e:
                        logger.error(f"Fehler bei Gedicht {i+1}: {e}")
                        all_results.append({
                            'poem_text': poem_text,
                            'poem_num': i+1,
                            'error': str(e),
                            'success': False
                        })

                progress_bar.empty()

                # Ergebnisse anzeigen
                with results_container:
                    st.markdown("---")
                    st.header("Analyse-Ergebnisse")

                    # Übersicht
                    col1, col2, col3 = st.columns(3)
                    successful = sum(1 for r in all_results if r['success'])

                    with col1:
                        st.metric("Gedichte gesamt", len(all_results))
                    with col2:
                        st.metric("Erfolgreich", successful)
                    with col3:
                        st.metric("Fehlgeschlagen", len(all_results) - successful)

                    st.markdown("---")
                    # ZUSATZ IN ui.py (nach dem File-Upload, vor der Analyse)

if st.session_state.get('extracted_text'):
    st.header("📐 Gedichte trennen")
    
    st.markdown("""
    **Das System hat Vorschläge für Gedicht-Trennungen gemacht.**
    *   **Überprüfe** die roten Trennlinien im Text.
    *   **Verschiebe** sie per Slider.
    *   **Füge neue hinzu** oder **lösche** sie, indem du den Slider auf eine neue Position ziehst und auf "Trenner hinzufügen/entfernen" klickst.
    """)
    
    # 1. Automatische Vorschläge generieren (mit deiner EnhancedPoemSplitter-Logik)
    if 'splitters' not in st.session_state:
        config = TajikCyrillicConfig()
        splitter = EnhancedPoemSplitter(config)
        all_lines = st.session_state['extracted_text'].split('\n')
        
        # Einfache Heuristik für Start-Vorschläge: Leerzeilen finden
        proposed_split_indices = [i for i, line in enumerate(all_lines) if line.strip() == '']
        # Fallback: Gleichmäßig verteilen, falls keine Leerzeilen
        if not proposed_split_indices and len(all_lines) > 10:
            proposed_split_indices = list(range(10, len(all_lines), 20))
        
        st.session_state['splitters'] = proposed_split_indices
        st.session_state['all_lines'] = all_lines
    
    # 2. Interaktive Anzeige und Bearbeitung
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        st.subheader("Text mit Trennvorschlägen")
        display_text = ""
        for i, line in enumerate(st.session_state['all_lines']):
            # Füge eine markierte Trennlinie ein, wenn dieser Index in der Splitter-Liste ist
            if i in st.session_state['splitters']:
                display_text += f"\n--- 🟥 **TRENNER** (vor Zeile {i+1}) ---\n"
            display_text += line + "\n"
        st.text_area("Vorschau", display_text, height=400, key="display_area", disabled=True)
    
    with col_right:
        st.subheader("Trenner steuern")
        
        # Wähle einen Trenner zum Bearbeiten aus oder füge einen neuen hinzu
        all_positions = list(range(len(st.session_state['all_lines'])))
        current_splitters = st.session_state['splitters']
        
        # Slider zum Verschieben oder Auswählen einer neuen Position
        selected_position = st.slider(
            "Zeilenindex für Trenner",
            0,
            len(st.session_state['all_lines'])-1,
            value=0 if not current_splitters else min(current_splitters),
            key="splitter_slider"
        )
        
        col_add_remove, col_clear = st.columns(2)
        with col_add_remove:
            if selected_position in current_splitters:
                if st.button("Delete splitter"):
                    st.session_state['splitters'].remove(selected_position)
                    st.rerun()
            else:
                if st.button("Add splitter"):
                    st.session_state['splitters'].append(selected_position)
                    st.session_state['splitters'].sort()  # Ordnung halten
                    st.rerun()
        
        with col_clear:
            if st.button("Delete all"):
                st.session_state['splitters'] = []
                st.rerun()
        
        st.markdown("---")
        st.markdown(f"Aktuelle Trenner an Zeilen: **{', '.join(map(str, sorted(current_splitters)))}**")
        
        # 3. Bestätigen und zur Analyse übergehen
        if st.button("Trennung bestätigen & Analyse starten", type="primary"):
            # Text anhand der bestätigten Trenner aufteilen
            split_indices = sorted(st.session_state['splitters'])
            all_lines = st.session_state['all_lines']
            
            poems = []
            start_idx = 0
            for split_idx in split_indices:
                poem_lines = all_lines[start_idx:split_idx]
                poem_text = '\n'.join(poem_lines).strip()
                if poem_text:  # Leere "Gedichte" vermeiden
                    poems.append(poem_text)
                start_idx = split_idx
            # Letztes Gedicht hinzufügen
            final_poem = '\n'.join(all_lines[start_idx:]).strip()
            if final_poem:
                poems.append(final_poem)
            
            st.session_state['final_poems'] = poems
            st.success(f"Erfolgreich in {len(poems)} Gedicht(e) geteilt. Starte Analyse...")
            # Setze einen Flag, um mit der Analyse fortzufahren
            st.session_state['proceed_to_analysis'] = True

# Später im Code: Wenn 'proceed_to_analysis' True ist, analysiere jedes Gedicht in st.session_state['final_poems']

                    # Einzelne Gedichte
                    for result in all_results:
                        if not result['success']:
                            st.error(f"Gedicht {result['poem_num']}: {result['error']}")
                            continue

                        analysis = result['analysis']
                        poem_text = result['poem_text']
                        poem_num = result['poem_num']

                        with st.expander(f"Gedicht {poem_num} - {len(poem_text.split())} Wörter"):
                            # Inhalt
                            st.subheader("Inhalt")
                            st.text(poem_text[:300] + "..." if len(poem_text) > 300 else poem_text)

                            st.markdown("---")

                            # Strukturelle Analyse
                            st.subheader("Strukturelle Analyse")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write(f"**Zeilen:** {analysis.structural.lines}")
                                st.write(f"**Silben/Zeile:** {analysis.structural.avg_syllables:.1f}")
                                st.write(f"**Strophenform:** {analysis.structural.stanza_structure}")

                            with col2:
                                if hasattr(analysis.structural, 'aruz_analysis'):
                                    st.write(f"**Metrum:** {analysis.structural.aruz_analysis.identified_meter}")
                                    st.write(f"**Konfidenz:** {analysis.structural.aruz_analysis.confidence.value}")
                                st.write(f"**Reimschema:** {analysis.structural.rhyme_pattern}")

                            st.markdown("---")

                            # Inhaltliche Analyse
                            st.subheader("Inhaltliche Analyse")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Häufigste Wörter:**")
                                for word, count in analysis.content.word_frequencies[:5]:
                                    st.write(f"- {word}: {count}x")

                            with col2:
                                st.write("**Themen:**")
                                themes = [k for k, v in analysis.content.theme_distribution.items() if v > 0]
                                if themes:
                                    for theme in themes:
                                        st.write(f"- {theme}")
                                else:
                                    st.write("Keine Themen erkannt")

                            if analysis.content.neologisms:
                                st.write(f"**Neologismen:** {', '.join(analysis.content.neologisms[:5])}")

                            if analysis.content.archaisms:
                                st.write(f"**Archaismen:** {', '.join(analysis.content.archaisms[:5])}")

                            st.markdown("---")

                            # Qualität
                            if hasattr(analysis, 'quality_metrics'):
                                st.subheader("Qualität")
                                quality_cols = st.columns(4)
                                metrics = list(analysis.quality_metrics.items())[:4]
                                for col, (metric, score) in zip(quality_cols, metrics):
                                    if isinstance(score, (int, float)):
                                        col.metric(metric.replace('_', ' ').title(), f"{score:.2f}")

                st.success("Analyse abgeschlossen")

        finally:
            # Temporäre Datei löschen
            if tmp_path.exists():
                tmp_path.unlink()

    else:
        st.info("Bitte laden Sie eine PDF- oder TXT-Datei hoch, um zu beginnen.")


if __name__ == "__main__":
    main()
