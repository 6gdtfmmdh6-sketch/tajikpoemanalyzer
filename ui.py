#!/usr/bin/env python3
"""
Tajik Poetry Analyzer - Auto-redirect to Analyze page
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Tajik Poetry Analyzer",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-redirect to Analyze page
st.switch_page("pages/1_Analyze.py")
