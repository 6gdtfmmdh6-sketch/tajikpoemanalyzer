#!/usr/bin/env python3
"""
Simple runner for Tajik Poetry Analyzer
"""

import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        'streamlit',
        'openpyxl',
        'pandas',
        'numpy',
        'PyPDF2',
        'pdf2image',
        'pytesseract',
        'Pillow'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    return missing

def main():
    print("Tajik Poetry Analyzer")
    print("=" * 50)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("pip install streamlit openpyxl pandas numpy")
        print("pip install PyPDF2 pdf2image pytesseract Pillow")
        return
    
    # Check for analyzer.py
    if not Path("analyzer.py").exists():
        print("Error: analyzer.py not found in current directory")
        return
    
    # Run the fixed app
    print("\nStarting Streamlit app...")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app_fixed.py", "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
