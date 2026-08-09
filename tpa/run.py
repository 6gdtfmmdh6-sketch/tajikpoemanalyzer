#!/usr/bin/env python3
"""
Easy startup script for Tajik Poetry Analyzer v2.0
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Simple startup script"""
    print("🚀 Starting Tajik Poetry Analyzer v2.0")
    
    # Check if main.py exists
    main_script = Path("main.py")
    if not main_script.exists():
        print("❌ main.py not found in current directory")
        print("   Please run this script from the project root directory")
        return 1
    
    # Check if setup was run
    config_dir = Path("config")
    if not config_dir.exists():
        print("📋 First time setup detected...")
        print("   Running setup script...")
        try:
            subprocess.check_call([sys.executable, "setup.py"])
            print("✅ Setup completed!")
        except subprocess.CalledProcessError:
            print("❌ Setup failed. Please run setup.py manually")
            return 1
    
    # Start the application
    print("🌐 Starting web interface...")
    print("   The application will open in your browser")
    print("   Press Ctrl+C to stop")
    print("")
    
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
