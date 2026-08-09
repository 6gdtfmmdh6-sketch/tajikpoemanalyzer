#!/usr/bin/env python3
"""
Setup and Installation Script for Tajik Poetry Analyzer v2.0
"""

import sys
import os
import json
import logging
from pathlib import Path
import subprocess
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        logger.error(f"Current version: {sys.version}")
        return False
    
    logger.info(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    logger.info("Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directory structure...")
    
    directories = [
        "data/corpus",
        "data/lexicons", 
        "data/models",
        "exports/results",
        "exports/reports",
        "logs",
        "temp",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created: {directory}")
    
    return True

def create_default_config():
    """Create default configuration files"""
    logger.info("Creating default configuration...")
    
    # Development config
    dev_config = {
        "environment": "development",
        "debug_mode": True,
        "security": {
            "max_file_size_mb": 50,
            "max_files_per_batch": 100,
            "max_batch_size_mb": 1024,
            "max_memory_usage_mb": 1024,
            "max_concurrent_analyses": 2,
            "timeout_per_poem_seconds": 30,
            "allowed_file_extensions": [".txt", ".docx", ".pdf", ".rtf"],
            "enable_content_scanning": True
        },
        "database": {
            "db_path": "data/corpus/poetry_corpus.db",
            "backup_enabled": True,
            "backup_interval_hours": 24
        },
        "processing": {
            "default_batch_size": 25,
            "max_batch_size": 100,
            "enable_streaming": True,
            "chunk_size_mb": 5,
            "enable_parallel_processing": True
        },
        "logging": {
            "level": "INFO",
            "log_file": "logs/analyzer.log",
            "max_file_size_mb": 10,
            "enable_json_logging": False
        }
    }
    
    # Save development config
    with open("config/development.json", "w", encoding="utf-8") as f:
        json.dump(dev_config, f, indent=2, ensure_ascii=False)
    
    # Production config (more restrictive)
    prod_config = dev_config.copy()
    prod_config.update({
        "environment": "production",
        "debug_mode": False,
        "security": {
            **dev_config["security"],
            "max_memory_usage_mb": 2048,
            "max_concurrent_analyses": 4
        },
        "logging": {
            **dev_config["logging"],
            "level": "WARNING",
            "enable_json_logging": True
        }
    })
    
    with open("config/production.json", "w", encoding="utf-8") as f:
        json.dump(prod_config, f, indent=2, ensure_ascii=False)
    
    logger.info("✅ Configuration files created")
    return True

def create_default_lexicon():
    """Create a basic Tajik lexicon file"""
    logger.info("Creating default lexicon...")
    
    # Basic Tajik words for initial testing
    basic_lexicon = [
        # Common words
        "ман", "ту", "ӯ", "мо", "шумо", "онҳо",
        "аст", "буд", "хоҳад", "мехоҳам", "мебошад",
        
        # Poetry terms
        "шеър", "газал", "рубоӣ", "қасида", "байт",
        "қофия", "баҳр", "зарб", "радиф",
        
        # Common themes
        "муҳаббат", "ишқ", "дил", "ҷон", "ҷонон",
        "ватан", "диёр", "кишвар", "хона", "манзил",
        "гул", "булбул", "наргис", "сарв", "чаман",
        "оби", "дарё", "баҳр", "кӯҳ", "осмон",
        "хуршед", "моҳ", "ситора", "нур", "зулмат",
        
        # Religious/mystical terms
        "худо", "аллоҳ", "пайғамбар", "ҷаннат", "дӯзах",
        "тариқат", "ҳақиқат", "мақом", "ҳол", "вақт",
        
        # Emotions
        "хушӣ", "ғам", "андӯҳ", "шодӣ", "ҳайрат",
        "умед", "бим", "таваккал", "сабр", "интизор"
    ]
    
    lexicon_path = Path("data/lexicons/tajik_lexicon.json")
    with open(lexicon_path, "w", encoding="utf-8") as f:
        json.dump(basic_lexicon, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Basic lexicon created: {lexicon_path}")
    return True

def create_test_data():
    """Create sample test data"""
    logger.info("Creating test data...")
    
    # Sample Tajik poems for testing
    test_poems = [
        {
            "title": "Рубоии аввал",
            "content": """Ай дили ман, ту беқарорӣ,
Дар ҳама ҳол интизорӣ.
Гарчи бисёр шуд ғаму андӯҳ,
Боз ҳам умедворӣ."""
        },
        {
            "title": "Дар ситоиши ватан", 
            "content": """Ватани азизи ман, Тоҷикистон,
Зебои ту намешавад ба забон.
Кӯҳу дарёи ту чӯн қасри ҷаҳон,
Ҳар варакаш ғазали дилкашон."""
        },
        {
            "title": "Ба ёди дӯст",
            "content": """Рафт дӯсти дили ман ба сафар,
Монд аз ӯ дар дил танҳо хабар.
Гар кунад як рӯз бозомад ба мо,
Хонаро гулшан кунам зери қадам."""
        }
    ]
    
    # Save as individual files
    test_dir = Path("data/test_poems")
    test_dir.mkdir(exist_ok=True)
    
    for i, poem in enumerate(test_poems, 1):
        filename = f"test_poem_{i:02d}.txt"
        file_path = test_dir / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"{poem['title']}\n\n{poem['content']}")
        
        logger.info(f"✅ Created test file: {filename}")
    
    # Create a batch test file
    batch_file = test_dir / "batch_poems.txt"
    with open(batch_file, "w", encoding="utf-8") as f:
        for poem in test_poems:
            f.write(f"{poem['title']}\n\n{poem['content']}\n\n")
            f.write("="*50 + "\n\n")
    
    logger.info("✅ Test data created")
    return True

def validate_installation():
    """Validate that installation was successful"""
    logger.info("Validating installation...")
    
    try:
        # Test imports
        from secure_config import get_config
        from improved_analyzer import ProductionAnalyzer, BatchConfig
        from analyzer import TajikPoemAnalyzer, PoemData
        
        logger.info("✅ Core modules import successfully")
        
        # Test configuration
        config = get_config()
        logger.info(f"✅ Configuration loaded: {config.environment.value}")
        
        # Test basic analyzer creation
        analyzer = TajikPoemAnalyzer()
        logger.info("✅ Analyzer creates successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Setting up Tajik Poetry Analyzer v2.0")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing requirements", install_requirements),
        ("Creating directories", create_directories),
        ("Creating configuration", create_default_config),
        ("Creating lexicon", create_default_lexicon),
        ("Creating test data", create_test_data),
        ("Validating installation", validate_installation)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"📋 {step_name}...")
        
        try:
            if not step_func():
                logger.error(f"❌ Setup failed at: {step_name}")
                return 1
        except Exception as e:
            logger.error(f"❌ Setup failed at {step_name}: {e}")
            return 1
    
    logger.info("""
🎉 Setup completed successfully!

Next steps:
1. Run the application: python main.py
2. Or use CLI mode: python main.py --mode cli --input data/test_poems/test_poem_01.txt
3. Or run tests: python -m pytest tests/ -v

Configuration files:
- Development: config/development.json
- Production: config/production.json

Test data available in: data/test_poems/

For more information, see README.md
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
