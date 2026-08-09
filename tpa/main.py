#!/usr/bin/env python3
"""
Main Application Entry Point for Tajik Poetry Analyzer v2.0
Production-Ready with CLI and UI modes
"""

import sys
import os
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Optional, List, Any
import argparse
import asyncio

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from secure_config import get_config, set_config, AppConfig, Environment
from improved_analyzer import ProductionAnalyzer, BatchConfig, EnhancedCorpusManager
from enhanced_ui import main as run_streamlit_ui

def setup_logging(config: AppConfig):
    """Setup logging configuration"""
    log_config = config.logging
    
    # Create logs directory
    log_file = Path(log_config.log_file)
    log_file.parent.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_config.level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_config.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

class ApplicationManager:
    """Manages the application lifecycle"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.shutdown_event = threading.Event()
        self.analyzer: Optional[ProductionAnalyzer] = None
        self.corpus_manager: Optional[EnhancedCorpusManager] = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()

    def initialize(self) -> None:
        """Initialize application components"""
        self.logger.info("Initializing Tajik Poetry Analyzer v2.0...")
        
        try:
            # Initialize batch configuration
            batch_config = BatchConfig(
                max_text_length=self.config.security.max_file_size_mb * 1024 * 1024,
                batch_size=self.config.processing.default_batch_size,
                max_concurrent=self.config.security.max_concurrent_analyses,
                memory_limit_mb=self.config.security.max_memory_usage_mb,
                enable_streaming=self.config.processing.enable_streaming
            )
            
            # Initialize analyzer
            self.analyzer = ProductionAnalyzer(batch_config)
            self.logger.info("✅ Production analyzer initialized")
            
            # Initialize corpus manager
            db_path = self.config.database.get_safe_db_path()
            self.corpus_manager = EnhancedCorpusManager(str(db_path), batch_config)
            self.logger.info(f"✅ Database initialized at {db_path}")
            
            # Validate system requirements
            self._validate_system()
            
            self.logger.info("🚀 Application initialization complete")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize application: {e}")
            raise
    
    def _validate_system(self) -> None:
        """Validate system requirements"""
        try:
            import psutil
            
            # Check available memory
            available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
            required_memory = self.config.security.max_memory_usage_mb
            
            if available_memory < required_memory * 1.5:
                self.logger.warning(
                    f"⚠️ Low system memory: {available_memory:.0f}MB available, "
                    f"{required_memory}MB required"
                )
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / 1024 / 1024 / 1024
            
            if free_space_gb < 5:
                self.logger.warning(f"⚠️ Low disk space: {free_space_gb:.1f}GB available")
            
        except ImportError:
            self.logger.warning("psutil not available, skipping system validation")
        
        # Create necessary directories
        directories = ['data', 'exports', 'logs', 'temp']
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
        
        self.logger.info("✅ System validation complete")
    
    def run_cli_mode(self, args: argparse.Namespace) -> int:
        """Run in CLI mode for batch processing"""
        self.logger.info("Starting CLI batch processing mode")
        
        if not args.input_files:
            self.logger.error("No input files specified")
            return 1
        
        # Validate input files
        input_paths = []
        total_size = 0
        
        for file_path in args.input_files:
            path = Path(file_path)
            if not path.exists():
                self.logger.error(f"File not found: {file_path}")
                return 1
            
            file_size = path.stat().st_size
            total_size += file_size
            
            # Validate file
            allowed, message = self.config.is_file_allowed(path.name, file_size)
            if not allowed:
                self.logger.error(f"File {file_path} not allowed: {message}")
                return 1
            
            input_paths.append(path)
        
        # Validate batch
        allowed, message = self.config.is_batch_allowed(len(input_paths), total_size)
        if not allowed:
            self.logger.error(f"Batch not allowed: {message}")
            return 1
        
        # Process files
        try:
            results = asyncio.run(self._process_batch_cli(input_paths, args))
            
            # Save results
            if args.output:
                output_path = Path(args.output)
                self.analyzer.save_batch_results(
                    results['results'], 
                    output_path, 
                    args.format
                )
                self.logger.info(f"✅ Results saved to {output_path}")
            
            # Print summary
            summary = results['summary']
            self.logger.info(f"""
📊 Batch Processing Complete:
   Total poems: {summary['total_poems']}
   Successful: {summary['successful']}
   Failed: {summary['failed']}
   Success rate: {summary['success_rate']:.1%}
""")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            return 1
    
    async def _process_batch_cli(self, input_paths: List[Path], args):
        """Process batch in CLI mode"""
        def progress_callback(progress):
            if progress.processed_poems % 10 == 0:
                self.logger.info(
                    f"Progress: {progress.completion_rate:.1%} "
                    f"({progress.processed_poems}/{progress.total_poems}) - "
                    f"Success rate: {progress.success_rate:.1%}"
                )
        
        return await self.analyzer.analyze_batch_files(input_paths, progress_callback)
    
    def run_ui_mode(self) -> None:
        """Run in UI mode with Streamlit"""
        self.logger.info("Starting Streamlit UI mode")
        
        try:
            # Set global config for UI
            set_config(self.config)
            
            # Run Streamlit
            run_streamlit_ui()
            
        except KeyboardInterrupt:
            self.logger.info("UI mode interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ UI mode failed: {e}")
            raise
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.logger.info("Cleaning up application resources...")
        
        if hasattr(self.corpus_manager, 'close'):
            try:
                self.corpus_manager.close()
            except:
                pass
        
        self.logger.info("✅ Cleanup complete")

def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Tajik Poetry Analyzer v2.0 - Advanced batch processing for poetry analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run UI mode (default)
  python main.py
  
  # Batch process files
  python main.py --mode cli --input file1.txt file2.txt --output results.json
  
  # Process with custom config
  python main.py --config production_config.json
  
  # Set environment
  ANALYZER_ENV=production python main.py
"""
    )
    
    parser.add_argument(
        '--mode',
        choices=['ui', 'cli'],
        default='ui',
        help='Application mode (default: ui)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--input',
        dest='input_files',
        nargs='+',
        help='Input files for batch processing (CLI mode only)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (CLI mode only)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'excel', 'csv'],
        default='json',
        help='Output format (CLI mode only)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Tajik Poetry Analyzer v2.0.0'
    )
    
    return parser

def main():
    """Main application entry point"""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = AppConfig.from_file(args.config)
        else:
            config = get_config()
        
        # Override log level if verbose
        if args.verbose:
            config.logging.level = "DEBUG"
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info(f"🚀 Starting Tajik Poetry Analyzer v2.0.0")
        logger.info(f"Environment: {config.environment.value}")
        logger.info(f"Mode: {args.mode}")
        
        # Initialize application
        app = ApplicationManager(config)
        app.initialize()
        
        # Run application based on mode
        if args.mode == 'cli':
            exit_code = app.run_cli_mode(args)
        else:
            app.run_ui_mode()
            exit_code = 0
        
        # Cleanup
        app.cleanup()
        
        logger.info("👋 Application shutdown complete")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️ Application interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.exception("Fatal error occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()
