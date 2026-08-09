#!/usr/bin/env python3
"""
Secure Configuration Management for Tajik Poetry Analyzer
Production-ready configuration with validation and security features
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tempfile

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class SecurityConfig:
    """Security-related configuration"""
    max_file_size_mb: int = 100
    max_files_per_batch: int = 1000
    max_batch_size_mb: int = 10240  # 10GB
    max_memory_usage_mb: int = 2048
    max_concurrent_analyses: int = 4
    timeout_per_poem_seconds: int = 30
    allowed_file_extensions: List[str] = field(default_factory=lambda: ['.txt', '.docx', '.pdf', '.rtf'])
    blocked_file_patterns: List[str] = field(default_factory=lambda: ['*.exe', '*.bat', '*.sh', '*.cmd'])
    enable_content_scanning: bool = True
    max_poem_length_chars: int = 50000
    
    def validate(self) -> List[str]:
        """Validate security configuration"""
        errors = []
        
        if self.max_file_size_mb <= 0 or self.max_file_size_mb > 500:
            errors.append("max_file_size_mb must be between 1 and 500")
        
        if self.max_files_per_batch <= 0 or self.max_files_per_batch > 5000:
            errors.append("max_files_per_batch must be between 1 and 5000")
        
        if self.max_memory_usage_mb < 512 or self.max_memory_usage_mb > 16384:
            errors.append("max_memory_usage_mb must be between 512 and 16384")
        
        if self.max_concurrent_analyses <= 0 or self.max_concurrent_analyses > 16:
            errors.append("max_concurrent_analyses must be between 1 and 16")
        
        if not self.allowed_file_extensions:
            errors.append("allowed_file_extensions cannot be empty")
        
        return errors

@dataclass 
class DatabaseConfig:
    """Database configuration"""
    db_path: str = "data/corpus/poetry_corpus.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_connections: int = 10
    connection_timeout: int = 30
    enable_wal_mode: bool = True
    
    def get_safe_db_path(self) -> Path:
        """Get validated database path"""
        path = Path(self.db_path)
        
        # Resolve to absolute path
        if not path.is_absolute():
            path = Path.cwd() / path
        
        # Security check - ensure path is within allowed directories
        allowed_dirs = [Path.cwd(), Path(tempfile.gettempdir())]
        
        try:
            # Check if path is within allowed directories
            path_resolved = path.resolve()
            for allowed_dir in allowed_dirs:
                try:
                    path_resolved.relative_to(allowed_dir.resolve())
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Database path {path} is outside allowed directories")
            
            # Create parent directories
            path_resolved.parent.mkdir(parents=True, exist_ok=True)
            
            return path_resolved
            
        except Exception as e:
            raise ValueError(f"Invalid database path {path}: {e}")

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    default_batch_size: int = 50
    max_batch_size: int = 500
    enable_streaming: bool = True
    chunk_size_mb: int = 10
    auto_save_enabled: bool = True
    checkpoint_interval: int = 100
    enable_parallel_processing: bool = True
    
    def validate(self) -> List[str]:
        """Validate processing configuration"""
        errors = []
        
        if self.default_batch_size <= 0 or self.default_batch_size > self.max_batch_size:
            errors.append(f"default_batch_size must be between 1 and {self.max_batch_size}")
        
        if self.chunk_size_mb <= 0 or self.chunk_size_mb > 100:
            errors.append("chunk_size_mb must be between 1 and 100")
        
        if self.checkpoint_interval <= 0:
            errors.append("checkpoint_interval must be positive")
        
        return errors

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_file: str = "logs/analyzer.log"
    max_file_size_mb: int = 50
    backup_count: int = 5
    enable_json_logging: bool = False
    log_to_console: bool = True
    
    def validate(self) -> List[str]:
        """Validate logging configuration"""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            errors.append(f"level must be one of {valid_levels}")
        
        if self.max_file_size_mb <= 0 or self.max_file_size_mb > 1000:
            errors.append("max_file_size_mb must be between 1 and 1000")
        
        return errors

@dataclass
class AppConfig:
    """Main application configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = True
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_environment(cls) -> 'AppConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Environment
        env = os.getenv('ANALYZER_ENV', 'development').lower()
        if env in [e.value for e in Environment]:
            config.environment = Environment(env)
        
        # Debug mode
        config.debug_mode = os.getenv('DEBUG', 'true').lower() == 'true'
        
        # Security settings
        config.security.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', config.security.max_file_size_mb))
        config.security.max_memory_usage_mb = int(os.getenv('MAX_MEMORY_MB', config.security.max_memory_usage_mb))
        config.security.max_concurrent_analyses = int(os.getenv('MAX_CONCURRENT', config.security.max_concurrent_analyses))
        
        # Database settings
        config.database.db_path = os.getenv('DATABASE_PATH', config.database.db_path)
        
        # Processing settings
        config.processing.default_batch_size = int(os.getenv('BATCH_SIZE', config.processing.default_batch_size))
        config.processing.enable_streaming = os.getenv('ENABLE_STREAMING', 'true').lower() == 'true'
        
        # Logging settings
        config.logging.level = os.getenv('LOG_LEVEL', config.logging.level).upper()
        config.logging.log_file = os.getenv('LOG_FILE', config.logging.log_file)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate entire configuration"""
        errors = []
        
        errors.extend(self.security.validate())
        errors.extend(self.processing.validate())
        errors.extend(self.logging.validate())
        
        return errors
    
    def is_file_allowed(self, filename: str, file_size: int) -> Tuple[bool, str]:
        """Check if file is allowed for processing"""
        file_path = Path(filename)
        
        # Check file extension
        if file_path.suffix.lower() not in self.security.allowed_file_extensions:
            return False, f"File extension {file_path.suffix} not allowed"
        
        # Check file size
        max_size = self.security.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            return False, f"File size {file_size / 1024 / 1024:.1f}MB exceeds limit of {self.security.max_file_size_mb}MB"
        
        # Check blocked patterns
        for pattern in self.security.blocked_file_patterns:
            if file_path.match(pattern):
                return False, f"File matches blocked pattern {pattern}"
        
        return True, "File allowed"
    
    def is_batch_allowed(self, file_count: int, total_size: int) -> Tuple[bool, str]:
        """Check if batch is allowed for processing"""
        # Check file count
        if file_count > self.security.max_files_per_batch:
            return False, f"Batch has {file_count} files, exceeds limit of {self.security.max_files_per_batch}"
        
        # Check total size
        max_size = self.security.max_batch_size_mb * 1024 * 1024
        if total_size > max_size:
            return False, f"Batch size {total_size / 1024 / 1024:.1f}MB exceeds limit of {self.security.max_batch_size_mb}MB"
        
        return True, "Batch allowed"

# Global configuration instance
_global_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get global configuration instance"""
    global _global_config
    
    if _global_config is None:
        _global_config = AppConfig.from_environment()
        
        # Validate configuration
        errors = _global_config.validate()
        if errors:
            logger.warning(f"Configuration validation errors: {errors}")
    
    return _global_config

def set_config(config: AppConfig) -> None:
    """Set global configuration instance"""
    global _global_config
    _global_config = config
