#!/usr/bin/env python3
"""
Custom Exceptions for Tajik Poetry Analyzer
Provides specific exception types for better error handling
"""

class AnalyzerException(Exception):
    """Base exception for all analyzer errors"""
    pass


class ValidationError(AnalyzerException):
    """Raised when input validation fails"""
    pass


class PoemParsingError(AnalyzerException):
    """Raised when poem parsing fails"""
    pass


class ConfigurationError(AnalyzerException):
    """Raised when configuration is invalid"""
    pass


class DatabaseError(AnalyzerException):
    """Raised when database operations fail"""
    pass


class FileProcessingError(AnalyzerException):
    """Raised when file processing fails"""
    pass


class MemoryLimitExceededError(AnalyzerException):
    """Raised when memory limits are exceeded"""
    pass


class BatchProcessingError(AnalyzerException):
    """Raised when batch processing encounters errors"""
    def __init__(self, message: str, failed_items: list = None):
        super().__init__(message)
        self.failed_items = failed_items or []


class TimeoutError(AnalyzerException):
    """Raised when an operation times out"""
    pass


class UnsupportedFormatError(AnalyzerException):
    """Raised when an unsupported file format is encountered"""
    pass
