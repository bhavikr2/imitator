"""
LogAndLearn - A lightweight function monitoring and I/O logging framework

A powerful yet simple framework for monitoring function I/O, tracking performance,
and collecting data for machine learning, debugging, and analysis.
"""

from .monitor import monitor_function, FunctionMonitor
from .types import FunctionCall, IORecord, FunctionSignature
from .storage import LocalStorage

__version__ = "0.1.0"
__author__ = "LogAndLearn Team"
__email__ = "contact@logandlearn.dev"
__license__ = "MIT"
__description__ = "A lightweight Python framework for monitoring function I/O with automatic logging and type validation"

__all__ = [
    "monitor_function",
    "FunctionMonitor", 
    "FunctionCall",
    "IORecord",
    "FunctionSignature",
    "LocalStorage"
] 