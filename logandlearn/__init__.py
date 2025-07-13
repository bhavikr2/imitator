"""
LogAndLearn - A lightweight function monitoring and I/O logging framework
"""

from .monitor import monitor_function, wait_for_all_saves
from .types import FunctionCall, IORecord
from .storage import LocalStorage

__version__ = "0.1.0"
__all__ = ["monitor_function", "FunctionCall", "IORecord", "LocalStorage", "wait_for_all_saves"] 