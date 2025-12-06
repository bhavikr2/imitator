"""
Imitator: Function monitoring and simulation library
"""

from .monitor import (
    FunctionMonitor,
    monitor_function,
    LocalStorage,
    DatabaseStorage,
    PostgreSQLConnector,
    MongoDBConnector,
    CouchbaseConnector,
    SQLiteConnector,
)
from .types import FunctionCall, FunctionSignature, IORecord

from .ml import train, TrainingDaemon

__version__ = "0.3.0"

__all__ = [
    "FunctionMonitor",
    "monitor_function",
    "FunctionCall",
    "FunctionSignature",
    "IORecord",
    "LocalStorage",
    "DatabaseStorage",
    "PostgreSQLConnector",
    "MongoDBConnector",
    "CouchbaseConnector",
    "SQLiteConnector",
    "train",
    "TrainingDaemon",
]
