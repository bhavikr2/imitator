"""
Pytest configuration and fixtures for LogAndLearn tests
"""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import glob
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logandlearn import LocalStorage


def clean_test_logs():
    """Helper function to clean up test log files"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Common log directories to clean
    log_dirs = [
        project_root / "logs",
        project_root / "examples" / "logs",
        project_root / "test_logs",
        Path.cwd() / "logs",
        Path.cwd() / "test_logs"
    ]
    
    # Clean log files from all directories
    for log_dir in log_dirs:
        if log_dir.exists():
            for log_file in log_dir.glob("*.jsonl"):
                try:
                    log_file.unlink()
                except (OSError, FileNotFoundError):
                    pass  # Ignore errors if file doesn't exist or can't be deleted


def wait_for_logs(timeout=2.0):
    """Wait for daemon threads to flush logs"""
    time.sleep(timeout)


@pytest.fixture
def clean_logs():
    """Fixture to clean up log files before and after tests"""
    clean_test_logs()
    yield
    clean_test_logs()


@pytest.fixture
def temp_storage():
    """Fixture to provide a temporary storage directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorage(log_dir=temp_dir)
        yield storage


@pytest.fixture
def sample_data():
    """Fixture providing sample data for tests"""
    return {
        "integers": [1, 2, 3, 4, 5],
        "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
        "strings": ["hello", "world", "test", "data"],
        "nested_dict": {
            "level1": {
                "level2": {
                    "values": [1, 2, 3]
                }
            }
        },
        "empty_list": [],
        "mixed_types": [1, "two", 3.0, True, None]
    } 