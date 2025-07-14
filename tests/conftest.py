"""
Pytest configuration and fixtures for Imitator tests
"""

import pytest
from pathlib import Path
import tempfile
import glob
import time
import asyncio
from imitator.monitor import FunctionMonitor
from imitator import LocalStorage

def clean_test_logs():
    """Helper function to clean up test log files."""
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


async def wait_for_logs(monitor: FunctionMonitor, timeout: float = 4.0) -> bool:
    """Wait for a specific FunctionMonitor instance to finish its async log saves."""
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        with monitor._lock:
            # Clean up finished threads before checking
            monitor._active_threads = {t for t in monitor._active_threads if t.is_alive()}
            if not monitor._active_threads:
                return True
        
        # Asynchronously wait before the next check
        await asyncio.sleep(0.05)
        
    return False


@pytest.fixture
def clean_logs():
    """Fixture to clean up log files before and after tests"""
    clean_test_logs()
    yield
    clean_test_logs()


@pytest.fixture
def function_monitor() -> FunctionMonitor:
    """Fixture to provide an isolated FunctionMonitor instance for a single test."""
    return FunctionMonitor()


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