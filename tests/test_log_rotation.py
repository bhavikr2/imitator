"""
Test cases for log rotation functionality
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import time
import threading

from imitator import LocalStorage, monitor_function, get_monitor


class TestLogRotation:
    """Test log rotation functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(
            log_dir=self.temp_dir,
            max_file_size_mb=0.001,  # 1KB for testing
            rotation_batch_size=3
        )
    
    def teardown_method(self):
        """Clean up test environment"""
        # Wait a moment for any async operations to complete
        time.sleep(0.1)
        try:
            shutil.rmtree(self.temp_dir)
        except FileNotFoundError:
            pass  # Directory already cleaned up
    
    def test_file_size_detection(self):
        """Test file size detection"""
        # Create a test function
        @monitor_function(storage=self.storage)
        def test_func(data: str) -> str:
            return f"Processed: {data}"
        
        # Generate enough data to exceed size limit
        large_data = "x" * 200  # Large string to increase file size
        for i in range(10):
            test_func(large_data)
        
        # Wait for async saves to complete
        time.sleep(0.1)
        
        # Check file info
        info = self.storage.get_log_file_info("test_func")
        assert info["exists"]
        assert info["size_bytes"] > 0
        assert info["entry_count"] > 0
    
    def test_rotation_settings(self):
        """Test rotation settings management"""
        # Check default settings (use approximate comparison for floating point)
        settings = self.storage.get_rotation_settings()
        assert abs(settings["max_file_size_mb"] - 0.001) < 0.0001
        assert settings["rotation_batch_size"] == 3
        
        # Update settings
        self.storage.set_rotation_settings(
            max_file_size_mb=0.002,
            rotation_batch_size=5
        )
        
        updated_settings = self.storage.get_rotation_settings()
        assert abs(updated_settings["max_file_size_mb"] - 0.002) < 0.0001
        assert updated_settings["rotation_batch_size"] == 5
    
    def test_manual_rotation(self):
        """Test manual log rotation"""
        # Use larger limits to prevent automatic rotation during test
        storage = LocalStorage(
            log_dir=self.temp_dir,
            max_file_size_mb=1.0,  # 1MB - large enough to prevent auto-rotation
            rotation_batch_size=3
        )
        
        # Create entries
        @monitor_function(storage=storage)
        def test_func(value: int) -> dict:
            return {"value": value, "data": "x" * 10}  # Smaller data
        
        # Generate entries
        for i in range(10):
            test_func(i)
        
        # Wait for async saves
        time.sleep(0.1)
        
        # Check initial count
        info_before = storage.get_log_file_info("test_func")
        initial_count = info_before["entry_count"]
        assert initial_count == 10
        
        # Force rotation
        success = storage.force_rotate_log("test_func", entries_to_remove=5)
        assert success
        
        # Check after rotation
        info_after = storage.get_log_file_info("test_func")
        assert info_after["entry_count"] == initial_count - 5
    
    def test_automatic_rotation(self):
        """Test automatic rotation when size limit is exceeded"""
        # Create monitor with very small size limit
        monitor = get_monitor(
            storage=LocalStorage(
                log_dir=self.temp_dir,
                max_file_size_mb=0.0005,  # Very small limit
                rotation_batch_size=2
            )
        )
        
        @monitor.monitor
        def test_func(data: str) -> str:
            return f"Large output: {data * 100}"  # Create large output
        
        # Generate entries - some should trigger rotation
        for i in range(15):
            test_func(f"data_{i}")
        
        # Wait for async saves and rotations
        time.sleep(0.2)
        
        # File should exist and rotation should have occurred
        final_info = monitor.storage.get_log_file_info("test_func")
        assert final_info["exists"]
        # Due to rotation, final count should be less than total entries generated
        assert final_info["entry_count"] < 15
    
    def test_get_all_log_files_info(self):
        """Test getting information for all log files"""
        # Use larger limits to prevent rotation
        storage = LocalStorage(
            log_dir=self.temp_dir,
            max_file_size_mb=1.0,
            rotation_batch_size=50
        )
        
        # Create multiple functions
        @monitor_function(storage=storage)
        def func1(x: int) -> int:
            return x * 2
        
        @monitor_function(storage=storage)
        def func2(x: str) -> str:
            return x.upper()
        
        # Generate entries
        for i in range(5):
            func1(i)
            func2(f"test_{i}")
        
        # Wait for async saves
        time.sleep(0.1)
        
        # Get all info
        all_info = storage.get_all_log_files_info()
        
        assert "func1" in all_info
        assert "func2" in all_info
        assert all_info["func1"]["entry_count"] == 5
        assert all_info["func2"]["entry_count"] == 5
    
    def test_nonexistent_file_rotation(self):
        """Test rotation of nonexistent file"""
        success = self.storage.force_rotate_log("nonexistent_func")
        assert not success
    
    def test_entry_counter_tracking(self):
        """Test that entry counters work correctly"""
        monitor = get_monitor(
            storage=LocalStorage(
                log_dir=self.temp_dir,
                max_file_size_mb=0.001,
                rotation_batch_size=3
            )
        )
        
        @monitor.monitor
        def test_func(data: str) -> str:
            return f"Output: {data * 50}"
        
        # Generate entries to trigger counter tracking
        for i in range(8):  # More entries to ensure counter is used
            test_func(f"data_{i}")
        
        # Wait for processing
        time.sleep(0.1)
        
        settings = monitor.storage.get_rotation_settings()
        # Entry counter should exist for the function (might be 0 if reset after rotation)
        assert "test_func" in settings["entry_counters"]


if __name__ == "__main__":
    pytest.main([__file__]) 