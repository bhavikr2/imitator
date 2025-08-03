#!/usr/bin/env python3
"""
Example demonstrating log rotation functionality in Imitator

This script shows how to configure and use log rotation to manage
file sizes automatically when they exceed preset limits.
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import imitator
sys.path.insert(0, str(Path(__file__).parent.parent))

from imitator import monitor_function, LocalStorage, get_monitor


def cleanup_logs():
    """Clean up example log files"""
    log_dir = Path("rotation_logs")
    if log_dir.exists():
        for file in log_dir.glob("*.jsonl"):
            file.unlink()
        log_dir.rmdir()


@monitor_function(
    # Set small limits for demonstration
    max_file_size_mb=0.001,  # 1KB for demo purposes
    rotation_batch_size=5    # Remove 5 entries when rotating
)
def example_function(x: int, y: str = "default") -> dict:
    """Example function that generates logs"""
    return {
        "result": x * 2,
        "message": f"Processed {x} with {y}",
        "timestamp": time.time()
    }


def demonstrate_rotation():
    """Demonstrate log rotation in action"""
    print("🔄 Log Rotation Demonstration")
    print("=" * 50)
    
    # Clean up any existing logs
    cleanup_logs()
    
    # Create storage with custom settings
    storage = LocalStorage(
        log_dir="rotation_logs",
        max_file_size_mb=0.001,  # 1KB for demo
        rotation_batch_size=3
    )
    
    print(f"📊 Initial rotation settings:")
    settings = storage.get_rotation_settings()
    print(f"   Max file size: {settings['max_file_size_mb']:.3f} MB")
    print(f"   Rotation batch size: {settings['rotation_batch_size']}")
    print()
    
    # Create monitor with custom storage
    monitor = get_monitor(
        storage=storage,
        max_file_size_mb=0.001,
        rotation_batch_size=3
    )
    
    @monitor.monitor
    def test_function(value: int) -> str:
        """Test function for rotation demo"""
        return f"Result for {value}: {'x' * 100}"  # Larger output to increase file size
    
    print("📝 Generating log entries...")
    
    # Generate entries and monitor file growth
    for i in range(20):
        test_function(i)
        
        # Check file info every few entries
        if i % 3 == 0:
            info = storage.get_log_file_info("test_function")
            print(f"   Entry {i:2d}: {info['size_bytes']:4d} bytes, "
                  f"{info['entry_count']:2d} entries, "
                  f"rotation needed: {info['rotation_needed']}")
            
            if info['rotation_needed'] and info['entries_until_rotation'] <= 3:
                print(f"   🔄 Rotation will happen soon! "
                      f"({info['entries_until_rotation']} entries until rotation)")
    
    print()
    print("📈 Final file information:")
    info = storage.get_log_file_info("test_function")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print()
    print("🔧 Manual rotation demonstration:")
    print(f"   Entries before manual rotation: {info['entry_count']}")
    
    # Force a manual rotation
    success = storage.force_rotate_log("test_function", entries_to_remove=5)
    if success:
        info_after = storage.get_log_file_info("test_function")
        print(f"   Entries after removing 5: {info_after['entry_count']}")
        print(f"   File size reduced from {info['size_bytes']} to {info_after['size_bytes']} bytes")
    
    print()
    print("📋 All monitored functions:")
    all_info = storage.get_all_log_files_info()
    for func_name, func_info in all_info.items():
        print(f"   {func_name}: {func_info['entry_count']} entries, "
              f"{func_info['size_mb']:.3f} MB")


def demonstrate_settings_management():
    """Demonstrate how to manage rotation settings"""
    print("\n⚙️  Settings Management Demonstration")
    print("=" * 50)
    
    storage = LocalStorage()
    
    print("📊 Default settings:")
    settings = storage.get_rotation_settings()
    for key, value in settings.items():
        print(f"   {key}: {value}")
    
    print("\n🔧 Updating settings...")
    storage.set_rotation_settings(
        max_file_size_mb=100.0,  # 100MB
        rotation_batch_size=100  # Remove 100 entries at a time
    )
    
    print("📊 Updated settings:")
    settings = storage.get_rotation_settings()
    for key, value in settings.items():
        print(f"   {key}: {value}")


def demonstrate_info_queries():
    """Demonstrate querying log file information"""
    print("\n📊 Log Information Queries")
    print("=" * 50)
    
    # Generate some sample data first
    storage = LocalStorage(log_dir="info_demo_logs")
    
    @monitor_function(storage=storage)
    def sample_func(n: int) -> dict:
        return {"value": n, "squared": n**2}
    
    # Generate some entries
    for i in range(10):
        sample_func(i)
    
    print("📄 Single function info:")
    info = storage.get_log_file_info("sample_func")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n📚 All functions info:")
    all_info = storage.get_all_log_files_info()
    for func_name, func_info in all_info.items():
        print(f"   {func_name}:")
        print(f"     - Entries: {func_info['entry_count']}")
        print(f"     - Size: {func_info['size_mb']:.3f} MB")
        print(f"     - Rotation needed: {func_info['rotation_needed']}")
    
    # Clean up
    cleanup_demo_logs("info_demo_logs")


def cleanup_demo_logs(log_dir: str):
    """Clean up demonstration log files"""
    log_path = Path(log_dir)
    if log_path.exists():
        for file in log_path.glob("*.jsonl"):
            file.unlink()
        log_path.rmdir()


if __name__ == "__main__":
    try:
        demonstrate_rotation()
        demonstrate_settings_management()
        demonstrate_info_queries()
        
        print("\n✅ Log rotation demonstration completed!")
        print("\n💡 Key takeaways:")
        print("   • Log files automatically rotate when they exceed size limits")
        print("   • Rotation removes oldest entries in configurable batches")
        print("   • You can monitor file sizes and entry counts in real-time")
        print("   • Manual rotation is available when needed")
        print("   • Settings can be adjusted at runtime")
        
    finally:
        # Clean up all demo files
        cleanup_logs()
        cleanup_demo_logs("info_demo_logs")
        print("\n🧹 Demo files cleaned up.") 