#!/usr/bin/env python3
"""
Advanced Monitoring Example - Imitator Framework

This example demonstrates advanced monitoring features including:
- Custom storage configuration
- Sampling rates and rate limiting
- Class method monitoring
- Async function monitoring
- Error handling and recovery
- Performance analysis
"""

import sys
import os
import asyncio
import time
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imitator import monitor_function, LocalStorage, FunctionMonitor


# Custom storage configuration
custom_storage = LocalStorage(log_dir="advanced_logs", format="json")


# Example 1: Class with monitored methods
class DataProcessor:
    """A class that processes data with monitored methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
    
    @monitor_function(storage=custom_storage)
    def process_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of items."""
        if not items:
            return {"processed": 0, "errors": 0, "results": []}
        
        results = []
        errors = 0
        
        for item in items:
            try:
                # Simulate processing
                processed_item = self._process_single_item(item)
                results.append(processed_item)
                self.processed_count += 1
            except Exception as e:
                errors += 1
                results.append({"error": str(e), "original": item})
        
        return {
            "processed": len(results) - errors,
            "errors": errors,
            "results": results,
            "total_processed": self.processed_count
        }
    
    def _process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item (helper method)."""
        # Simulate some processing logic
        if not isinstance(item, dict):
            raise ValueError("Item must be a dictionary")
        
        if "value" not in item:
            raise KeyError("Item must have 'value' field")
        
        return {
            "id": item.get("id", "unknown"),
            "value": item["value"] * 2,  # Double the value
            "processed_by": self.name,
            "timestamp": datetime.now().isoformat()
        }
    
    @monitor_function(storage=custom_storage)
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processor_name": self.name,
            "total_processed": self.processed_count,
            "status": "active" if self.processed_count > 0 else "idle"
        }


# Example 2: Async function monitoring
@monitor_function(storage=custom_storage)
async def fetch_data(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Simulate fetching data from an API."""
    await asyncio.sleep(0.1)  # Simulate network delay
    
    # Simulate different responses based on URL
    if "error" in url:
        raise ConnectionError(f"Failed to fetch data from {url}")
    elif "slow" in url:
        await asyncio.sleep(1.0)  # Simulate slow response
        return {"data": "slow response", "url": url, "delay": 1.0}
    else:
        return {"data": "success", "url": url, "timestamp": datetime.now().isoformat()}


@monitor_function(storage=custom_storage)
async def process_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """Process multiple URLs concurrently."""
    tasks = [fetch_data(url) for url in urls]
    results = []
    
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "type": type(e).__name__})
    
    return results


# Example 3: Rate-limited monitoring
high_frequency_monitor = FunctionMonitor(
    storage=custom_storage,
    sampling_rate=0.1,  # Only log 10% of calls
    max_calls_per_minute=30  # Maximum 30 calls per minute
)

@high_frequency_monitor.monitor
def high_frequency_function(x: int) -> int:
    """A function that might be called very frequently."""
    return x * x + x + 1


# Example 4: Functions with in-place modifications
@monitor_function(storage=custom_storage)
def sort_and_modify(data: List[int], reverse: bool = False) -> List[int]:
    """Sort a list and modify it in place."""
    data.sort(reverse=reverse)  # In-place modification
    return data


@monitor_function(storage=custom_storage)
def update_dictionary(data: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update a dictionary with new values."""
    original_keys = set(data.keys())
    data.update(updates)  # In-place modification
    data["_updated_keys"] = list(set(data.keys()) - original_keys)
    return data


# Example 5: Performance monitoring utilities
def performance_test():
    """Test performance monitoring capabilities."""
    print("\n🚀 Performance Testing")
    print("=" * 40)
    
    # Test high-frequency function
    print("Testing high-frequency function (with rate limiting)...")
    start_time = time.time()
    for i in range(100):
        result = high_frequency_function(i)
    end_time = time.time()
    print(f"Completed 100 calls in {end_time - start_time:.3f} seconds")
    
    # Test data processing
    print("\nTesting data processing...")
    processor = DataProcessor("TestProcessor")
    
    # Test with valid data
    test_data = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30}
    ]
    result = processor.process_batch(test_data)
    print(f"Processed {len(test_data)} items: {result['processed']} succeeded, {result['errors']} failed")
    
    # Test with some invalid data
    mixed_data = [
        {"id": 4, "value": 40},
        {"id": 5},  # Missing 'value' field - will cause error
        {"id": 6, "value": 60}
    ]
    result = processor.process_batch(mixed_data)
    print(f"Processed {len(mixed_data)} items: {result['processed']} succeeded, {result['errors']} failed")
    
    # Get stats
    stats = processor.get_stats()
    print(f"Processor stats: {stats}")


async def async_test():
    """Test asynchronous function monitoring."""
    print("\n🌐 Async Function Testing")
    print("=" * 40)
    
    # Test individual async function
    print("Testing individual async calls...")
    try:
        result = await fetch_data("https://api.example.com/data")
        print(f"Fetch successful: {result}")
    except Exception as e:
        print(f"Fetch failed: {e}")
    
    # Test with error
    try:
        result = await fetch_data("https://api.example.com/error")
        print(f"Fetch successful: {result}")
    except Exception as e:
        print(f"Fetch failed: {e}")
    
    # Test concurrent processing
    print("\nTesting concurrent URL processing...")
    test_urls = [
        "https://api.example.com/data1",
        "https://api.example.com/data2",
        "https://api.example.com/error",
        "https://api.example.com/slow",
        "https://api.example.com/data3"
    ]
    
    start_time = time.time()
    results = await process_urls(test_urls)
    end_time = time.time()
    
    print(f"Processed {len(test_urls)} URLs in {end_time - start_time:.3f} seconds")
    for i, result in enumerate(results):
        if "error" in result:
            print(f"  URL {i+1}: ERROR - {result['error']}")
        else:
            print(f"  URL {i+1}: SUCCESS - {result.get('data', 'No data')}")


def modification_test():
    """Test detection of in-place modifications."""
    print("\n🔄 In-Place Modification Testing")
    print("=" * 40)
    
    # Test list sorting
    original_list = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"Original list: {original_list}")
    
    sorted_list = sort_and_modify(original_list.copy())
    print(f"Sorted list: {sorted_list}")
    
    reverse_sorted = sort_and_modify(original_list.copy(), reverse=True)
    print(f"Reverse sorted: {reverse_sorted}")
    
    # Test dictionary updates
    original_dict = {"name": "John", "age": 30}
    print(f"Original dict: {original_dict}")
    
    updates = {"age": 31, "city": "New York", "country": "USA"}
    updated_dict = update_dictionary(original_dict.copy(), updates)
    print(f"Updated dict: {updated_dict}")


def analyze_advanced_logs():
    """Analyze logs from advanced monitoring."""
    print("\n📊 Advanced Log Analysis")
    print("=" * 40)
    
    # Get all monitored functions
    functions = custom_storage.get_all_functions()
    print(f"Total monitored functions: {len(functions)}")
    
    # Analyze each function
    for func_name in functions:
        calls = custom_storage.load_calls(func_name)
        if not calls:
            continue
        
        print(f"\n📋 Function: {func_name}")
        print(f"   Total calls: {len(calls)}")
        
        # Execution time analysis
        times = [call.io_record.execution_time_ms for call in calls 
                if call.io_record.execution_time_ms is not None]
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"   Execution time: avg={avg_time:.3f}ms, min={min_time:.3f}ms, max={max_time:.3f}ms")
        
        # Error analysis
        errors = [call for call in calls if isinstance(call.io_record.output, dict) 
                 and "error" in call.io_record.output]
        if errors:
            print(f"   Errors: {len(errors)}/{len(calls)} calls failed")
            error_types = defaultdict(int)
            for error_call in errors:
                error_type = error_call.io_record.output.get("type", "Unknown")
                error_types[error_type] += 1
            print(f"   Error types: {dict(error_types)}")
        
        # Modification analysis
        modifications = [call for call in calls if call.io_record.input_modifications]
        if modifications:
            print(f"   In-place modifications: {len(modifications)}/{len(calls)} calls")
            for mod_call in modifications[:2]:  # Show first 2 examples
                mods = mod_call.io_record.input_modifications
                print(f"     Modified parameters: {list(mods.keys())}")
    
    print(f"\n📁 Advanced logs stored in: {custom_storage.log_dir}")


async def main():
    """Run all advanced monitoring examples."""
    print("🚀 Imitator Framework - Advanced Monitoring Demo")
    print("=" * 60)
    
    # Run performance tests
    performance_test()
    
    # Run async tests
    await async_test()
    
    # Run modification tests
    modification_test()
    
    print("\n" + "=" * 60)
    print("✅ All advanced monitoring tests completed!")
    
    # Analyze logs
    analyze_advanced_logs()


if __name__ == "__main__":
    asyncio.run(main()) 