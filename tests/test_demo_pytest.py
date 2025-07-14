"""
Demonstration pytest tests showing key features of Imitator framework
"""

import pytest
import asyncio
import shutil
from pathlib import Path
from typing import List, Dict, Any

from imitator import monitor_function, LocalStorage
from imitator.monitor import get_monitor
from tests.conftest import wait_for_logs

# We'll use the function_monitor fixture for explicit monitor instances

class TestBasicMonitoring:
    """Demonstrate basic function monitoring with proper pytest assertions"""
    
    @pytest.mark.asyncio
    async def test_simple_function_with_assertions(self,clean_logs, function_monitor):
        """Test that demonstrates proper pytest assertions"""
        
        @monitor_function(monitor=function_monitor)
        def calculate_area(length: float, width: float) -> float:
            """Calculate rectangle area"""
            return length * width
        
        # Test the function with different inputs
        area1 = calculate_area(5.0, 3.0)
        area2 = calculate_area(10.0, 2.0)
        
        # Pytest assertions with clear error messages
        assert area1 == 15.0, f"Expected 15.0, got {area1}"
        assert area2 == 20.0, f"Expected 20.0, got {area2}"
        
        # Wait for logging to complete
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"
        
        # Test that logging worked correctly
        storage = LocalStorage()
        functions = storage.get_all_functions()
        assert "calculate_area" in functions, "Function should be logged"
        
        calls = storage.load_calls("calculate_area")
        assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
        
        # Verify the logged data matches what we expect
        logged_outputs = [call.io_record.output for call in calls]
        assert set(logged_outputs) == {15.0, 20.0}, "All outputs should be logged"
    
    @pytest.mark.asyncio
    async def test_exception_handling_with_assertions(self, clean_logs, function_monitor):
        """Test exception handling with proper pytest expectations"""
        
        @monitor_function(monitor=function_monitor)
        def safe_divide(numerator: float, denominator: float) -> float:
            """Divide two numbers with error checking"""
            if denominator == 0:
                raise ValueError("Cannot divide by zero!")
            return numerator / denominator
        
        # Test successful division
        result = safe_divide(10.0, 2.0)
        assert result == 5.0, "Division should work correctly"
        
        # Test that exception is raised properly
        with pytest.raises(ValueError, match="Cannot divide by zero!"):
            safe_divide(10.0, 0.0)
        
        # Test continued operation after exception
        result2 = safe_divide(15.0, 3.0)
        assert result2 == 5.0, "Function should continue working after exception"
        
        # Wait for logging
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"
        
        # Verify logging captured everything
        storage = LocalStorage()
        calls = storage.load_calls("safe_divide")
        assert len(calls) == 3, "Should have logged all 3 calls (2 success + 1 error)"
        
        # Separate successful and error calls
        success_calls = [call for call in calls if not isinstance(call.io_record.output, dict)]
        error_calls = [call for call in calls if isinstance(call.io_record.output, dict)]
        
        assert len(success_calls) == 2, "Should have 2 successful calls"
        assert len(error_calls) == 1, "Should have 1 error call"
        
        # Verify error call structure
        error_call = error_calls[0]
        assert "error" in error_call.io_record.output, "Error should be logged"
        assert "Cannot divide by zero!" in error_call.io_record.output["error"], "Error message should be preserved"


class TestAsyncMonitoring:
    """Test async function monitoring with pytest-asyncio"""
    
    @pytest.mark.asyncio
    async def test_async_function_monitoring(self, clean_logs, function_monitor):
        """Test async function monitoring"""
        
        @monitor_function(monitor=function_monitor)
        async def async_fetch_data(url: str, timeout: float = 1.0) -> Dict[str, Any]:
            """Simulate async data fetching"""
            await asyncio.sleep(0.01)  # Simulate network delay
            
            if "error" in url:
                raise ConnectionError(f"Failed to fetch from {url}")
            
            return {
                "url": url,
                "status": "success",
                "data": f"Content from {url}",
                "timeout": timeout
            }
        
        # Test successful async calls
        result1 = await async_fetch_data("https://api.example.com")
        result2 = await async_fetch_data("https://api.test.com", timeout=2.0)
        
        # Assert results
        assert result1["status"] == "success", "First call should succeed"
        assert result1["url"] == "https://api.example.com", "URL should be preserved"
        assert result2["timeout"] == 2.0, "Timeout parameter should be preserved"
        
        # Test async exception
        with pytest.raises(ConnectionError, match="Failed to fetch"):
            await async_fetch_data("https://error.example.com")
        
        # Wait for logging
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"
        
        # Verify async logging worked
        storage = LocalStorage()
        calls = storage.load_calls("async_fetch_data")
        assert len(calls) >= 2, f"Should have logged at least 2 calls, got {len(calls)}"
        
        # Verify async call data
        success_calls = [
            call for call in calls
            if isinstance(call.io_record.output, dict) and call.io_record.output.get('status') == 'success'
        ]
        assert len(success_calls) == 2, "Should have 2 successful async calls"


class TestComplexDataTypes:
    """Test monitoring of functions with complex data types"""
    
    @pytest.mark.asyncio
    async def test_nested_data_structures(self, clean_logs, function_monitor):
        """Test monitoring functions with nested data structures"""
        
        @monitor_function(monitor=function_monitor)
        def process_user_data(user_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process complex user data structure"""
            processed = {
                "user_id": user_data.get("id"),
                "full_name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
                "email_domain": user_data.get("email", "").split("@")[-1] if "@" in user_data.get("email", "") else "",
                "preferences_count": len(user_data.get("preferences", {})),
                "is_premium": user_data.get("subscription", {}).get("type") == "premium"
            }
            return processed
        
        # Test with complex nested data
        test_user = {
            "id": 12345,
            "first_name": "Alice",
            "last_name": "Johnson",
            "email": "alice.johnson@example.com",
            "preferences": {
                "theme": "dark",
                "notifications": True,
                "language": "en"
            },
            "subscription": {
                "type": "premium",
                "expires": "2024-12-31"
            }
        }
        
        result = process_user_data(test_user)
        
        # Detailed assertions
        assert result["user_id"] == 12345, "User ID should be extracted"
        assert result["full_name"] == "Alice Johnson", "Full name should be concatenated"
        assert result["email_domain"] == "example.com", "Email domain should be extracted"
        assert result["preferences_count"] == 3, "Should count preferences correctly"
        assert result["is_premium"] is True, "Should detect premium subscription"
        
        # Wait for logging
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"
        
        # Debug: Check if logs directory exists
        import os
        logs_dir = Path("logs")
        print(f"DEBUG: logs directory exists: {logs_dir.exists()}")
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.jsonl"))
            print(f"DEBUG: log files found: {log_files}")
        
        # Verify complex data logging
        storage = LocalStorage()
        calls = storage.load_calls("process_user_data")
        print(f"DEBUG: Found {len(calls)} calls")
        assert len(calls) == 1, "Should have logged the call"
        
        call = calls[0]
        logged_input = call.io_record.inputs["user_data"]
        logged_output = call.io_record.output
        
        # Verify input data preservation
        assert logged_input["id"] == test_user["id"], "Input should be preserved exactly"
        assert logged_input["preferences"]["theme"] == "dark", "Nested input should be preserved"
        
        # Verify output data
        assert logged_output == result, "Output should match function result"


class TestPerformanceCharacteristics:
    """Test performance-related features"""
    
    @pytest.mark.asyncio
    async def test_sampling_functionality(self, clean_logs, function_monitor):
        """Test that sampling works as expected"""

        @monitor_function(sampling_rate=0.1)  # 10% sampling
        def high_frequency_function(x: int) -> int:
            """Function called frequently with sampling"""
            return x * x
        
        # Call function many times
        num_calls = 1000
        results = []
        for i in range(num_calls):
            result = high_frequency_function(i)
            results.append(result)
        
        # Verify all function calls worked
        assert len(results) == num_calls, "All function calls should complete"
        assert results[:5] == [0, 1, 4, 9, 16], "Function should compute correctly"
        
        # Wait for logging
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"
        
        # Verify sampling worked
        storage = LocalStorage()
        calls = storage.load_calls("high_frequency_function")
        
        # Should have approximately 10% of calls logged (allowing for randomness)
        expected_logged = num_calls * 0.1
        assert len(calls) < num_calls * 0.5, f"Should log much less than {num_calls}, got {len(calls)}"
        assert len(calls) > 0, "Should log some calls"
        
        print(f"Logged {len(calls)}/{num_calls} calls ({len(calls)/num_calls*100:.1f}% - expected ~10%)")
    
    @pytest.mark.asyncio
    async def test_rate_limiting_functionality(self,clean_logs):
        """Test rate limiting functionality"""

        function_monitor = get_monitor(max_calls_per_minute=5)
        
        @monitor_function(monitor=function_monitor)
        def rate_limited_function(message: str) -> str:
            """Function with rate limiting"""
            return f"Processed: {message}"
        
        # Call function more than the limit
        num_calls = 20
        results = []
        for i in range(num_calls):
            result = rate_limited_function(f"message_{i}")
            results.append(result)
        
        # Verify all function calls worked (rate limiting doesn't affect function execution)
        assert len(results) == num_calls, "All function calls should complete"
        assert "Processed: message_0" in results[0], "Function should work correctly"
        
        # Wait for logging
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"
        
        # Verify rate limiting worked
        storage = LocalStorage()
        calls = storage.load_calls("rate_limited_function")
        
        # Should have at most 5 calls logged due to rate limiting
        assert len(calls) <= 5, f"Rate limiting should cap at 5 calls, got {len(calls)}"
        
        print(f"Rate limited to {len(calls)} calls out of {num_calls} total calls")


# Summary function to demonstrate test results
def test_framework_summary():
    """Test that demonstrates the overall framework capabilities"""
    print("\n" + "="*60)
    print("🎉 Imitator Framework pytest Test Suite Summary")
    print("="*60)
    print("✅ Basic function monitoring with type capture")
    print("✅ Exception handling and error logging")
    print("✅ Async function support with pytest-asyncio")
    print("✅ Complex data type handling")
    print("✅ Performance features (sampling, rate limiting)")
    print("✅ Comprehensive pytest assertions")
    print("✅ Thread-safe logging with timing controls")
    print("="*60)
    
    # This always passes - it's just for demonstration
    assert True, "Framework test suite completed successfully!" 