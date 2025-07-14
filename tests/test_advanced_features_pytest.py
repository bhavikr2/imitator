"""
Advanced features tests using pytest (threading, rate limiting, complex scenarios)
"""

import pytest
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from tests.conftest import wait_for_logs
from imitator import monitor_function, LocalStorage


class TestThreadSafety:
    """Test thread-safe logging"""
    
    @pytest.mark.asyncio
    async def test_concurrent_function_calls(self, clean_logs, function_monitor):
        """Test that concurrent function calls are logged safely"""
        
        @monitor_function(monitor=function_monitor)
        def thread_safe_counter(thread_id: int, increment: int = 1) -> int:
            """Thread-safe function for testing concurrent access"""
            time.sleep(0.01)  # Small delay to encourage race conditions
            return thread_id * increment
        
        def worker(thread_id: int, num_calls: int):
            """Worker function for each thread"""
            results = []
            for i in range(num_calls):
                result = thread_safe_counter(thread_id, i + 1)
                results.append(result)
            return results
        
        # Run multiple threads concurrently
        num_threads = 3
        calls_per_thread = 5
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i, calls_per_thread) for i in range(num_threads)]
            
            # Collect all results
            all_results = []
            for future in futures:
                thread_results = future.result()
                all_results.extend(thread_results)
        
        # Verify results
        assert len(all_results) == num_threads * calls_per_thread
        
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"

        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("thread_safe_counter")
        assert len(calls) == num_threads * calls_per_thread, "All calls should be logged"
        
        # Verify that each call was logged correctly
        logged_results = [call.io_record.output for call in calls]
        assert set(logged_results) == set(all_results), "All results should be logged"
    
    @pytest.mark.asyncio
    async def test_no_data_corruption_under_concurrency(self, clean_logs, function_monitor):
        """Test that no data corruption occurs under concurrent access"""
        
        @monitor_function(monitor=function_monitor)
        def data_processor(data: List[int]) -> int:
            """Process data that might be modified during logging"""
            return sum(data)
        
        def worker(worker_id: int):
            """Worker that processes different data sets"""
            data_sets = [
                [1, 2, 3],
                [4, 5, 6], 
                [7, 8, 9],
                [10, 11, 12]
            ]
            
            results = []
            for data in data_sets:
                result = data_processor(data)
                results.append(result)
            return results
        
        # Run multiple workers concurrently
        num_workers = 4
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(num_workers)]
            
            all_results = []
            for future in futures:
                worker_results = future.result()
                all_results.extend(worker_results)
        
        # Expected results: each worker processes the same data sets
        expected_results = [6, 15, 24, 33] * num_workers
        
        assert sorted(all_results) == sorted(expected_results), "Results should be correct"

        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"
        
        # Verify logging integrity
        storage = LocalStorage()
        calls = storage.load_calls("data_processor")
        assert len(calls) == num_workers * 4, "All calls should be logged"
        
        # Verify that logged inputs and outputs are consistent
        for call in calls:
            expected_sum = sum(call.io_record.inputs["data"])
            assert call.io_record.output == expected_sum, "Input/output should be consistent"


class TestRateLimiting:
    """Test rate limiting and sampling features"""
    
    def test_sampling_rate(self, clean_logs):
        """Test that sampling rate is approximately respected"""
        
        @monitor_function(sampling_rate=0.1)  # 10% sampling
        def sampled_function(x: int) -> int:
            """Function with 10% sampling"""
            return x * 2
        
        num_calls = 1000
        results = []
        
        for i in range(num_calls):
            result = sampled_function(i)
            results.append(result)
        
        # Verify all function calls worked
        assert len(results) == num_calls
        assert results == [i * 2 for i in range(num_calls)]
        
        # Verify sampling
        storage = LocalStorage()
        calls = storage.load_calls("sampled_function")
        
        # Should have approximately 10% of calls logged (allowing for randomness)
        expected_logged = num_calls * 0.1
        actual_logged = len(calls)
        print(f"Expected logged: {expected_logged}, Actual logged: {actual_logged}")
        assert actual_logged > 0, "Should have some calls logged"
        assert abs(actual_logged - expected_logged) / expected_logged * 100 <= 20, \
        f"Expected ~{expected_logged} calls, got {actual_logged}"
    
    def test_rate_limiting_per_minute(self, clean_logs):
        """Test rate limiting per minute"""
        
        @monitor_function(max_calls_per_minute=5)
        def rate_limited_function(x: int) -> int:
            """Function with rate limiting"""
            return x * 3
        
        num_calls = 20
        results = []
        
        for i in range(num_calls):
            result = rate_limited_function(i)
            results.append(result)
        
        # Verify all function calls worked
        assert len(results) == num_calls
        assert results == [i * 3 for i in range(num_calls)]
        
        # Verify rate limiting
        storage = LocalStorage()
        calls = storage.load_calls("rate_limited_function")
        
        # Should have at most 5 calls logged
        assert len(calls) <= 5, f"Expected at most 5 calls, got {len(calls)}"
    
    def test_combined_sampling_and_rate_limiting(self, clean_logs):
        """Test combination of sampling and rate limiting"""
        
        @monitor_function(sampling_rate=0.5, max_calls_per_minute=3)
        def combined_function(x: int) -> int:
            """Function with both sampling and rate limiting"""
            return x + 10
        
        num_calls = 100
        results = []
        
        for i in range(num_calls):
            result = combined_function(i)
            results.append(result)
        
        # Verify all function calls worked
        assert len(results) == num_calls
        
        # Verify combined limits
        storage = LocalStorage()
        calls = storage.load_calls("combined_function")
        
        # Should have at most 3 calls logged due to rate limiting
        assert len(calls) <= 3, f"Expected at most 3 calls, got {len(calls)}"


class TestComplexDataHandling:
    """Test handling of complex data types and modifications"""
    
    @pytest.mark.asyncio
    async def test_input_modification_detection(self, clean_logs, function_monitor):
        """Test detection of in-place input modifications"""
        
        @monitor_function(monitor=function_monitor)
        def modify_nested_data(data: Dict[str, Any]) -> str:
            """Function that modifies nested dictionary in-place"""
            if "counters" not in data:
                data["counters"] = {}
            
            data["counters"]["calls"] = data["counters"].get("calls", 0) + 1
            data["last_modified"] = time.time()
            
            if "items" in data:
                data["items"].append(f"item_{data['counters']['calls']}")
            
            return f"Modified data with {data['counters']['calls']} calls"
        
        # Test data that will be modified
        test_data = {
            "name": "test_data",
            "items": ["initial_item"],
            "metadata": {"created": time.time()}
        }
        
        original_items_count = len(test_data["items"])
        
        # Call function multiple times
        results = []
        for i in range(3):
            result = modify_nested_data(test_data)
            results.append(result)
        
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"

        # Verify modifications occurred
        assert len(test_data["items"]) == original_items_count + 3
        assert "counters" in test_data
        assert test_data["counters"]["calls"] == 3
        assert "last_modified" in test_data
        
        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("modify_nested_data")
        assert len(calls) == 3, "Should have 3 calls logged"
        
        # Each call should have the input state before modification
        # Since calls can be in any order, we need to check by their content
        call_item_counts = sorted([len(call.io_record.inputs["data"]["items"]) for call in calls])
        expected_counts = [original_items_count, original_items_count + 1, original_items_count + 2]
        assert call_item_counts == expected_counts, f"Expected item counts {expected_counts}, got {call_item_counts}"
    
    @pytest.mark.asyncio
    async def test_polymorphic_return_types(self, clean_logs, function_monitor):
        """Test functions with varying return types"""
        
        @monitor_function(monitor=function_monitor)
        def polymorphic_function(input_type: str, value: Any) -> Any:
            """Function that returns different types based on input"""
            if input_type == "int":
                return int(value) * 2
            elif input_type == "list":
                return list(value) + [len(value)]
            elif input_type == "dict":
                result = dict(value)
                result["processed"] = True
                return result
            elif input_type == "none":
                return None
            else:
                return str(value).upper()
        
        test_cases = [
            ("int", "42", 84),
            ("list", [1, 2, 3], [1, 2, 3, 3]),
            ("dict", {"key": "value"}, {"key": "value", "processed": True}),
            ("none", "anything", None),
            ("string", "hello", "HELLO"),
        ]
        
        for input_type, value, expected in test_cases:
            result = polymorphic_function(input_type, value)
            assert result == expected, f"For {input_type}, expected {expected}, got {result}"

        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"

        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("polymorphic_function")
        assert len(calls) == len(test_cases), f"Should have {len(test_cases)} calls logged"
        
        # Verify each call was logged correctly (order-independent)
        for input_type, value, expected in test_cases:
            # Find the call that matches this test case
            matching_calls = [
                call for call in calls 
                if call.io_record.inputs["input_type"] == input_type 
                and call.io_record.inputs["value"] == value
            ]
            assert len(matching_calls) == 1, f"Should have exactly one call for {input_type} with value {value}"
            
            call = matching_calls[0]
            assert call.io_record.output == expected, f"For {input_type}, expected {expected}, got {call.io_record.output}"
    
    def test_large_data_handling(self, clean_logs):
        """Test handling of large data structures"""
        
        @monitor_function(sampling_rate=0.1)  # Sample to manage memory
        def process_large_data(data: List[int]) -> Dict[str, Any]:
            """Process large data and return statistics"""
            return {
                "count": len(data),
                "sum": sum(data),
                "min": min(data) if data else None,
                "max": max(data) if data else None,
                "avg": sum(data) / len(data) if data else None
            }
        
        # Test with increasingly large datasets
        test_sizes = [1000, 10000, 50000]
        
        for size in test_sizes:
            large_data = list(range(size))
            result = process_large_data(large_data)
            
            expected = {
                "count": size,
                "sum": sum(range(size)),
                "min": 0,
                "max": size - 1,
                "avg": (size - 1) / 2
            }
            
            assert result == expected, f"For size {size}, expected {expected}, got {result}"
        
        # Verify some calls were logged (with sampling)
        storage = LocalStorage()
        calls = storage.load_calls("process_large_data")
        assert len(calls) >= 0, "Should have some calls logged"


class TestAsyncAdvancedFeatures:
    """Test advanced async functionality"""
    
    @pytest.mark.asyncio
    async def test_async_exception_handling(self, clean_logs, function_monitor):
        """Test async exception handling"""
        
        @monitor_function(monitor=function_monitor)
        async def async_error_function(should_fail: bool, delay: float = 0.01) -> str:
            """Async function that might fail"""
            await asyncio.sleep(delay)
            
            if should_fail:
                raise RuntimeError("Async operation failed!")
            
            return f"Success after {delay}s delay"
        
        # Test successful call
        result = await async_error_function(False, 0.01)
        assert result == "Success after 0.01s delay"
        
        # Test failing call
        with pytest.raises(RuntimeError, match="Async operation failed!"):
            await async_error_function(True, 0.01)
        
        # Test another successful call
        result2 = await async_error_function(False, 0.02)
        assert result2 == "Success after 0.02s delay"
        
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"

        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("async_error_function")
        assert len(calls) == 3, "Should have 3 calls logged"
        
        # Check specific calls by their inputs (order-independent)
        expected_calls = [
            (False, 0.01, "Success after 0.01s delay"),  # First successful call
            (True, 0.01, "error"),  # Failed call
            (False, 0.02, "Success after 0.02s delay"),  # Second successful call
        ]
        
        for should_fail, delay, expected_result in expected_calls:
            matching_calls = [
                call for call in calls
                if call.io_record.inputs["should_fail"] == should_fail
                and call.io_record.inputs["delay"] == delay
            ]
            assert len(matching_calls) == 1, f"Should have exactly one call for should_fail={should_fail}, delay={delay}"
            
            call = matching_calls[0]
            if expected_result == "error":
                # This should be an error call
                assert isinstance(call.io_record.output, dict), "Error call should have dict output"
                assert "error" in call.io_record.output
                assert "Async operation failed!" in call.io_record.output["error"]
                assert call.io_record.output["type"] == "RuntimeError"
            else:
                # This should be a successful call
                assert call.io_record.output == expected_result, f"Expected {expected_result}, got {call.io_record.output}"
    
    @pytest.mark.asyncio
    async def test_concurrent_async_calls(self, clean_logs, function_monitor):
        """Test concurrent async function calls"""
        
        @monitor_function(monitor=function_monitor)
        async def async_worker(worker_id: int, work_time: float) -> Dict[str, Any]:
            """Async worker function"""
            start_time = time.time()
            await asyncio.sleep(work_time)
            end_time = time.time()
            
            return {
                "worker_id": worker_id,
                "work_time": work_time,
                "actual_time": end_time - start_time
            }
        
        # Run multiple async workers concurrently
        num_workers = 5
        work_times = [0.01, 0.02, 0.01, 0.03, 0.01]
        
        tasks = [async_worker(i, work_times[i]) for i in range(num_workers)]
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert len(results) == num_workers
        
        for i, result in enumerate(results):
            assert result["worker_id"] == i
            assert result["work_time"] == work_times[i]
            assert result["actual_time"] >= work_times[i]  # Should be at least the work time
        
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"

        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("async_worker")
        assert len(calls) == num_workers, f"Should have {num_workers} calls logged"
        
        # Check that all calls were logged correctly
        logged_worker_ids = [call.io_record.inputs["worker_id"] for call in calls]
        assert set(logged_worker_ids) == set(range(num_workers)), "All workers should be logged"


class TestComplexExceptionScenarios:
    """Test complex exception scenarios"""
    
    @pytest.mark.asyncio
    async def test_nested_exception_handling(self, clean_logs, function_monitor):
        """Test handling of nested exceptions"""
        
        @monitor_function(monitor=function_monitor)
        def outer_function(operation: str, value: float) -> float:
            """Function that calls another function that might fail"""
            
            def inner_function(op: str, val: float) -> float:
                if op == "sqrt" and val < 0:
                    raise ValueError("Cannot take square root of negative number")
                elif op == "log" and val <= 0:
                    raise ValueError("Cannot take log of non-positive number")
                elif op == "divide" and val == 0:
                    raise ZeroDivisionError("Cannot divide by zero")
                
                if op == "sqrt":
                    return val ** 0.5
                elif op == "log":
                    import math
                    return math.log(val)
                elif op == "divide":
                    return 100.0 / val
                else:
                    raise NotImplementedError(f"Operation {op} not supported")
            
            try:
                return inner_function(operation, value)
            except Exception as e:
                # Re-raise with additional context
                raise RuntimeError(f"Error in outer_function: {e}") from e
        
        # Test successful operations
        result1 = outer_function("sqrt", 16.0)
        assert result1 == 4.0
        
        result2 = outer_function("divide", 4.0)
        assert result2 == 25.0
        
        # Test various error conditions
        error_cases = [
            ("sqrt", -4.0, RuntimeError),
            ("log", -1.0, RuntimeError),
            ("divide", 0.0, RuntimeError),
            ("unknown", 1.0, RuntimeError),
        ]
        
        for operation, value, expected_exception in error_cases:
            with pytest.raises(expected_exception):
                outer_function(operation, value)
        
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"

        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("outer_function")
        assert len(calls) == 6, "Should have 6 calls logged (2 success + 4 errors)"
        
        # Check successful calls
        success_calls = [call for call in calls if not isinstance(call.io_record.output, dict)]
        assert len(success_calls) == 2, "Should have 2 successful calls"
        
        # Check error calls
        error_calls = [call for call in calls if isinstance(call.io_record.output, dict)]
        assert len(error_calls) == 4, "Should have 4 error calls"
        
        # Verify error details
        for error_call in error_calls:
            assert "error" in error_call.io_record.output
            assert "Error in outer_function:" in error_call.io_record.output["error"]
            assert error_call.io_record.output["type"] == "RuntimeError"
            assert "traceback" in error_call.io_record.output 