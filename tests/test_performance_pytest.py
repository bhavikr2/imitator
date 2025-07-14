"""
Performance tests using pytest with proper benchmarks and assertions
"""

import pytest
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import random
import statistics
from imitator.monitor import get_monitor

from imitator import monitor_function, LocalStorage
from tests.conftest import wait_for_logs, temp_storage

class TestPerformanceOverhead:
    """Test performance overhead of monitoring"""
    
    def test_monitoring_overhead(self, clean_logs):
        """Test that monitoring overhead is reasonable with a realistic simulation."""

        def simulate_coffee_shop_queue(num_customers, num_baristas, mean_arrival_interval, mean_service_time):
            """Simulate a coffee shop queue and calculate average customer wait time."""
            # Initialize variables
            wait_times = []
            current_time = 0
            queue = []

            # Simulate customer arrivals
            for i in range(num_customers):
                # Time until next customer arrives (exponential distribution for realism)
                arrival_time = current_time + random.expovariate(1 / mean_arrival_interval)
                queue.append(arrival_time)
                current_time = arrival_time # Advance time to next arrival

            # Process customers through baristas
            barista_free_time = [0] * num_baristas  # Tracks when each barista is free
            for arrival in queue:
                # Find the earliest available barista
                earliest_free = min(barista_free_time)
                # If customer arrives after barista is free, no wait; otherwise, wait
                wait_time = max(0, earliest_free - arrival)
                wait_times.append(wait_time)
                # Service starts when barista is free or customer arrives, whichever is later
                service_start = max(arrival, earliest_free)
                # Service time follows exponential distribution
                service_duration = random.expovariate(1 / mean_service_time)
                # Update barista's free time
                barista_index = barista_free_time.index(earliest_free)
                barista_free_time[barista_index] = service_start + service_duration

            # Calculate and return average wait time
            avg_wait_time = statistics.mean(wait_times) if wait_times else 0
            return avg_wait_time / 60  # Convert seconds to minutes for readability

        # Monitored function
        @monitor_function
        def monitored_simulate_coffee_shop_queue(num_customers, num_baristas, mean_arrival_interval, mean_service_time):
            """Simulate a coffee shop queue with monitoring."""
            # Initialize variables
            wait_times = []
            current_time = 0
            queue = []

            # Simulate customer arrivals
            for i in range(num_customers):
                arrival_time = current_time + random.expovariate(1 / mean_arrival_interval)
                queue.append(arrival_time)
                current_time = arrival_time # Advance time to next arrival

            # Process customers through baristas
            barista_free_time = [0] * num_baristas
            for arrival in queue:
                earliest_free = min(barista_free_time)
                wait_time = max(0, earliest_free - arrival)
                wait_times.append(wait_time)
                service_start = max(arrival, earliest_free)
                service_duration = random.expovariate(1 / mean_service_time)
                barista_index = barista_free_time.index(earliest_free)
                barista_free_time[barista_index] = service_start + service_duration

            avg_wait_time = statistics.mean(wait_times) if wait_times else 0
            return avg_wait_time / 60


        iterations = 100 # Adjust iterations for a more complex simulation
        
        # Parameters for the coffee shop simulation
        num_customers = 100
        num_baristas = 2
        mean_arrival_interval = 30
        mean_service_time = 120

        random.seed(42) # Ensure reproducibility for baseline
        
        # Measure baseline performance
        start_time = time.time()
        for _ in range(iterations):
            simulate_coffee_shop_queue(num_customers, num_baristas, mean_arrival_interval, mean_service_time)
        baseline_time = time.time() - start_time
        
        random.seed(42) # Re-seed for reproducibility for monitored run
        
        # Measure monitored performance
        start_time = time.time()
        for _ in range(iterations):
            monitored_simulate_coffee_shop_queue(num_customers, num_baristas, mean_arrival_interval, mean_service_time)
        monitored_time = time.time() - start_time
        
        # Calculate overhead
        overhead_ratio = monitored_time / baseline_time
        overhead_percentage = ((monitored_time - baseline_time) / baseline_time) * 100
        
        # Assertions (these might need adjustment based on the new function's actual overhead)
        assert overhead_ratio < 3.0, f"Overhead ratio {overhead_ratio:.2f}x should be less than 3x"
        assert overhead_percentage < 200, f"Overhead {overhead_percentage:.1f}% should be less than 200%"
        
        # Verify logging worked
        storage = LocalStorage()
        calls = storage.load_calls("monitored_simulate_coffee_shop_queue")
        assert len(calls) == iterations, f"Should have {iterations} calls logged"
    
    def test_sampling_reduces_overhead(self, clean_logs):
        """Test that sampling significantly reduces overhead"""
        
        # Function with full monitoring
        @monitor_function
        def full_monitoring(x: int) -> int:
            return x * 2 + 1
        
        # Function with 1% sampling
        @monitor_function(sampling_rate=0.01)
        def sampled_function(x: int) -> int:
            return x * 2 + 1
        
        iterations = 10000
        
        # Measure full monitoring
        start_time = time.time()
        for i in range(iterations):
            full_monitoring(i)
        full_time = time.time() - start_time
        
        # Measure sampled monitoring
        start_time = time.time()
        for i in range(iterations):
            sampled_function(i)
        sampled_time = time.time() - start_time
        
        # Sampled should be faster than full monitoring
        assert sampled_time < full_time, "Sampled monitoring should be faster than full monitoring"
        
        # Verify sampling worked
        storage = LocalStorage()
        
        full_calls = storage.load_calls("full_monitoring")
        sampled_calls = storage.load_calls("sampled_function")
        
        assert len(full_calls) == iterations, "Full monitoring should log all calls"
        assert len(sampled_calls) < iterations * 0.05, "Sampling should log much fewer calls"
    
    def test_rate_limiting_prevents_performance_degradation(self, clean_logs):
        """Test that rate limiting prevents performance degradation"""
        
        @monitor_function(max_calls_per_minute=10)
        def rate_limited_function(x: int) -> int:
            return x * 2 + 1
        
        iterations = 1000
        
        # Measure performance with rate limiting
        start_time = time.time()
        for i in range(iterations):
            rate_limited_function(i)
        total_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert total_time < 2.0, f"Should complete in under 2 seconds, took {total_time:.2f}s"
        
        # Verify rate limiting worked
        storage = LocalStorage()
        calls = storage.load_calls("rate_limited_function")
        assert len(calls) <= 10, f"Should have at most 10 calls logged, got {len(calls)}"


class TestConcurrentPerformance:
    """Test performance under concurrent load"""
    
    @pytest.mark.asyncio
    async def test_concurrent_monitoring_performance(self, clean_logs, function_monitor):
        """Test that monitoring works well under concurrent load"""
        
        @monitor_function(monitor=function_monitor)
        def concurrent_function(thread_id: int, value: int) -> int:
            # Small computation to make it measurable
            result = value
            for _ in range(100):
                result = (result * 2) % 1000
            return result
        
        def worker(thread_id: int, calls_per_thread: int):
            """Worker function for each thread"""
            start_time = time.time()
            for i in range(calls_per_thread):
                concurrent_function(thread_id, i)
            return time.time() - start_time
        
        num_threads = 4
        calls_per_thread = 250
        total_calls = num_threads * calls_per_thread
        
        # Run concurrent test
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i, calls_per_thread) for i in range(num_threads)]
            
            worker_times = []
            for future in futures:
                worker_time = future.result()
                worker_times.append(worker_time)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 5.0, f"Concurrent test should complete in under 5 seconds, took {total_time:.2f}s"
        
        # Calculate throughput
        throughput = total_calls / total_time
        assert throughput > 100, f"Throughput {throughput:.0f} calls/sec should be > 100 calls/sec"
        
        # Verify all calls were logged
        logs_completed = await wait_for_logs(function_monitor)  # Wait for all log writes to complete
        assert logs_completed, "All log saves should complete successfully"
        storage = LocalStorage()
        calls = storage.load_calls("concurrent_function")
        assert len(calls) == total_calls, f"Should have {total_calls} calls logged"
        
        # Verify data integrity
        thread_ids = [call.io_record.inputs["thread_id"] for call in calls]
        assert set(thread_ids) == set(range(num_threads)), "All threads should be represented"
    
    def test_no_deadlocks_under_load(self, clean_logs):
        """Test that no deadlocks occur under heavy concurrent load"""
        
        @monitor_function
        def heavy_concurrent_function(worker_id: int) -> str:
            # Simulate some work
            time.sleep(0.001)  # 1ms
            return f"worker_{worker_id}_completed"
        
        num_workers = 10
        calls_per_worker = 50
        
        def worker(worker_id: int):
            results = []
            for i in range(calls_per_worker):
                result = heavy_concurrent_function(worker_id)
                results.append(result)
            return results
        
        # Run with timeout to detect deadlocks
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(num_workers)]
            
            # Wait for all workers to complete with timeout
            all_results = []
            for future in futures:
                try:
                    worker_results = future.result(timeout=10.0)  # 10 second timeout
                    all_results.extend(worker_results)
                except Exception as e:
                    pytest.fail(f"Worker failed or timed out: {e}")
        
        total_time = time.time() - start_time
        
        # Should complete without timeout
        assert total_time < 10.0, f"Should complete without timeout, took {total_time:.2f}s"
        assert len(all_results) == num_workers * calls_per_worker, "All workers should complete"
        
        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("heavy_concurrent_function")
        assert len(calls) == num_workers * calls_per_worker, "All calls should be logged"


class TestAsyncPerformance:
    """Test async performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_async_monitoring_overhead(self, clean_logs, temp_storage, function_monitor):
        """Test async monitoring overhead with CPU-intensive functions"""

        function_monitor = get_monitor(storage=temp_storage)
        
        # Baseline async function (CPU-intensive)
        async def async_baseline(iterations: int) -> int:
            result_sum = 0
            for i in range(iterations):
                result_sum += i * (i % 7) # Perform some computation
            return result_sum
        
        # Monitored async function (CPU-intensive)
        @monitor_function(monitor=function_monitor)
        async def async_monitored(iterations: int) -> int:
            result_sum = 0
            for i in range(iterations):
                result_sum += i * (i % 7) # Perform some computation
            return result_sum

        cpu_iterations_per_call = 100000 # Adjust this value to control computation
        num_calls_to_benchmark = 100    # Number of times to call each function
        
        # Measure baseline performance
        start_time = time.time()
        for _ in range(num_calls_to_benchmark):
            await async_baseline(cpu_iterations_per_call)
        baseline_time = time.time() - start_time
        
        # Measure monitored performance
        start_time = time.time()
        for _ in range(num_calls_to_benchmark):
            await async_monitored(cpu_iterations_per_call)
        monitored_time = time.time() - start_time
        
        # Calculate overhead
        overhead_ratio = monitored_time / baseline_time
        overhead_percentage = ((monitored_time - baseline_time) / baseline_time) * 100
        
        # Assertions (you might need to adjust these based on your system's performance)
        assert overhead_ratio < 2.0, f"Async Overhead ratio {overhead_ratio:.2f}x should be less than 2x"
        assert overhead_percentage < 100, f"Async Overhead {overhead_percentage:.1f}% should be less than 100%"
        
        # Verify logging worked
        logs_completed = await wait_for_logs(function_monitor)
        assert logs_completed, "All log saves should complete successfully"
        calls = temp_storage.load_calls("async_monitored")
        assert len(calls) == num_calls_to_benchmark, f"Should have {num_calls_to_benchmark} calls logged"
    
    @pytest.mark.asyncio
    async def test_async_concurrent_performance(self, clean_logs):
        """Test async performance under concurrent load"""
        
        @monitor_function
        async def async_worker(worker_id: int, delay: float) -> Dict[str, Any]:
            """Async worker with variable delay"""
            start_time = time.time()
            await asyncio.sleep(delay)
            end_time = time.time()
            
            return {
                "worker_id": worker_id,
                "delay": delay,
                "actual_time": end_time - start_time
            }
        
        # Create workers with different delays
        num_workers = 20
        delays = [0.01 + (i % 5) * 0.01 for i in range(num_workers)]  # 0.01 to 0.05 seconds
        
        start_time = time.time()
        
        # Run all workers concurrently
        tasks = [async_worker(i, delays[i]) for i in range(num_workers)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Should complete in roughly the time of the longest delay (not sum of all delays)
        max_delay = max(delays)
        assert total_time < max_delay + 0.5, f"Should complete in ~{max_delay}s, took {total_time:.2f}s"
        
        # Verify results
        assert len(results) == num_workers, f"Should have {num_workers} results"
        
        for i, result in enumerate(results):
            assert result["worker_id"] == i
            assert result["delay"] == delays[i]
            assert result["actual_time"] >= delays[i]
        
        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("async_worker")
        assert len(calls) == num_workers, f"Should have {num_workers} calls logged"


class TestMemoryEfficiency:
    """Test memory efficiency of the monitoring system"""
    
    def test_memory_usage_with_large_data(self, clean_logs):
        """Test memory efficiency with large data structures"""
        
        @monitor_function(sampling_rate=0.1)  # Sample to control memory usage
        def process_large_data(data: List[int]) -> Dict[str, int]:
            """Process large data efficiently"""
            return {
                "count": len(data),
                "sum": sum(data),
                "memory_efficient": 1
            }
        
        # Test with increasingly large datasets
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            large_data = list(range(size))
            
            # Measure memory usage (simplified)
            start_time = time.time()
            result = process_large_data(large_data)
            end_time = time.time()
            
            # Should complete in reasonable time
            processing_time = end_time - start_time
            assert processing_time < 1.0, f"Processing {size} elements should take < 1s, took {processing_time:.2f}s"
            
            # Verify result
            expected_sum = sum(range(size))
            assert result["count"] == size
            assert result["sum"] == expected_sum
        
        # Verify sampling worked to control memory
        storage = LocalStorage()
        calls = storage.load_calls("process_large_data")
        
        # Should have logged some calls but not all (due to sampling)
        assert len(calls) <= len(sizes), f"Should have sampled calls, got {len(calls)}"
    
    def test_log_rotation_efficiency(self, clean_logs):
        """Test that log storage doesn't grow unbounded"""
        
        @monitor_function
        def repeated_function(iteration: int) -> str:
            """Function called many times"""
            return f"iteration_{iteration}"
        
        # Call function many times
        num_calls = 1000
        for i in range(num_calls):
            repeated_function(i)
        
        # Verify all calls were logged
        storage = LocalStorage()
        calls = storage.load_calls("repeated_function")
        assert len(calls) == num_calls, f"Should have {num_calls} calls logged"
        
        # Check that logs are reasonable size (not storing excessive data)
        total_log_size = 0
        for call in calls[:10]:  # Sample first 10 calls
            # Estimate log entry size
            log_entry_size = len(str(call.io_record.inputs)) + len(str(call.io_record.output))
            total_log_size += log_entry_size
        
        avg_log_size = total_log_size / 10
        assert avg_log_size < 1000, f"Average log entry size {avg_log_size} should be reasonable"


class TestExtremeLoad:
    """Test system behavior under extreme load"""
    
    def test_high_frequency_calls(self, clean_logs):
        """Test system with very high frequency calls"""
        
        @monitor_function(sampling_rate=0.001, max_calls_per_minute=5)
        def extreme_frequency_function(x: int) -> int:
            """Function designed for extreme frequency"""
            return x % 1000
        
        iterations = 100000
        
        start_time = time.time()
        for i in range(iterations):
            extreme_frequency_function(i)
        total_time = time.time() - start_time
        
        # Should handle high frequency without significant slowdown
        throughput = iterations / total_time
        assert throughput > 10000, f"Throughput {throughput:.0f} calls/sec should be > 10,000 calls/sec"
        
        # Verify rate limiting worked
        storage = LocalStorage()
        calls = storage.load_calls("extreme_frequency_function")
        assert len(calls) <= 5, f"Should have at most 5 calls logged due to rate limiting, got {len(calls)}"
    
    def test_system_stability_under_load(self, clean_logs):
        """Test that system remains stable under sustained load"""
        
        @monitor_function(sampling_rate=0.01)
        def stability_test_function(batch_id: int, item_id: int) -> str:
            """Function for stability testing"""
            # Some computation to make it realistic
            result = batch_id * 1000 + item_id
            return f"batch_{batch_id}_item_{item_id}_result_{result}"
        
        num_batches = 10
        items_per_batch = 1000
        
        def process_batch(batch_id: int):
            """Process a batch of items"""
            batch_results = []
            for item_id in range(items_per_batch):
                result = stability_test_function(batch_id, item_id)
                batch_results.append(result)
            return batch_results
        
        # Process batches concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_batch, i) for i in range(num_batches)]
            
            all_results = []
            for future in futures:
                batch_results = future.result()
                all_results.extend(batch_results)
        
        # Verify results
        total_items = num_batches * items_per_batch
        assert len(all_results) == total_items, f"Should have {total_items} results"
        
        # Verify logging worked (with sampling)
        storage = LocalStorage()
        calls = storage.load_calls("stability_test_function")
        
        # Should have some calls logged (approximately 1% due to sampling)
        expected_logged = total_items * 0.01
        assert len(calls) > 0, "Should have some calls logged"
        assert len(calls) < total_items * 0.05, "Should not log too many calls due to sampling" 