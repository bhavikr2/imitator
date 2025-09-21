"""
Test TimeInterval filtering functionality for load_calls
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from imitator import monitor_function, LocalStorage
from imitator.types import TimeInterval, FunctionCall, FunctionSignature, IORecord


class TestTimeIntervalFiltering:
    """Test TimeInterval filtering across multiple days"""
    
    def test_load_calls_multi_day_filtering(self):
        """Test load_calls filtering across multiple days using call_id timestamps"""
        
        # Step 1: Create a temporary directory for test logs
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorage(log_dir=temp_dir, format="jsonl")
            
            # Step 1: Artificially generate logs over 3 days
            base_time = datetime(2024, 1, 15, 12, 0, 0)  # January 15, 2024 at noon
            
            # Day 1 logs (January 15)
            day1_calls = []
            for i in range(3):
                call_time = base_time + timedelta(hours=i)
                call_id = str(call_time.timestamp())
                
                function_call = FunctionCall(
                    function_signature=FunctionSignature(
                        name="test_function",
                        parameters={"x": "int"},
                        return_type="int"
                    ),
                    io_record=IORecord(
                        inputs={"x": i},
                        output=i * 2,
                        timestamp=call_time,
                        execution_time_ms=10.0
                    ),
                    call_id=call_id
                )
                day1_calls.append(function_call)
            
            # Day 2 logs (January 16)
            day2_base = base_time + timedelta(days=1)
            day2_calls = []
            for i in range(3):
                call_time = day2_base + timedelta(hours=i)
                call_id = str(call_time.timestamp())
                
                function_call = FunctionCall(
                    function_signature=FunctionSignature(
                        name="test_function",
                        parameters={"x": "int"},
                        return_type="int"
                    ),
                    io_record=IORecord(
                        inputs={"x": i + 10},
                        output=(i + 10) * 2,
                        timestamp=call_time,
                        execution_time_ms=15.0
                    ),
                    call_id=call_id
                )
                day2_calls.append(function_call)
            
            # Day 3 logs (January 17)
            day3_base = base_time + timedelta(days=2)
            day3_calls = []
            for i in range(3):
                call_time = day3_base + timedelta(hours=i)
                call_id = str(call_time.timestamp())
                
                function_call = FunctionCall(
                    function_signature=FunctionSignature(
                        name="test_function",
                        parameters={"x": "int"},
                        return_type="int"
                    ),
                    io_record=IORecord(
                        inputs={"x": i + 20},
                        output=(i + 20) * 2,
                        timestamp=call_time,
                        execution_time_ms=20.0
                    ),
                    call_id=call_id
                )
                day3_calls.append(function_call)
            
            # Write logs to files manually (simulating different days)
            all_calls = day1_calls + day2_calls + day3_calls
            
            # Group calls by date for file writing
            calls_by_date = {}
            for call in all_calls:
                date_str = call.io_record.timestamp.strftime("%Y%m%d")
                if date_str not in calls_by_date:
                    calls_by_date[date_str] = []
                calls_by_date[date_str].append(call)
            
            # Write each day's calls to separate files
            for date_str, calls in calls_by_date.items():
                log_file = Path(temp_dir) / f"test_function_{date_str}.jsonl"
                with open(log_file, "w", encoding="utf-8") as f:
                    for call in calls:
                        f.write(call.model_dump_json() + "\n")
            
            # Step 2: Request to load_calls starting from day 2
            day2_start = day2_base  # Start of day 2
            day3_end = day3_base + timedelta(hours=23, minutes=59, seconds=59)  # End of day 3
            
            time_interval = TimeInterval(
                start_time=day2_start,
                end_time=day3_end
            )
            
            # Load calls with time interval
            filtered_calls = storage.load_calls("test_function", time_interval=time_interval)
            
            # Step 3: Verify that calls from day 1 were not returned
            # Should only get calls from day 2 and day 3 (6 calls total)
            assert len(filtered_calls) == 6, f"Expected 6 calls (day 2 + day 3), got {len(filtered_calls)}"
            
            # Verify that no calls from day 1 are included
            day1_call_ids = {call.call_id for call in day1_calls}
            returned_call_ids = {call.call_id for call in filtered_calls}
            
            # Assert no overlap with day 1
            assert not day1_call_ids.intersection(returned_call_ids), \
                "Day 1 calls should not be returned when filtering from day 2 onwards"
            
            # Verify that day 2 and day 3 calls are included
            day2_call_ids = {call.call_id for call in day2_calls}
            day3_call_ids = {call.call_id for call in day3_calls}
            expected_call_ids = day2_call_ids.union(day3_call_ids)
            
            assert returned_call_ids == expected_call_ids, \
                "Should return exactly the calls from day 2 and day 3"
            
            # Verify the calls are ordered by timestamp (call_id)
            call_timestamps = [float(call.call_id) for call in filtered_calls]
            assert call_timestamps == sorted(call_timestamps), \
                "Calls should be ordered by timestamp"
            
            # Verify specific input values to ensure correct calls were returned
            input_values = [call.io_record.inputs["x"] for call in filtered_calls]
            expected_inputs = [10, 11, 12, 20, 21, 22]  # Day 2: 10,11,12; Day 3: 20,21,22
            assert input_values == expected_inputs, \
                f"Expected inputs {expected_inputs}, got {input_values}"
    
    def test_load_calls_with_start_time_only(self):
        """Test load_calls with only start_time (no end_time)"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorage(log_dir=temp_dir, format="jsonl")
            
            # Create calls spanning 2 hours
            base_time = datetime(2024, 1, 15, 12, 0, 0)
            all_calls = []
            
            for i in range(4):
                call_time = base_time + timedelta(minutes=30 * i)  # Every 30 minutes
                call_id = str(call_time.timestamp())
                
                function_call = FunctionCall(
                    function_signature=FunctionSignature(
                        name="test_function",
                        parameters={"x": "int"},
                        return_type="int"
                    ),
                    io_record=IORecord(
                        inputs={"x": i},
                        output=i * 2,
                        timestamp=call_time,
                        execution_time_ms=10.0
                    ),
                    call_id=call_id
                )
                all_calls.append(function_call)
            
            # Write to file
            date_str = base_time.strftime("%Y%m%d")
            log_file = Path(temp_dir) / f"test_function_{date_str}.jsonl"
            with open(log_file, "w", encoding="utf-8") as f:
                for call in all_calls:
                    f.write(call.model_dump_json() + "\n")
            
            # Request calls starting from 1 hour after base_time (should get last 2 calls)
            start_time = base_time + timedelta(hours=1)
            time_interval = TimeInterval(start_time=start_time)  # No end_time
            
            filtered_calls = storage.load_calls("test_function", time_interval=time_interval)
            
            # Should get the last 2 calls (at 1:00 and 1:30)
            assert len(filtered_calls) == 2, f"Expected 2 calls, got {len(filtered_calls)}"
            
            # Verify correct calls were returned
            input_values = [call.io_record.inputs["x"] for call in filtered_calls]
            assert input_values == [2, 3], f"Expected inputs [2, 3], got {input_values}"

    def test_load_calls_no_time_interval_uses_default(self):
        """Test that load_calls without time_interval uses default (last 1 hour)"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorage(log_dir=temp_dir, format="jsonl")
            
            # Create calls: some old (2 hours ago) and some recent (30 minutes ago)
            now = datetime.now()
            old_time = now - timedelta(hours=2)
            recent_time = now - timedelta(minutes=30)
            
            calls = []
            
            # Old call (should not be returned with default 1-hour window)
            old_call = FunctionCall(
                function_signature=FunctionSignature(
                    name="test_function",
                    parameters={"x": "int"},
                    return_type="int"
                ),
                io_record=IORecord(
                    inputs={"x": 1},
                    output=2,
                    timestamp=old_time,
                    execution_time_ms=10.0
                ),
                call_id=str(old_time.timestamp())
            )
            calls.append(old_call)
            
            # Recent call (should be returned)
            recent_call = FunctionCall(
                function_signature=FunctionSignature(
                    name="test_function",
                    parameters={"x": "int"},
                    return_type="int"
                ),
                io_record=IORecord(
                    inputs={"x": 2},
                    output=4,
                    timestamp=recent_time,
                    execution_time_ms=10.0
                ),
                call_id=str(recent_time.timestamp())
            )
            calls.append(recent_call)
            
            # Write to file
            date_str = now.strftime("%Y%m%d")
            log_file = Path(temp_dir) / f"test_function_{date_str}.jsonl"
            with open(log_file, "w", encoding="utf-8") as f:
                for call in calls:
                    f.write(call.model_dump_json() + "\n")
            
            # Load calls without specifying time_interval (should use default)
            filtered_calls = storage.load_calls("test_function")
            
            # Should only get the recent call (within last hour)
            assert len(filtered_calls) == 1, f"Expected 1 call, got {len(filtered_calls)}"
            assert filtered_calls[0].io_record.inputs["x"] == 2, \
                "Should return the recent call, not the old one"
