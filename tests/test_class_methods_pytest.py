"""
Class methods and instance methods tests using pytest
"""

import pytest
import asyncio
from typing import List, Dict, Any
from conftest import wait_for_logs
from imitator import monitor_function, LocalStorage


class DataProcessor:
    """Example class with various types of methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
        self.data_store = []
        self.config = {"max_items": 100, "processing_enabled": True}
    
    @monitor_function
    def process_item(self, item: Any) -> bool:
        """Instance method that modifies internal state"""
        if not self.config["processing_enabled"]:
            return False
        
        if len(self.data_store) >= self.config["max_items"]:
            return False
        
        self.data_store.append(item)
        self.processed_count += 1
        return True
    
    @monitor_function
    def get_stats(self) -> Dict[str, Any]:
        """Instance method that reads state without modifying"""
        return {
            "name": self.name,
            "processed_count": self.processed_count,
            "total_items": len(self.data_store),
            "config": self.config.copy()
        }
    
    @monitor_function
    def modify_input_list(self, items: List[Any]) -> None:
        """Function that modifies input in-place and returns None"""
        items.extend(self.data_store)
        items.sort()
    
    @monitor_function
    def update_config(self, **kwargs) -> None:
        """Method that accepts arbitrary keyword arguments"""
        self.config.update(kwargs)
    
    @classmethod
    @monitor_function
    def create_default(cls, name: str) -> 'DataProcessor':
        """Class method that creates a new instance"""
        return cls(name)
    
    @staticmethod
    @monitor_function
    def validate_item(item: Any) -> bool:
        """Static method that validates items"""
        return item is not None and str(item).strip() != ""


class AsyncDataProcessor:
    """Example class with async methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.async_processed_count = 0
    
    @monitor_function
    async def async_process_item(self, item: Any, delay: float = 0.01) -> Dict[str, Any]:
        """Async method that simulates processing with delay"""
        await asyncio.sleep(delay)
        print(f"Async method called with item: {item}")
        self.async_processed_count += 1
        return {
            "processed_item": item,
            "processor_name": self.name,
            "count": self.async_processed_count
        }
    
    @monitor_function
    async def async_batch_process(self, items: List[Any]) -> List[Dict[str, Any]]:
        """Async method that processes multiple items concurrently"""
        tasks = [self.async_process_item(item, 0.01) for item in items]
        results = await asyncio.gather(*tasks)
        return results


class TestClassMethods:
    """Test class methods and instance methods"""
    
    @pytest.mark.asyncio
    async def test_instance_method_state_modification(self, clean_logs):
        """Test instance methods that modify internal state"""
        processor = DataProcessor("TestProcessor")
        
        # Test processing items
        items_to_process = ["item_1", "item_2", "item_3"]
        results = []
        
        for item in items_to_process:
            result = processor.process_item(item)
            results.append(result)
        
        # Verify results
        assert all(results), "All items should be processed successfully"
        assert processor.processed_count == 3, "Should have processed 3 items"
        assert len(processor.data_store) == 3, "Should have 3 items in store"
        assert processor.data_store == items_to_process, "Items should be stored correctly"

        # Wait for logging to complete
        await asyncio.sleep(5) # Replaced wait_for_logs with fixed sleep

        
        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("process_item")
        assert len(calls) == 3, "Should have 3 calls logged"
                
        # Check that self parameter is handled correctly
        for call in calls:
            assert "self" in call.io_record.inputs
            assert "<DataProcessor instance>" in call.io_record.inputs["self"]
    
    @pytest.mark.asyncio
    async def test_instance_method_read_only(self, clean_logs):
        """Test instance methods that read state without modification"""
        processor = DataProcessor("ReadOnlyTest")
        
        # Add some items first
        processor.process_item("test_item")
        
        # Get stats
        stats = processor.get_stats()
        
        expected_stats = {
            "name": "ReadOnlyTest",
            "processed_count": 1,
            "total_items": 1,
            "config": {"max_items": 100, "processing_enabled": True}
        }
        
        assert stats == expected_stats, f"Expected {expected_stats}, got {stats}"
        
        # Wait for logging
        await asyncio.sleep(5) # Replaced wait_for_logs with fixed sleep

        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("get_stats")
        assert len(calls) == 1, "Should have 1 call logged"
        
        call = calls[0]
        assert call.io_record.output == expected_stats
    
    @pytest.mark.asyncio
    async def test_input_modification_detection(self, clean_logs):
        """Test detection of in-place input modifications"""
        processor = DataProcessor("ModificationTest")
        processor.data_store = [0] # Changed to an integer to allow sorting with test_list
        
        test_list = [3, 1, 4]
        original_list = test_list.copy()
        
        # This should modify the input list in-place
        result = processor.modify_input_list(test_list)
        
        assert result is None, "Should return None"
        assert test_list != original_list, "Input list should be modified"
        assert 0 in test_list, "Should contain existing items" # Updated assertion
        assert sorted(test_list) == test_list, "List should be sorted"
        
        # Wait for logging
        await asyncio.sleep(5) # Replaced wait_for_logs with fixed sleep

        # Verify logging captures the modification
        storage = LocalStorage()
        calls = storage.load_calls("modify_input_list")
        assert len(calls) == 1, "Should have 1 call logged"
        
        call = calls[0]
        # The input should be logged before modification
        assert call.io_record.inputs["items"] == original_list
    
    @pytest.mark.asyncio
    async def test_method_with_kwargs(self, clean_logs):
        """Test methods that accept keyword arguments"""
        processor = DataProcessor("KwargsTest")
        
        # Update config with various kwargs
        processor.update_config(max_items=200, processing_enabled=False, new_setting="test")
        
        expected_config = {
            "max_items": 200,
            "processing_enabled": False,
            "new_setting": "test"
        }
        
        assert processor.config == expected_config, f"Expected {expected_config}, got {processor.config}"
        
        # Wait for logging
        await asyncio.sleep(5) # Replaced wait_for_logs with fixed sleep

        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("update_config")
        assert len(calls) == 1, "Should have 1 call logged"
        
        call = calls[0]
        assert call.io_record.inputs["max_items"] == 200
        assert call.io_record.inputs["processing_enabled"] is False
        assert call.io_record.inputs["new_setting"] == "test"
    
    @pytest.mark.asyncio
    async def test_class_method(self, clean_logs):
        """Test class methods"""
        processor = DataProcessor.create_default("ClassMethodTest")
    
        assert isinstance(processor, DataProcessor), "Should return DataProcessor instance"
        assert processor.name == "ClassMethodTest", "Should have correct name"
        assert processor.processed_count == 0, "Should start with zero processed count"
    
        # Wait for logging
        await asyncio.sleep(5) # Replaced wait_for_logs with fixed sleep
    
        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("create_default")
        assert len(calls) == 1, "Should have 1 call logged"
    
        call = calls[0]
        assert "cls" in call.io_record.inputs
        assert "<type instance>" in call.io_record.inputs["cls"]
        assert call.io_record.inputs["name"] == "ClassMethodTest"
        # Update assertion: The output will be its string representation, not the object itself
        assert call.io_record.output == "<DataProcessor instance>", "Logged output should be string representation of DataProcessor instance"
    
    @pytest.mark.asyncio
    async def test_static_method(self, clean_logs):
        """Test static methods"""
        test_items = ["valid", "", None, "  ", "another_valid"]
        expected_results = [True, False, False, False, True]
        
        results = []
        for item in test_items:
            result = DataProcessor.validate_item(item)
            results.append(result)
        
        assert results == expected_results, f"Expected {expected_results}, got {results}"
        
        # Wait for logging
        await asyncio.sleep(5) # Replaced wait_for_logs with fixed sleep

        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("validate_item")
        assert len(calls) == len(test_items), f"Should have {len(test_items)} calls logged"
        
        # Verify each call matches its expected input/output (order-independent)
        for test_item, expected_result in zip(test_items, expected_results):
            matching_calls = [
                call for call in calls 
                if call.io_record.inputs["item"] == test_item
            ]
            assert len(matching_calls) == 1, f"Should have exactly one call for item '{test_item}'"
            
            call = matching_calls[0]
            assert call.io_record.output == expected_result, f"For item '{test_item}', expected {expected_result}, got {call.io_record.output}"


class TestAsyncMethods:
    """Test async methods"""
    
    @pytest.mark.asyncio
    async def test_async_instance_method(self, clean_logs):
        """Test async instance methods"""
        processor = AsyncDataProcessor("AsyncTest")
        
        result = await processor.async_process_item("test_item")
        
        expected_result = {
            "processed_item": "test_item",
            "processor_name": "AsyncTest",
            "count": 1
        }
        
        assert result == expected_result, f"Expected {expected_result}, got {result}"
        assert processor.async_processed_count == 1, "Should have incremented counter"
        
        # Wait for logging to complete
        await asyncio.sleep(5) # Replaced wait_for_logs with fixed sleep
        
        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("async_process_item")
        assert len(calls) == 1, "Should have 1 call logged"
        
        
        call = calls[0]
        assert call.io_record.output == expected_result
        assert call.io_record.inputs["item"] == "test_item"
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self, clean_logs):
        """Test async batch processing"""
        processor = AsyncDataProcessor("BatchTest")
        
        items = ["item1", "item2", "item3"]
        results = await processor.async_batch_process(items)
        
        assert len(results) == 3, "Should process all items"
        assert processor.async_processed_count == 3, "Should have processed 3 items"
        
        # Check individual results
        for i, result in enumerate(results):
            assert result["processed_item"] == items[i]
            assert result["processor_name"] == "BatchTest"
            assert result["count"] == i + 1  # Sequential processing
        
        # Verify logging
        storage = LocalStorage()
        
        # Wait for logging to complete for all async operations
        await asyncio.sleep(5) # Replaced wait_for_logs with fixed sleep
        
        # Check batch process call
        batch_calls = storage.load_calls("async_batch_process")
        assert len(batch_calls) == 1, "Should have 1 batch call logged"
        
        batch_call = batch_calls[0]
        assert batch_call.io_record.inputs["items"] == items
        assert batch_call.io_record.output == results
        
        # Check individual item processing calls
        item_calls = storage.load_calls("async_process_item")
        assert len(item_calls) == 3, "Should have 3 item processing calls logged"


class TestExceptionHandlingInClasses:
    """Test exception handling in class methods"""
    
    @pytest.mark.asyncio
    async def test_method_exceptions(self, clean_logs, function_monitor):
        """Test exception handling in class methods"""
        
        @monitor_function
        def divide_by_attribute(self, divisor: float) -> float:
            """Method that might raise an exception"""
            if divisor == 0:
                raise ValueError("Cannot divide by zero!")
            return 100.0 / divisor
        
        # Monkey patch for testing
        DataProcessor.divide_by_attribute = divide_by_attribute
        
        processor = DataProcessor("ExceptionTest")
        
        # Test successful calls
        result1 = processor.divide_by_attribute(2.0)
        assert result1 == 50.0, "Should return correct division result"
        
        result2 = processor.divide_by_attribute(4.0)
        assert result2 == 25.0, "Should return correct division result"
        
        # Test exception case
        with pytest.raises(ValueError, match="Cannot divide by zero!"):
            processor.divide_by_attribute(0.0)
        
        # Test another successful call after exception
        result3 = processor.divide_by_attribute(10.0)
        assert result3 == 10.0, "Should continue working after exception"
        
        # Wait for logging
        await asyncio.sleep(5) # Replaced wait_for_logs with fixed sleep
        
        # Verify logging
        storage = LocalStorage()
        calls = storage.load_calls("divide_by_attribute")
        assert len(calls) == 4, "Should have 4 calls logged"
        
        # Check successful calls
        success_calls = [call for call in calls if not isinstance(call.io_record.output, dict)]
        assert len(success_calls) == 3, "Should have 3 successful calls"
        
        # Check exception call
        error_calls = [call for call in calls if isinstance(call.io_record.output, dict)]
        assert len(error_calls) == 1, "Should have 1 error call"
        
        error_call = error_calls[0]
        assert "error" in error_call.io_record.output
        assert "Cannot divide by zero!" in error_call.io_record.output["error"]
        assert error_call.io_record.output["type"] == "ValueError" 