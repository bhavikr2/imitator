"""
Function monitoring decorator and utilities
"""

import time
import asyncio
from functools import wraps
from typing import Callable, Optional, Any
import threading
from collections import defaultdict
from datetime import datetime, timedelta

from .types import FunctionCall, FunctionSignature, IORecord
from .storage import LocalStorage


class FunctionMonitor:
    """Central function monitoring system"""
    
    def __init__(self, storage: Optional[LocalStorage] = None, 
                 sampling_rate: float = 1.0, 
                 max_calls_per_minute: Optional[int] = None):
        """
        Initialize function monitor
        
        Args:
            storage: Storage backend, defaults to LocalStorage
            sampling_rate: Fraction of calls to log (0.0 to 1.0)
            max_calls_per_minute: Maximum calls to log per minute per function
        """
        self.storage = storage or LocalStorage()
        self.sampling_rate = sampling_rate
        self.max_calls_per_minute = max_calls_per_minute
        self._call_counts = defaultdict(list)  # Track calls per function
        self._lock = threading.Lock()
        self._active_threads = set()  # Track active save threads
        
        # Import here to avoid circular imports
        import random
        self._random = random
    
    def _should_log(self, function_name: str) -> bool:
        """Determine if this call should be logged based on sampling rules"""
        
        # Check sampling rate
        if self.sampling_rate < 1.0 and self._random.random() > self.sampling_rate:
            return False
        
        # Check rate limiting
        if self.max_calls_per_minute is not None:
            with self._lock:
                now = datetime.now()
                # Clean old entries (older than 1 minute)
                cutoff = now - timedelta(minutes=1)
                self._call_counts[function_name] = [
                    t for t in self._call_counts[function_name] if t > cutoff
                ]
                
                # Check if we've exceeded the limit
                if len(self._call_counts[function_name]) >= self.max_calls_per_minute:
                    return False
                
                # Add current call
                self._call_counts[function_name].append(now)
        
        return True
    
    def monitor(self, func: Callable) -> Callable:
        """
        Decorator to monitor function I/O
        
        Args:
            func: Function to monitor
        
        Returns:
            Wrapped function with monitoring
        """
        # Check if it's an async function
        if asyncio.iscoroutinefunction(func):
            return self._monitor_async(func)
        else:
            return self._monitor_sync(func)
    
    def _monitor_sync(self, func: Callable) -> Callable:
        """Monitor synchronous function"""
        # Extract function signature once
        signature = FunctionSignature.from_function(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if we should log this call
            if not self._should_log(signature.name):
                return func(*args, **kwargs)
            
            # Record start time
            start_time = time.time()
            
            # Prepare inputs for logging
            inputs = self._prepare_inputs(signature, args, kwargs)
            
            # Create deep copy of inputs for comparison (in case of in-place modification)
            import copy
            inputs_before = copy.deepcopy(inputs)
            
            # Execute the function
            try:
                result = func(*args, **kwargs)
                
                # Record execution time
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Check for in-place modifications
                inputs_after = copy.deepcopy(inputs)
                modifications = self._detect_modifications(inputs_before, inputs_after)
                
                # Prepare output for serialization
                serializable_output = self._prepare_output(result)
                
                # Create I/O record
                io_record = IORecord(
                    inputs=inputs_before,
                    output=serializable_output,
                    execution_time_ms=execution_time,
                    input_modifications=modifications if modifications else None
                )
                
                # Create function call record
                function_call = FunctionCall(
                    function_signature=signature,
                    io_record=io_record
                )
                
                # Save to storage (async to minimize overhead)
                self._save_async(function_call)
                
                return result
                
            except Exception as e:
                # Record execution time even for exceptions
                execution_time = (time.time() - start_time) * 1000
                
                # Create I/O record with exception
                io_record = IORecord(
                    inputs=inputs_before,
                    output={"error": str(e), "type": type(e).__name__, "traceback": self._get_traceback()},
                    execution_time_ms=execution_time
                )
                
                # Create function call record
                function_call = FunctionCall(
                    function_signature=signature,
                    io_record=io_record
                )
                
                # Save to storage
                self._save_async(function_call)
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    def _monitor_async(self, func: Callable) -> Callable:
        """Monitor asynchronous function"""
        # Extract function signature once
        signature = FunctionSignature.from_function(func)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if we should log this call
            if not self._should_log(signature.name):
                return await func(*args, **kwargs)
            
            # Record start time
            start_time = time.time()
            
            # Prepare inputs for logging
            inputs = self._prepare_inputs(signature, args, kwargs)
            
            # Create deep copy of inputs for comparison
            import copy
            inputs_before = copy.deepcopy(inputs)
            
            # Execute the function
            try:
                result = await func(*args, **kwargs)
                
                # Record execution time
                execution_time = (time.time() - start_time) * 1000
                
                # Check for in-place modifications
                inputs_after = copy.deepcopy(inputs)
                modifications = self._detect_modifications(inputs_before, inputs_after)
                
                # Prepare output for serialization
                serializable_output = self._prepare_output(result)
                
                # Create I/O record
                io_record = IORecord(
                    inputs=inputs_before,
                    output=serializable_output,
                    execution_time_ms=execution_time,
                    input_modifications=modifications if modifications else None
                )
                
                # Create function call record
                function_call = FunctionCall(
                    function_signature=signature,
                    io_record=io_record
                )
                
                # Save to storage (async to minimize overhead)
                self._save_async(function_call)
                
                return result
                
            except Exception as e:
                # Record execution time even for exceptions
                execution_time = (time.time() - start_time) * 1000
                
                # Create I/O record with exception
                io_record = IORecord(
                    inputs=inputs_before,
                    output={"error": str(e), "type": type(e).__name__, "traceback": self._get_traceback()},
                    execution_time_ms=execution_time
                )
                
                # Create function call record
                function_call = FunctionCall(
                    function_signature=signature,
                    io_record=io_record
                )
                
                # Save to storage
                self._save_async(function_call)
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    def _prepare_inputs(self, signature: FunctionSignature, args: tuple, kwargs: dict) -> dict:
        """Prepare inputs for logging, handling class methods appropriately"""
        inputs = {}
        
        # Map positional arguments to parameter names
        param_names = list(signature.parameters.keys())
        for i, arg in enumerate(args):
            if i < len(param_names):
                param_name = param_names[i]
                # For class methods, don't log 'self' or 'cls' - just record the class name
                if param_name in ('self', 'cls'):
                    inputs[param_name] = f"<{type(arg).__name__} instance>"
                else:
                    inputs[param_name] = arg
        
        # Add keyword arguments
        inputs.update(kwargs)
        
        return inputs
    
    def _prepare_output(self, output: Any) -> Any:
        """Prepare output for serialization, handling complex objects"""
        try:
            # Try to serialize with json to check if it's already serializable
            import json
            json.dumps(output, default=str)
            return output
        except (TypeError, ValueError):
            # If it's not serializable, convert to string representation
            if hasattr(output, '__dict__'):
                # For custom class instances, return a representation
                return f"<{type(output).__name__} instance>"
            else:
                # For other non-serializable types, convert to string
                return str(output)
    
    def _detect_modifications(self, before: dict, after: dict) -> Optional[dict]:
        """Detect if any input parameters were modified in-place"""
        modifications = {}
        
        for key, value_before in before.items():
            if key in after:
                value_after = after[key]
                if value_before != value_after:
                    modifications[key] = {
                        "before": value_before,
                        "after": value_after
                    }
        
        return modifications if modifications else None
    
    def _get_traceback(self) -> str:
        """Get traceback string for exceptions"""
        import traceback
        return traceback.format_exc()
    
    def _save_async(self, function_call: FunctionCall):
        """Save function call asynchronously in a thread"""
        def  save_in_thread():
            try:
                self.storage.save_call(function_call)
            except Exception as e:
            finally:
                # Remove thread from active set when done
                with self._lock:
                    self._active_threads.discard(thread)
        
        thread = threading.Thread(target=save_in_thread)
        thread.daemon = False  # Non-daemon so threads complete before exit
        
        # Add to active threads set
        with self._lock:
            self._active_threads.add(thread)
        thread.start()
    
    def wait_for_all_saves(self, timeout: float = 5.0) -> bool:
        """
        Wait for all pending save operations to complete
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all saves completed, False if timeout occurred
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                active_threads = list(self._active_threads)
            
            if not active_threads:
                return True
                
            # Wait a bit for threads to complete
            time.sleep(0.01)
            
            # Clean up any finished threads
            with self._lock:
                self._active_threads = {t for t in self._active_threads if t.is_alive()}
        
        return False
    



# Global monitor instance for convenience
_global_monitor = FunctionMonitor()


def monitor_function(func: Optional[Callable] = None, *, 
                    storage: Optional[LocalStorage] = None,
                    sampling_rate: float = 1.0,
                    max_calls_per_minute: Optional[int] = None) -> Callable:
    """
    Decorator to monitor function I/O
    
    Can be used as:
    @monitor_function
    def my_func():
        pass
    
    Or:
    @monitor_function(sampling_rate=0.1, max_calls_per_minute=100)
    def my_func():
        pass
    
    Args:
        func: Function to monitor (for direct decoration)
        storage: Custom storage backend
        sampling_rate: Fraction of calls to log (0.0 to 1.0)
        max_calls_per_minute: Maximum calls to log per minute
    
    Returns:
        Decorated function or decorator
    """
    def decorator(f: Callable) -> Callable:
        monitor = FunctionMonitor(storage, sampling_rate, max_calls_per_minute)
        return monitor.monitor(f)
    
    if func is None:
        # Called with arguments: @monitor_function(sampling_rate=...)
        return decorator
    else:
        # Called without arguments: @monitor_function
        return decorator(func)


def get_monitor(storage: Optional[LocalStorage] = None, 
                sampling_rate: float = 1.0,
                max_calls_per_minute: Optional[int] = None) -> FunctionMonitor:
    """Get a function monitor instance"""
    return FunctionMonitor(storage, sampling_rate, max_calls_per_minute)


def wait_for_all_saves(timeout: float = 5.0) -> bool:
    """
    Wait for all pending save operations from the global monitor to complete
    
    Args:
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if all saves completed, False if timeout occurred
    """
    return _global_monitor.wait_for_all_saves(timeout) 