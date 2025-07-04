"""
Function monitoring decorator and utilities
"""

import time
from functools import wraps
from typing import Callable, Optional, Any

from .types import FunctionCall, FunctionSignature, IORecord
from .storage import LocalStorage


class FunctionMonitor:
    """Central function monitoring system"""
    
    def __init__(self, storage: Optional[LocalStorage] = None):
        """
        Initialize function monitor
        
        Args:
            storage: Storage backend, defaults to LocalStorage
        """
        self.storage = storage or LocalStorage()
    
    def monitor(self, func: Callable) -> Callable:
        """
        Decorator to monitor function I/O
        
        Args:
            func: Function to monitor
        
        Returns:
            Wrapped function with monitoring
        """
        # Extract function signature once
        signature = FunctionSignature.from_function(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Record start time
            start_time = time.time()
            
            # Prepare inputs for logging
            inputs = {}
            
            # Map positional arguments to parameter names
            param_names = list(signature.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(param_names):
                    inputs[param_names[i]] = arg
            
            # Add keyword arguments
            inputs.update(kwargs)
            
            # Execute the function
            try:
                result = func(*args, **kwargs)
                
                # Record execution time
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Create I/O record
                io_record = IORecord(
                    inputs=inputs,
                    output=result,
                    execution_time_ms=execution_time
                )
                
                # Create function call record
                function_call = FunctionCall(
                    function_signature=signature,
                    io_record=io_record
                )
                
                # Save to storage
                self.storage.save_call(function_call)
                
                return result
                
            except Exception as e:
                # Record execution time even for exceptions
                execution_time = (time.time() - start_time) * 1000
                
                # Create I/O record with exception
                io_record = IORecord(
                    inputs=inputs,
                    output={"error": str(e), "type": type(e).__name__},
                    execution_time_ms=execution_time
                )
                
                # Create function call record
                function_call = FunctionCall(
                    function_signature=signature,
                    io_record=io_record
                )
                
                # Save to storage
                self.storage.save_call(function_call)
                
                # Re-raise the exception
                raise
        
        return wrapper


# Global monitor instance for convenience
_global_monitor = FunctionMonitor()


def monitor_function(func: Optional[Callable] = None, *, storage: Optional[LocalStorage] = None) -> Callable:
    """
    Decorator to monitor function I/O
    
    Can be used as:
    @monitor_function
    def my_func():
        pass
    
    Or:
    @monitor_function(storage=custom_storage)
    def my_func():
        pass
    
    Args:
        func: Function to monitor (for direct decoration)
        storage: Custom storage backend
    
    Returns:
        Decorated function or decorator
    """
    def decorator(f: Callable) -> Callable:
        if storage is not None:
            monitor = FunctionMonitor(storage)
            return monitor.monitor(f)
        else:
            return _global_monitor.monitor(f)
    
    if func is None:
        # Called with arguments: @monitor_function(storage=...)
        return decorator
    else:
        # Called without arguments: @monitor_function
        return decorator(func)


def get_monitor(storage: Optional[LocalStorage] = None) -> FunctionMonitor:
    """Get a function monitor instance"""
    return FunctionMonitor(storage) if storage else _global_monitor 