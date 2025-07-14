"""
Toy example demonstrating the Imitator framework
"""

import sys
import os
import time
from typing import List, Dict, Tuple

# Add the parent directory to the path so we can import imitator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imitator import monitor_function, LocalStorage


# Example 1: Simple mathematical function
@monitor_function
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


# Example 2: Function with complex types
@monitor_function
def process_data(data: List[float], multiplier: float = 1.0) -> Dict[str, float]:
    """Process a list of numbers and return statistics"""
    if not data:
        return {"mean": 0.0, "sum": 0.0, "count": 0}
    
    total = sum(x * multiplier for x in data)
    mean = total / len(data)
    
    return {
        "mean": mean,
        "sum": total,
        "count": len(data)
    }


# Example 3: Function that can raise exceptions
@monitor_function
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# Example 4: Function with multiple return types
@monitor_function
def analyze_text(text: str) -> Tuple[int, int, List[str]]:
    """Analyze text and return word count, char count, and words"""
    words = text.split()
    return len(words), len(text), words


def demonstrate_framework():
    """Demonstrate the framework with various function calls"""
    print("🚀 Imitator Framework Demo")
    print("=" * 40)
    
    # Example 1: Simple function calls
    print("\n1. Simple mathematical operations:")
    results = []
    for i in range(5):
        result = add_numbers(i, i + 1)
        results.append(result)
        print(f"   add_numbers({i}, {i+1}) = {result}")
    
    # Example 2: Complex data processing
    print("\n2. Data processing:")
    test_data = [
        ([1.0, 2.0, 3.0, 4.0], 1.0),
        ([10.0, 20.0, 30.0], 2.0),
        ([], 1.0),  # Edge case: empty list
        ([5.5, 7.3, 2.1], 0.5)
    ]
    
    for data, multiplier in test_data:
        result = process_data(data, multiplier)
        print(f"   process_data({data}, {multiplier}) = {result}")
    
    # Example 3: Error handling
    print("\n3. Error handling:")
    division_cases = [
        (10.0, 2.0),
        (15.0, 3.0),
        (8.0, 0.0),  # This will raise an exception
        (20.0, 4.0)
    ]
    
    for a, b in division_cases:
        try:
            result = divide_numbers(a, b)
            print(f"   divide_numbers({a}, {b}) = {result}")
        except ValueError as e:
            print(f"   divide_numbers({a}, {b}) raised: {e}")
    
    # Example 4: Text analysis
    print("\n4. Text analysis:")
    texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Python is amazing!",
        ""  # Edge case: empty string
    ]
    
    for text in texts:
        result = analyze_text(text)
        print(f"   analyze_text('{text}') = {result}")
    
    print("\n" + "=" * 40)
    print("✅ All function calls completed!")
    
    # Now let's examine what was logged
    examine_logs()


def examine_logs():
    """Examine the logged function calls"""
    print("\n📊 Examining Logged Data")
    print("=" * 40)
    
    storage = LocalStorage()
    
    # Get all monitored functions
    functions = storage.get_all_functions()
    print(f"\nMonitored functions: {functions}")
    
    # Analyze each function's logs
    for func_name in functions:
        print(f"\n🔍 Function: {func_name}")
        calls = storage.load_calls(func_name)
        
        print(f"   Total calls: {len(calls)}")
        
        if calls:
            # Show function signature
            sig = calls[0].function_signature
            print(f"   Signature: {sig.name}({', '.join(f'{k}: {v}' for k, v in sig.parameters.items())})")
            if sig.return_type:
                print(f"   Return type: {sig.return_type}")
            
            # Show execution time statistics
            execution_times = [call.io_record.execution_time_ms for call in calls if call.io_record.execution_time_ms]
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                print(f"   Execution time: avg={avg_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms")
            
            # Show some example calls
            print(f"   Example calls:")
            for i, call in enumerate(calls[:3]):  # Show first 3 calls
                inputs = call.io_record.inputs
                output = call.io_record.output
                print(f"     [{i+1}] Input: {inputs} → Output: {output}")
                
                # Check for errors
                if isinstance(output, dict) and "error" in output:
                    print(f"         ⚠️  Error: {output['error']}")
            
            if len(calls) > 3:
                print(f"     ... and {len(calls) - 3} more calls")
    
    print(f"\n💾 Log files stored in: {storage.log_dir}")


if __name__ == "__main__":
    demonstrate_framework() 