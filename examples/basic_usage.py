#!/usr/bin/env python3
"""
Basic Usage Example - Imitator Framework

This example demonstrates the core functionality of the Imitator framework
with simple, easy-to-understand functions.
"""

import sys
import os
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imitator import monitor_function, LocalStorage


# Example 1: Simple mathematical operations
@monitor_function
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@monitor_function
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@monitor_function
def divide_safely(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers, returning None if division by zero."""
    if b == 0:
        return None
    return a / b


# Example 2: String processing functions
@monitor_function
def format_name(first_name: str, last_name: str, title: str = "") -> str:
    """Format a person's name with optional title."""
    if title:
        return f"{title} {first_name} {last_name}"
    return f"{first_name} {last_name}"


@monitor_function
def count_words(text: str) -> Dict[str, int]:
    """Count words in a text string."""
    words = text.lower().split()
    word_count = {}
    for word in words:
        # Remove basic punctuation
        clean_word = word.strip('.,!?;:')
        if clean_word:
            word_count[clean_word] = word_count.get(clean_word, 0) + 1
    return word_count


@monitor_function
def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text (simple implementation)."""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


# Example 3: Data processing functions
@monitor_function
def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers."""
    if not numbers:
        return {"count": 0, "sum": 0, "mean": 0, "min": 0, "max": 0}
    
    total = sum(numbers)
    count = len(numbers)
    mean = total / count
    
    return {
        "count": count,
        "sum": total,
        "mean": mean,
        "min": min(numbers),
        "max": max(numbers)
    }


@monitor_function
def filter_even_numbers(numbers: List[int]) -> List[int]:
    """Filter out even numbers from a list."""
    return [num for num in numbers if num % 2 == 0]


@monitor_function
def group_by_length(words: List[str]) -> Dict[int, List[str]]:
    """Group words by their length."""
    groups = {}
    for word in words:
        length = len(word)
        if length not in groups:
            groups[length] = []
        groups[length].append(word)
    return groups


# Example 4: Functions with complex return types
@monitor_function
def parse_user_data(data_string: str) -> Tuple[str, int, List[str]]:
    """Parse user data from a formatted string."""
    parts = data_string.split('|')
    if len(parts) != 3:
        raise ValueError("Invalid data format")
    
    name = parts[0].strip()
    age = int(parts[1].strip())
    interests = [interest.strip() for interest in parts[2].split(',')]
    
    return name, age, interests


@monitor_function
def validate_password(password: str) -> Dict[str, Union[bool, str]]:
    """Validate a password and return detailed feedback."""
    result = {
        "valid": True,
        "message": "Password is valid",
        "length_ok": len(password) >= 8,
        "has_upper": any(c.isupper() for c in password),
        "has_lower": any(c.islower() for c in password),
        "has_digit": any(c.isdigit() for c in password),
        "has_special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    }
    
    issues = []
    if not result["length_ok"]:
        issues.append("at least 8 characters")
    if not result["has_upper"]:
        issues.append("uppercase letter")
    if not result["has_lower"]:
        issues.append("lowercase letter")
    if not result["has_digit"]:
        issues.append("digit")
    if not result["has_special"]:
        issues.append("special character")
    
    if issues:
        result["valid"] = False
        result["message"] = f"Password needs: {', '.join(issues)}"
    
    return result


def demonstrate_basic_usage():
    """Demonstrate basic usage of the Imitator framework."""
    print("🚀 Imitator Framework - Basic Usage Demo")
    print("=" * 50)
    
    # Mathematical operations
    print("\n1. Mathematical Operations:")
    print(f"   add_numbers(5, 3) = {add_numbers(5, 3)}")
    print(f"   multiply_numbers(2.5, 4.0) = {multiply_numbers(2.5, 4.0)}")
    print(f"   divide_safely(10, 2) = {divide_safely(10, 2)}")
    print(f"   divide_safely(10, 0) = {divide_safely(10, 0)}")
    
    # String processing
    print("\n2. String Processing:")
    print(f"   format_name('John', 'Doe') = '{format_name('John', 'Doe')}'")
    print(f"   format_name('Jane', 'Smith', 'Dr.') = '{format_name('Jane', 'Smith', 'Dr.')}'")
    
    sample_text = "Hello world! This is a test. Hello again world."
    word_counts = count_words(sample_text)
    print(f"   count_words('{sample_text}') = {word_counts}")
    
    email_text = "Contact us at support@example.com or sales@company.org for more info."
    emails = extract_emails(email_text)
    print(f"   extract_emails(...) = {emails}")
    
    # Data processing
    print("\n3. Data Processing:")
    test_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats = calculate_statistics(test_numbers)
    print(f"   calculate_statistics({test_numbers}) = {stats}")
    
    even_nums = filter_even_numbers(test_numbers)
    print(f"   filter_even_numbers({test_numbers}) = {even_nums}")
    
    test_words = ["cat", "dog", "elephant", "bird", "butterfly"]
    grouped = group_by_length(test_words)
    print(f"   group_by_length({test_words}) = {grouped}")
    
    # Complex data types
    print("\n4. Complex Data Types:")
    user_data = "John Doe|30|programming,reading,hiking"
    try:
        parsed = parse_user_data(user_data)
        print(f"   parse_user_data('{user_data}') = {parsed}")
    except ValueError as e:
        print(f"   parse_user_data(...) raised: {e}")
    
    # Password validation
    test_passwords = ["weak", "StrongPass123!", "NoNumbers!", "nocaps123!"]
    for pwd in test_passwords:
        result = validate_password(pwd)
        print(f"   validate_password('{pwd}') = {result['valid']} - {result['message']}")
    
    print("\n" + "=" * 50)
    print("✅ All function calls completed and logged!")
    
    # Analyze the logs
    analyze_logs()


def analyze_logs():
    """Analyze the logged function calls."""
    print("\n📊 Log Analysis")
    print("=" * 50)
    
    storage = LocalStorage()
    
    # Get all monitored functions
    functions = storage.get_all_functions()
    print(f"\nTotal monitored functions: {len(functions)}")
    
    total_calls = 0
    for func_name in functions:
        calls = storage.load_calls(func_name)
        total_calls += len(calls)
        
        if calls:
            # Calculate execution time stats
            times = [call.io_record.execution_time_ms for call in calls 
                    if call.io_record.execution_time_ms is not None]
            avg_time = sum(times) / len(times) if times else 0
            
            print(f"  📋 {func_name}:")
            print(f"     Calls: {len(calls)}")
            print(f"     Avg execution time: {avg_time:.3f}ms")
            
            # Show sample I/O
            if calls:
                sample_call = calls[0]
                inputs = sample_call.io_record.inputs
                output = sample_call.io_record.output
                print(f"     Sample I/O: {inputs} → {output}")
    
    print(f"\n📈 Total function calls logged: {total_calls}")
    print(f"📁 Logs stored in: {storage.log_dir}")


if __name__ == "__main__":
    demonstrate_basic_usage() 