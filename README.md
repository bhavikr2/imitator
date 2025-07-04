# LogAndLearn Framework

A lightweight Python framework for monitoring function input-output pairs with automatic type validation and local storage. Perfect for collecting training data for machine learning models or analyzing function behavior.

## Features

- 🎯 **Simple Decorator**: Just add `@monitor_function` to any function
- 📊 **Type Validation**: Uses Pydantic models for robust type handling
- 💾 **Local Storage**: Stores I/O logs as JSON/JSONL files
- ⚡ **Execution Timing**: Tracks function execution times
- 🚨 **Error Handling**: Captures and logs exceptions
- 🪶 **Minimal Dependencies**: Only requires Pydantic

## Installation

```bash
pip install pydantic>=2.0.0
```

## Quick Start

```python
from logandlearn import monitor_function

@monitor_function
def add_numbers(a: int, b: int) -> int:
    return a + b

# Use the function normally
result = add_numbers(5, 3)  # Automatically logged!
```

## Usage Examples

### Basic Function Monitoring

```python
from logandlearn import monitor_function

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

# Function calls are automatically logged
result = process_data([1.0, 2.0, 3.0], 2.0)
```

### Custom Storage Configuration

```python
from logandlearn import monitor_function, LocalStorage

# Custom storage location
custom_storage = LocalStorage(log_dir="my_logs", format="json")

@monitor_function(storage=custom_storage)
def my_function(x: int) -> int:
    return x * 2
```

### Examining Logged Data

```python
from logandlearn import LocalStorage

storage = LocalStorage()

# Get all monitored functions
functions = storage.get_all_functions()

# Load calls for a specific function
calls = storage.load_calls("add_numbers")

for call in calls:
    print(f"Input: {call.io_record.inputs}")
    print(f"Output: {call.io_record.output}")
    print(f"Execution time: {call.io_record.execution_time_ms}ms")
```

## Data Structure

The framework captures comprehensive information about each function call:

```python
class FunctionCall(BaseModel):
    function_signature: FunctionSignature  # Function name, parameters, return type
    io_record: IORecord                    # Inputs, output, timestamp, execution time
    call_id: str                          # Unique identifier
```

## Storage Format

Logs are stored as JSON/JSONL files in the `logs/` directory:

```
logs/
├── add_numbers_20241201.jsonl
├── process_data_20241201.jsonl
└── ...
```

Each log entry contains:
- Function signature with type annotations
- Input parameters with values
- Output value
- Execution time in milliseconds
- Timestamp
- Error information (if applicable)

## Error Handling

The framework gracefully handles and logs exceptions:

```python
@monitor_function
def divide_numbers(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

try:
    result = divide_numbers(10, 0)
except ValueError:
    pass  # Exception is logged with input parameters
```

## Framework Components

### Core Components

- **`monitor_function`**: Main decorator for function monitoring
- **`FunctionCall`**: Pydantic model for complete function call records
- **`IORecord`**: Pydantic model for input/output pairs
- **`LocalStorage`**: Local file-based storage backend

### Type System

The framework uses Pydantic for:
- Runtime type validation
- Automatic serialization/deserialization
- Schema generation
- Type-safe data structures

## Example Output

Running the toy example generates logs like:

```json
{
  "function_signature": {
    "name": "add_numbers",
    "parameters": {
      "a": "<class 'int'>",
      "b": "<class 'int'>"
    },
    "return_type": "<class 'int'>"
  },
  "io_record": {
    "inputs": {
      "a": 5,
      "b": 3
    },
    "output": 8,
    "timestamp": "2024-12-01T10:30:45.123456",
    "execution_time_ms": 0.05
  },
  "call_id": "1732123845.123456"
}
```

## Use Cases

- **ML Training Data**: Collect input-output pairs for training neural networks
- **Function Profiling**: Analyze function performance and behavior
- **Debugging**: Track function calls and their parameters
- **Testing**: Verify function behavior across different inputs
- **Monitoring**: Track function usage in production systems

## Running the Example

```bash
cd examples
python toy_example.py
```

This will demonstrate:
- Basic function monitoring
- Complex data type handling
- Error logging
- Data analysis of logged calls

## License

MIT License - feel free to use this framework in your projects! 