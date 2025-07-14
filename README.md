# Imitator

[![PyPI version](https://badge.fury.io/py/imitator.svg)](https://badge.fury.io/py/imitator)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python framework for monitoring and imitating function behavior with automatic I/O tracking and pattern learning. Perfect for collecting training data for machine learning models, debugging, performance analysis, and understanding function behavior in production systems with future capabilities for behavior imitation.

## ✨ Features

- 🎯 **Simple Decorator**: Just add `@monitor_function` to any function
- 📊 **Type Validation**: Uses Pydantic models for robust type handling
- 💾 **Flexible Storage**: Local JSON/JSONL files with configurable backends
- ⚡ **Performance Monitoring**: Tracks execution times and performance metrics
- 🚨 **Error Handling**: Captures and logs exceptions with full context
- 🔄 **Async Support**: Full support for asynchronous functions
- 📈 **Sampling & Rate Limiting**: Control overhead with smart sampling
- 🏗️ **Class Method Support**: Monitor class methods with proper handling
- 🔍 **Modification Detection**: Detect in-place parameter modifications
- 🪶 **Minimal Dependencies**: Only requires Pydantic (≥2.0.0)

## 🚀 Installation

```bash
pip install imitator
```

**Requirements**: Python 3.8+, Pydantic ≥2.0.0

## ⚡ Quick Start

```python
from imitator import monitor_function

@monitor_function
def add_numbers(a: int, b: int) -> int:
    return a + b

# Use the function normally
result = add_numbers(5, 3)  # Automatically logged!
```

That's it! Your function calls are now being monitored and logged automatically.

## 📖 Usage Examples

### Basic Function Monitoring

```python
from imitator import monitor_function
from typing import List, Dict

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

### Advanced Configuration

```python
from imitator import monitor_function, LocalStorage, FunctionMonitor

# Custom storage location
custom_storage = LocalStorage(log_dir="my_logs", format="json")

@monitor_function(storage=custom_storage)
def my_function(x: int) -> int:
    return x * 2

# Rate limiting and sampling for high-frequency functions
monitor = FunctionMonitor(
    sampling_rate=0.1,  # Log 10% of calls
    max_calls_per_minute=100  # Max 100 calls per minute
)

@monitor.monitor
def high_frequency_function(x: int) -> int:
    return x ** 2
```

### Async Function Support

```python
import asyncio
from imitator import monitor_function

@monitor_function
async def fetch_data(url: str) -> dict:
    """Simulate async data fetching"""
    await asyncio.sleep(0.1)
    return {"data": f"Response from {url}"}

# Async functions work seamlessly
async def main():
    result = await fetch_data("https://api.example.com")
    print(result)

asyncio.run(main())
```

### Class Method Monitoring

```python
from imitator import monitor_function

class DataProcessor:
    def __init__(self, name: str):
        self.name = name
    
    @monitor_function
    def process_batch(self, items: List[dict]) -> dict:
        """Process a batch of items"""
        processed = []
        for item in items:
            processed.append(self.process_item(item))
        return {"processed": len(processed), "results": processed}
    
    def process_item(self, item: dict) -> dict:
        # Helper method (not monitored)
        return {"id": item.get("id"), "processed_by": self.name}

processor = DataProcessor("BatchProcessor")
result = processor.process_batch([{"id": 1}, {"id": 2}])
```

### Examining Logged Data

```python
from imitator import LocalStorage

storage = LocalStorage()

# Get all monitored functions
functions = storage.get_all_functions()
print(f"Monitored functions: {functions}")

# Load calls for a specific function
calls = storage.load_calls("add_numbers")

for call in calls:
    print(f"Input: {call.io_record.inputs}")
    print(f"Output: {call.io_record.output}")
    print(f"Execution time: {call.io_record.execution_time_ms}ms")
    print(f"Timestamp: {call.io_record.timestamp}")
```

## 📊 Data Structure

The framework captures comprehensive information about each function call:

```python
class FunctionCall(BaseModel):
    function_signature: FunctionSignature  # Function name, parameters, return type
    io_record: IORecord                    # Inputs, output, timestamp, execution time
    call_id: str                          # Unique identifier

class IORecord(BaseModel):
    inputs: Dict[str, Any]                # Function input parameters
    output: Any                           # Function return value
    timestamp: str                        # ISO format timestamp
    execution_time_ms: float              # Execution time in milliseconds
    input_modifications: Optional[Dict]    # Detected in-place modifications
```

## 💾 Storage Format

Logs are stored as JSON/JSONL files in the `logs/` directory:

```
logs/
├── add_numbers_20241201.jsonl
├── process_data_20241201.jsonl
└── ...
```

Each log entry contains:
- **Function signature** with type annotations
- **Input parameters** with actual values
- **Output value** or exception details
- **Execution time** in milliseconds
- **Timestamp** in ISO format
- **Error information** (if applicable)
- **Input modifications** (if detected)

## 🚨 Error Handling

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
    pass  # Exception is logged with input parameters and full traceback
```

Exception logs include:
- **Input parameters** that caused the error
- **Exception type** and message
- **Full traceback** for debugging
- **Execution time** until the exception occurred

## 🏗️ Framework Components

### Core Components

- **`monitor_function`**: Main decorator for function monitoring
- **`FunctionMonitor`**: Advanced monitoring with configuration options
- **`FunctionCall`**: Pydantic model for complete function call records
- **`IORecord`**: Pydantic model for input/output pairs
- **`LocalStorage`**: Local file-based storage backend

### Type System

The framework uses Pydantic for:
- **Runtime type validation** and serialization
- **Automatic schema generation** from function signatures
- **Type-safe data structures** with validation
- **JSON serialization** with complex type support

## 📋 Example Output

Running monitored functions generates structured logs like:

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
    "timestamp": "2024-01-15T10:30:45.123456",
    "execution_time_ms": 0.05,
    "input_modifications": null
  },
  "call_id": "1705312245.123456"
}
```

## 🎯 Use Cases

### 🤖 Machine Learning
- **Training Data Collection**: Gather input-output pairs for model training
- **Model Inference Monitoring**: Track model performance and behavior
- **Feature Engineering**: Monitor data preprocessing pipelines
- **A/B Testing**: Compare different model versions

### 🔧 Development & Debugging
- **Function Profiling**: Analyze performance bottlenecks
- **Debugging**: Track function calls and parameter values
- **Integration Testing**: Monitor system component interactions
- **Behavior Analysis**: Understand function usage patterns

### 📊 Production Monitoring
- **System Health**: Monitor critical business functions
- **Performance Tracking**: Track execution times and error rates
- **User Behavior**: Analyze how functions are used in production
- **Compliance**: Maintain audit trails for regulatory requirements

### 🔬 Research & Analysis
- **Algorithm Analysis**: Study algorithm behavior with real data
- **Performance Optimization**: Identify optimization opportunities
- **Data Quality**: Monitor data processing pipelines
- **Experimentation**: Support research and development workflows

## 📚 Examples

The package includes comprehensive examples demonstrating various use cases:

### Available Examples
- **`basic_usage.py`**: Getting started with core features
- **`advanced_monitoring.py`**: Advanced configuration and async support
- **`real_world_simulation.py`**: Practical applications and systems

### Run Examples
```bash
# Install the package
pip install imitator

# Clone repository for examples (if needed)
git clone https://github.com/yourusername/imitator.git
cd imitator/examples

# Run examples
python basic_usage.py
python advanced_monitoring.py
python real_world_simulation.py
```

Each example demonstrates:
- Different monitoring strategies
- Error handling scenarios
- Performance analysis
- Log inspection and analysis

## 🚀 Getting Started

1. **Install**: `pip install imitator`
2. **Import**: `from imitator import monitor_function`
3. **Decorate**: Add `@monitor_function` to your functions
4. **Run**: Use your functions normally
5. **Analyze**: Check the generated logs in the `logs/` directory

## 📖 Documentation

- **Examples**: Comprehensive examples in the `examples/` directory
- **API Reference**: Detailed docstrings in all modules
- **Type Hints**: Full type annotation support
- **Error Handling**: Graceful handling of edge cases

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details. 