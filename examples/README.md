# Imitator Examples

This directory contains comprehensive examples demonstrating the Imitator framework's capabilities in various real-world scenarios.

## 📁 Available Examples

### 1. `basic_usage.py` - Getting Started
**Purpose**: Demonstrates core framework features with simple, easy-to-understand functions.

**Features Covered**:
- Basic function monitoring with `@monitor_function`
- Mathematical operations and string processing
- Data processing with complex types
- Error handling and exception logging
- Log analysis and inspection

**Run**: `python basic_usage.py`

### 2. `advanced_monitoring.py` - Advanced Features
**Purpose**: Shows sophisticated monitoring capabilities and configurations.

**Features Covered**:
- Custom storage configuration
- Sampling rates and rate limiting
- Class method monitoring
- Asynchronous function monitoring
- In-place modification detection
- Performance analysis

**Run**: `python advanced_monitoring.py`

### 3. `real_world_simulation.py` - Practical Applications
**Purpose**: Simulates real-world systems where function monitoring provides value.

**Features Covered**:
- E-commerce order processing system
- User authentication and session management
- Data analytics pipeline
- Machine learning model inference
- Comprehensive error handling and edge cases

**Run**: `python real_world_simulation.py`

### 4. `toy_example.py` - Legacy Example
**Purpose**: Original toy example for backward compatibility.

**Run**: `python toy_example.py`

## 🚀 Quick Start

1. **Install Imitator**:
   ```bash
   pip install imitator
   ```

2. **Run Basic Example**:
   ```bash
   cd examples
   python basic_usage.py
   ```

3. **Explore Generated Logs**:
   After running examples, check the generated log directories:
   - `logs/` - Default logs from basic_usage.py
   - `advanced_logs/` - Logs from advanced_monitoring.py
   - `simulation_logs/` - Logs from real_world_simulation.py

## 📊 Understanding the Output

Each example will:
1. **Execute monitored functions** with various inputs
2. **Display results** in the console
3. **Generate log files** with I/O records
4. **Analyze logs** and show statistics

### Log File Structure
```json
{
  "function_signature": {
    "name": "function_name",
    "parameters": {"param1": "type", "param2": "type"},
    "return_type": "type"
  },
  "io_record": {
    "inputs": {"param1": "value", "param2": "value"},
    "output": "result",
    "execution_time_ms": 0.123,
    "timestamp": "2024-01-15T10:30:45.123456"
  },
  "call_id": "unique_identifier"
}
```

## 🎯 Use Cases Demonstrated

### 1. **Development and Debugging**
- Track function calls and their parameters
- Identify performance bottlenecks
- Debug complex data transformations

### 2. **Machine Learning**
- Collect training data from function I/O
- Monitor model inference performance
- Track feature preprocessing steps

### 3. **System Monitoring**
- Monitor critical business functions
- Track error rates and patterns
- Analyze system performance over time

### 4. **Testing and Validation**
- Verify function behavior with various inputs
- Generate test datasets
- Validate system integration points

## 🔧 Customization Options

### Storage Configuration
```python
from imitator import LocalStorage, monitor_function

# Custom storage location and format
custom_storage = LocalStorage(log_dir="my_logs", format="json")

@monitor_function(storage=custom_storage)
def my_function(x):
    return x * 2
```

### Sampling and Rate Limiting
```python
from imitator import FunctionMonitor

# Monitor with sampling and rate limiting
monitor = FunctionMonitor(
    sampling_rate=0.1,  # Log 10% of calls
    max_calls_per_minute=100  # Max 100 calls per minute
)

@monitor.monitor
def high_frequency_function(x):
    return x ** 2
```

### Async Function Monitoring
```python
@monitor_function
async def async_function(data):
    # Async functions are automatically detected
    await asyncio.sleep(0.1)
    return process_data(data)
```

## 📈 Performance Considerations

The framework is designed for minimal overhead:
- **Sampling**: Use sampling rates for high-frequency functions
- **Rate Limiting**: Prevent log overflow with rate limits
- **Async Storage**: Logs are saved asynchronously
- **Selective Monitoring**: Only monitor functions that provide value

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Make sure Imitator is installed
   pip install imitator
   ```

2. **Permission Errors**:
   ```python
   # Use custom log directory with write permissions
   storage = LocalStorage(log_dir="./my_logs")
   ```

3. **High Memory Usage**:
   ```python
   # Use sampling for high-frequency functions
   @monitor_function(sampling_rate=0.01)  # 1% sampling
   def frequent_function(x):
       return x
   ```

## 📝 Example Modifications

### Adding Your Own Functions

1. **Simple Function**:
   ```python
   @monitor_function
   def your_function(param1, param2):
       # Your logic here
       return result
   ```

2. **With Custom Storage**:
   ```python
   custom_storage = LocalStorage(log_dir="your_logs")
   
   @monitor_function(storage=custom_storage)
   def your_function(data):
       return process(data)
   ```

3. **Class Methods**:
   ```python
   class YourClass:
       @monitor_function
       def your_method(self, param):
           return self.process(param)
   ```

## 📚 Next Steps

1. **Explore the Examples**: Run each example to understand different use cases
2. **Modify and Experiment**: Change parameters and add your own functions
3. **Analyze Logs**: Use the log analysis utilities to understand patterns
4. **Integrate**: Add monitoring to your own projects

## 🤝 Contributing

Found an issue or want to add a new example? 
- Check the main project repository for contribution guidelines
- Examples should be self-contained and demonstrate clear use cases
- Include both success and error scenarios in your examples

## 📖 Further Reading

- Main Documentation: See the project README.md
- API Reference: Check the docstrings in the imitator package
- Type Hints: All examples include comprehensive type annotations 