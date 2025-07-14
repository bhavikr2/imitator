# Contributing to Imitator

We welcome contributions to Imitator! This document outlines the process for contributing to the project.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/Imitator/Imitator.git
   cd imitator
   ```
3. **Set up development environment**:
   ```bash
   make install-dev
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** and commit them
6. **Test your changes**:
   ```bash
   make test
   ```
7. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Environment Setup

1. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

2. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   make pre-commit
   ```

### Development Workflow

1. **Make changes** to the code
2. **Run tests** to ensure everything works:
   ```bash
   make test
   ```
3. **Run linting** to check code quality:
   ```bash
   make lint
   ```
4. **Format code** (if needed):
   ```bash
   make format
   ```
5. **Test examples** to verify functionality:
   ```bash
   make examples
   ```

## 📋 Code Standards

### Code Quality
- **PEP 8**: Follow Python style guidelines
- **Type hints**: Use type annotations for all functions
- **Docstrings**: Include comprehensive docstrings
- **Black**: Code formatting with Black
- **Flake8**: Linting with Flake8
- **MyPy**: Type checking with MyPy

### Testing
- **pytest**: Use pytest for testing
- **Coverage**: Maintain high test coverage (>90%)
- **Async tests**: Use pytest-asyncio for async function tests
- **Mock**: Use pytest-mock for mocking external dependencies

### Documentation
- **Clear docstrings**: Follow Google/NumPy docstring format
- **Examples**: Include usage examples in docstrings
- **README**: Update README.md if adding new features
- **CHANGELOG**: Update CHANGELOG.md for notable changes

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_monitor.py -v

# Run tests with coverage
python -m pytest tests/ -v --cov=imitator --cov-report=html
```

### Test Structure

- `tests/`: Main test directory
- `tests/test_*.py`: Test files following pytest conventions
- `tests/conftest.py`: Shared test fixtures
- Follow the pattern: one test file per module

### Writing Tests

```python
import pytest
from imitator import monitor_function

def test_monitor_function_basic():
    """Test basic monitoring functionality."""
    
    @monitor_function
    def sample_function(x: int) -> int:
        return x * 2
    
    result = sample_function(5)
    assert result == 10
    
    # Test that logs were created
    # ... additional assertions
```

## 📚 Adding Examples

Examples help users understand how to use the framework:

1. **Create example file** in `examples/` directory
2. **Include comprehensive docstring** explaining the example
3. **Add real-world scenarios** without external dependencies
4. **Test the example** with `make examples`
5. **Update examples README** in `examples/README.md`

### Example Structure

```python
#!/usr/bin/env python3
"""
Example: [Brief Description]

This example demonstrates [detailed description of what it shows].
"""

import sys
import os
from typing import List, Dict

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imitator import monitor_function

# Your example code here...

if __name__ == "__main__":
    # Run the example
    pass
```

## 🔧 Code Architecture

### Module Structure
- `imitator/`: Main package directory
- `imitator/monitor.py`: Core monitoring functionality
- `imitator/storage.py`: Storage backends
- `imitator/types.py`: Type definitions
- `imitator/__init__.py`: Package initialization

### Key Components
- **FunctionMonitor**: Core monitoring class
- **LocalStorage**: File-based storage backend
- **IORecord**: Input/output record model
- **FunctionCall**: Complete function call record

## 🐛 Reporting Issues

### Bug Reports
1. **Check existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Include reproduction steps**
4. **Provide system information**:
   - Python version
   - Imitator version
   - Operating system
   - Relevant error messages

### Feature Requests
1. **Describe the use case** clearly
2. **Explain the expected behavior**
3. **Consider implementation complexity**
4. **Provide examples** if possible

## 📝 Pull Request Process

### Before Submitting
1. **Run all tests**: `make test`
2. **Check code quality**: `make lint`
3. **Test examples**: `make examples`
4. **Update documentation** if needed
5. **Add/update tests** for new features

### Pull Request Guidelines
1. **Clear title** describing the change
2. **Detailed description** of what was changed and why
3. **Link to related issues** if applicable
4. **Include test coverage** for new features
5. **Update CHANGELOG.md** for notable changes

### Review Process
1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on different Python versions
4. **Documentation** review if applicable

## 🎯 Types of Contributions

### Code Contributions
- **Bug fixes**: Fix existing issues
- **New features**: Add new functionality
- **Performance improvements**: Optimize existing code
- **Refactoring**: Improve code structure

### Documentation
- **API documentation**: Improve docstrings
- **Examples**: Add new usage examples
- **README**: Improve project documentation
- **Tutorials**: Create learning resources

### Testing
- **Test coverage**: Add missing tests
- **Edge cases**: Test unusual scenarios
- **Performance tests**: Benchmark improvements
- **Integration tests**: Test system interactions

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: contact@imitator.dev for direct contact

## 🏆 Recognition

Contributors are recognized in:
- **CHANGELOG.md**: For notable contributions
- **README.md**: For significant contributions
- **GitHub**: Through commit history and PR credits

## 📜 License

By contributing to Imitator, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Thank you for considering contributing to Imitator! Your contributions help make this project better for everyone. 