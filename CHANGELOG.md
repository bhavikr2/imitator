# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: Rebranded from "LogAndLearn" to "Imitator"
- Package name changed from `logandlearn` to `imitator`
- All import statements now use `from imitator import ...`
- Updated description to focus on monitoring and imitation capabilities
- Updated project URLs and metadata

### Added
- Initial PyPI package structure
- Comprehensive documentation and examples
- Type hints support with py.typed marker

## [0.1.0] - 2024-01-15

### Added
- Core `monitor_function` decorator for automatic I/O logging
- `FunctionMonitor` class for advanced monitoring configuration
- `LocalStorage` backend for file-based log storage
- Support for both synchronous and asynchronous functions
- Automatic type validation using Pydantic models
- Execution time tracking
- Exception logging and error handling
- Input modification detection for mutable parameters
- Sampling rate control for performance optimization
- Rate limiting for high-frequency functions
- Comprehensive test suite with pytest
- Class method monitoring with proper handling of `self` and `cls` parameters

### Features
- **Type Safety**: Full type annotations and Pydantic validation
- **Performance**: Minimal overhead with optional sampling
- **Flexibility**: Configurable storage backends and monitoring options
- **Robustness**: Comprehensive error handling and edge case support
- **Ease of Use**: Simple decorator-based API

### Storage Format
- JSON/JSONL file format for easy parsing and analysis
- Structured data with function signatures, I/O records, and metadata
- Automatic file rotation by date for organization

### Examples
- Basic function monitoring
- Complex data type handling
- Error case logging
- Performance analysis utilities
- Real-world usage patterns

## [0.0.1] - 2024-01-01

### Added
- Initial project structure
- Basic function monitoring concept
- Proof of concept implementation 