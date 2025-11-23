# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

## [0.3.0] - 2025-11-23

### Added
- `SQLiteConnector` for lightweight, file-based database logging without external dependencies
- Support for `ThreadPoolExecutor` in `FunctionMonitor` to handle async saves more efficiently

### Changed
- Refactored `FunctionMonitor` to use a thread pool instead of spawning new threads for each call, improving performance under high load
- Improved test suite stability with `wait_for_all_saves()` synchronization

## [0.2.0] - 2025-09-27

### Added
- Database integration via `DatabaseStorage` with connectors:
  - `PostgreSQLConnector` (psycopg2)
  - `MongoDBConnector` (pymongo)
  - `CouchbaseConnector` (couchbase SDK)
  - Non-blocking background writes using a thread pool; automatic connection lifecycle
- Time-interval querying with `TimeInterval` (including optional time zone) across local and database backends

## [0.1.0] - 2025-07-14

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

## [0.0.1] - 2025-07-04

### Added
- Initial project structure
- Basic function monitoring concept
- Proof of concept implementation 