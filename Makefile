# Makefile for Imitator package

.PHONY: help install install-dev test lint format examples build clean check upload upload-test all

# Default help target
help:
	@echo "Imitator Package Development"
	@echo "=============================="
	@echo
	@echo "Available targets:"
	@echo "  help         - Show this help message"
	@echo "  install      - Install the package"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests with coverage"
	@echo "  lint         - Run linting (flake8, mypy)"
	@echo "  format       - Format code with black"
	@echo "  examples     - Run example scripts"
	@echo "  build        - Build the package"
	@echo "  clean        - Clean build artifacts"
	@echo "  check        - Check the built package"
	@echo "  upload-test  - Upload to Test PyPI"
	@echo "  upload       - Upload to PyPI"
	@echo "  all          - Run all checks and build"

# Install package
install:
	pip install -e .

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Run tests
test:
	python -m pytest tests/ -v --cov=imitator --cov-report=html --cov-report=term

# Run linting
lint:
	python -m flake8 imitator/ --max-line-length=88 --extend-ignore=E203
	python -m mypy imitator/

# Format code
format:
	python -m black imitator/ tests/ examples/
	python -m isort imitator/ tests/ examples/

# Run examples
examples:
	cd examples && python basic_usage.py
	cd examples && python advanced_monitoring.py
	cd examples && python real_world_simulation.py

# Build package
build:
	python -m build

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Check package
check:
	python -m twine check dist/*

# Upload to Test PyPI
upload-test: build check
	python -m twine upload --repository testpypi dist/*

# Upload to PyPI
upload: build check
	python -m twine upload dist/*

# Run all checks and build
all: clean lint test examples build check
	@echo "✅ All checks passed! Package is ready for upload."

# Quick development cycle
dev: format lint test
	@echo "✅ Development cycle complete!"

# Install pre-commit hooks
pre-commit:
	pre-commit install
	pre-commit run --all-files 