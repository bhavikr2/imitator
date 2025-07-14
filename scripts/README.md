# Scripts

This directory contains utility scripts for building and maintaining the Imitator package.

## Available Scripts

### `build_package.py`
Comprehensive script for building and testing the package.

**Usage:**
```bash
# Run all steps (clean, lint, test, examples, build, check)
python scripts/build_package.py

# Run specific steps
python scripts/build_package.py --clean --test --build
python scripts/build_package.py --lint
python scripts/build_package.py --examples
```

**Available options:**
- `--clean`: Clean build artifacts
- `--test`: Run tests with coverage
- `--lint`: Run code linting (flake8, black, mypy)
- `--examples`: Run example scripts
- `--build`: Build the package
- `--check`: Check the built package
- `--all`: Run all steps (default)

## Development Workflow

1. **Make changes** to the code
2. **Run tests** to ensure everything works:
   ```bash
   python scripts/build_package.py --test
   ```
3. **Run linting** to check code quality:
   ```bash
   python scripts/build_package.py --lint
   ```
4. **Test examples** to verify functionality:
   ```bash
   python scripts/build_package.py --examples
   ```
5. **Build package** when ready:
   ```bash
   python scripts/build_package.py --build
   ```

## Publishing to PyPI

1. **Build and test** the package:
   ```bash
   python scripts/build_package.py
   ```

2. **Upload to PyPI** (requires credentials):
   ```bash
   python -m twine upload dist/*
   ```

3. **Upload to Test PyPI** first (recommended):
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

## Requirements

Make sure you have the development dependencies installed:
```bash
pip install -r requirements-dev.txt
```

## Notes

- The build script automatically cleans build artifacts before building
- All examples are run to ensure the package works correctly
- The package is checked for common issues before upload
- Type checking and linting ensure code quality 