#!/usr/bin/env python3
"""
Package build and test script for Imitator

This script helps with building, testing, and preparing the package for PyPI.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import shlex


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def clean_build_artifacts():
    """Clean build artifacts."""
    print("\n🧹 Cleaning build artifacts...")
    
    # Directories to clean
    dirs_to_clean = [
        "build",
        "dist",
        "*.egg-info",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".coverage",
        "htmlcov"
    ]
    
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            elif path.is_file():
                path.unlink()
                print(f"Removed file: {path}")
    
    # Clean Python cache files
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
                print(f"Removed __pycache__ in: {root}")


def run_tests():
    """Run the test suite."""
    py = shlex.quote(sys.executable)
    return run_command(
        f"{py} -m pytest tests/ -v --cov=imitator --cov-report=html --cov-report=term",
        "Running tests with coverage"
    )


def run_linting():
    """Run code linting."""
    py = shlex.quote(sys.executable)
    commands = [
        (f"{py} -m flake8 imitator/ --max-line-length=88 --extend-ignore=E203", "Flake8 linting"),
        (f"{py} -m black --check imitator/", "Black code formatting check"),
        (f"{py} -m mypy imitator/", "MyPy type checking"),
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed


def build_package():
    """Build the package."""
    py = shlex.quote(sys.executable)
    return run_command(f"{py} -m build", "Building package")


def check_package():
    """Check the built package."""
    py = shlex.quote(sys.executable)
    return run_command(f"{py} -m twine check dist/*", "Checking package")


def run_examples():
    """Run example scripts to verify functionality."""
    example_dir = Path("examples")
    if not example_dir.exists():
        print("⚠️  Examples directory not found")
        return True
    
    examples = [
        "basic_usage.py",
        "advanced_monitoring.py",
        "real_world_simulation.py"
    ]
    
    all_passed = True
    py = shlex.quote(sys.executable)
    for example in examples:
        example_path = example_dir / example
        if example_path.exists():
            if not run_command(f"{py} {example_path}", f"Running example: {example}"):
                all_passed = False
        else:
            print(f"⚠️  Example not found: {example}")
    
    return all_passed


def main():
    """Main function."""
    print("🚀 Imitator Package Build Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Build and test Imitator package")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--examples", action="store_true", help="Run examples")
    parser.add_argument("--build", action="store_true", help="Build package")
    parser.add_argument("--check", action="store_true", help="Check built package")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    # If no specific arguments, run all
    if not any([args.clean, args.test, args.lint, args.examples, args.build, args.check]):
        args.all = True
    
    success = True
    
    if args.clean or args.all:
        clean_build_artifacts()
    
    if args.lint or args.all:
        if not run_linting():
            success = False
    
    if args.test or args.all:
        if not run_tests():
            success = False
    
    if args.examples or args.all:
        if not run_examples():
            success = False
    
    if args.build or args.all:
        if not build_package():
            success = False
    
    if args.check or args.all:
        if not check_package():
            success = False
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 All steps completed successfully!")
        print("\nNext steps:")
        print("1. Review the built package in dist/")
        print("2. Test installation: pip install dist/imitator-*.whl")
        print("3. Upload to PyPI: python -m twine upload dist/*")
    else:
        print("❌ Some steps failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 