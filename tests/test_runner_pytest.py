"""
Pytest-based test runner for Imitator framework
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imitator import LocalStorage


def run_all_tests():
    """Run all pytest tests with comprehensive reporting"""
    
    # Clean up any existing logs
    clean_logs()
    
    # Configure pytest arguments
    pytest_args = [
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "--color=yes",          # Colored output
        "--strict-markers",     # Strict marker checking
        "-x",                   # Stop on first failure
        "--disable-warnings",   # Disable warnings for cleaner output
        "tests/",              # Test directory
    ]
    
    print("🚀 Running Imitator Framework Test Suite with pytest")
    print("=" * 60)
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    # Analyze results after tests
    analyze_test_logs()
    
    return exit_code


def run_specific_test_categories():
    """Run specific categories of tests"""
    
    categories = {
        "basic": "tests/test_basic_functionality.py",
        "class_methods": "tests/test_class_methods_pytest.py", 
        "advanced": "tests/test_advanced_features_pytest.py",
        "performance": "tests/test_performance_pytest.py",
    }
    
    print("📋 Available test categories:")
    for category, path in categories.items():
        print(f"   {category}: {path}")
    
    print("\n🎯 Running all categories...")
    
    results = {}
    
    for category, test_path in categories.items():
        print(f"\n{'='*40}")
        print(f"🧪 Running {category} tests")
        print(f"{'='*40}")
        
        # Clean logs before each category
        clean_logs()
        
        exit_code = pytest.main([
            "-v",
            "--tb=short", 
            "--color=yes",
            test_path
        ])
        
        results[category] = exit_code == 0
    
    # Summary
    print(f"\n📊 Test Category Results:")
    print("=" * 40)
    
    for category, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {category}: {status}")
    
    all_passed = all(results.values())
    print(f"\n🏁 Overall Result: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    return all_passed


def run_performance_benchmarks():
    """Run only performance tests with detailed reporting"""
    
    print("🏃‍♂️ Running Performance Benchmarks")
    print("=" * 50)
    
    clean_logs()
    
    exit_code = pytest.main([
        "-v",
        "--tb=short",
        "--color=yes",
        "-k", "performance",  # Run tests matching 'performance'
        "tests/test_performance_pytest.py"
    ])
    
    return exit_code == 0


def run_async_tests():
    """Run only async tests"""
    
    print("🚀 Running Async Tests")
    print("=" * 30)
    
    clean_logs()
    
    exit_code = pytest.main([
        "-v",
        "--tb=short",
        "--color=yes",
        "-m", "asyncio",  # Run tests marked with asyncio
        "tests/"
    ])
    
    return exit_code == 0


def clean_logs():
    """Clean up log files from previous runs"""
    log_dirs = ["logs", "examples/logs", "tests/logs"]
    
    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if log_path.exists():
            for log_file in log_path.glob("*.json*"):
                try:
                    log_file.unlink()
                except Exception:
                    pass


def analyze_test_logs():
    """Analyze logs generated during testing"""
    print("\n📊 Test Log Analysis")
    print("=" * 40)
    
    try:
        storage = LocalStorage()
        functions = storage.get_all_functions()
        
        if not functions:
            print("   No functions logged during tests")
            return
        
        print(f"   Functions monitored during tests: {len(functions)}")
        
        total_calls = 0
        error_calls = 0
        
        for func_name in functions:
            calls = storage.load_calls(func_name)
            total_calls += len(calls)
            
            # Count errors
            for call in calls:
                if isinstance(call.io_record.output, dict) and "error" in call.io_record.output:
                    error_calls += 1
        
        print(f"   Total function calls logged: {total_calls:,}")
        print(f"   Error calls logged: {error_calls}")
        print(f"   Success rate: {((total_calls - error_calls) / total_calls * 100):.1f}%")
        
    except Exception as e:
        print(f"   Could not analyze logs: {e}")


def main():
    """Main entry point for test runner"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "categories":
            success = run_specific_test_categories()
        elif command == "performance":
            success = run_performance_benchmarks()
        elif command == "async":
            success = run_async_tests()
        elif command == "all":
            success = run_all_tests() == 0
        else:
            print(f"Unknown command: {command}")
            print("Available commands: all, categories, performance, async")
            return 1
    else:
        success = run_all_tests() == 0
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 