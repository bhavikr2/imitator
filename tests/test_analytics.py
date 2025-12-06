import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pytest
from typing import List, Dict, Any

from imitator.analytics import FunctionAnalyzer, HAS_ANALYTICS
from imitator.types import FunctionCall, FunctionSignature, IORecord
from imitator.storage import LocalStorage

# Helper to create function calls
def create_mock_call(
    function_name: str,
    inputs: Dict[str, Any],
    output: Any,
    execution_time: float,
    timestamp: datetime = None
) -> FunctionCall:
    if timestamp is None:
        timestamp = datetime.now()
        
    signature = FunctionSignature(
        name=function_name,
        parameters={k: type(v).__name__ for k, v in inputs.items()},
        return_type=type(output).__name__
    )
    
    io_record = IORecord(
        inputs=inputs,
        output=output,
        execution_time_ms=execution_time,
        timestamp=timestamp
    )
    
    return FunctionCall(
        function_signature=signature,
        io_record=io_record
    )

@pytest.mark.skipif(not HAS_ANALYTICS, reason="Analytics dependencies not installed")
class TestFunctionAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.mock_storage = MagicMock(spec=LocalStorage)
        self.analyzer = FunctionAnalyzer(self.mock_storage)
        self.function_name = "test_func"

    def test_numerical_inputs_outputs(self):
        """Test analytics with pure numerical inputs and outputs"""
        # Create calls with numerical data
        calls = []
        for i in range(100):
            inputs = {"x": float(i), "y": float(i * 2)}
            output = float(i * 3)
            exec_time = 10.0 + (i % 10) # 10.0 to 19.0
            calls.append(create_mock_call(self.function_name, inputs, output, exec_time))
            
        self.mock_storage.load_calls.return_value = calls
        
        # 1. Test execution time stats
        stats = self.analyzer.get_execution_time_stats(self.function_name)
        self.assertEqual(stats["count"], 100)
        self.assertAlmostEqual(stats["min"], 10.0)
        self.assertAlmostEqual(stats["max"], 19.0)
        self.assertAlmostEqual(stats["mean"], 14.5)
        
        # 2. Test input distribution (numeric)
        input_analysis = self.analyzer.analyze_input_distribution(self.function_name, "x")
        self.assertEqual(input_analysis["type"], "numeric")
        self.assertEqual(input_analysis["stats"]["count"], 100)
        self.assertEqual(input_analysis["stats"]["min"], 0.0)
        self.assertEqual(input_analysis["stats"]["max"], 99.0)
        
        # 3. Test output distribution (numeric)
        output_analysis = self.analyzer.analyze_output_distribution(self.function_name)
        self.assertEqual(output_analysis["type"], "numeric")
        self.assertEqual(output_analysis["stats"]["count"], 100)
        self.assertEqual(output_analysis["stats"]["max"], 297.0) # 99 * 3

    def test_mixed_inputs(self):
        """Test analytics with a mix of numerical and categorical inputs"""
        calls = []
        categories = ["A", "B", "C"]
        for i in range(60):
            # Mix of numeric 'val' and categorical 'cat'
            inputs = {
                "val": i,
                "cat": categories[i % 3] # A, B, C repeating
            }
            calls.append(create_mock_call(self.function_name, inputs, "result", 5.0))
            
        self.mock_storage.load_calls.return_value = calls
        
        # Test numeric input
        val_analysis = self.analyzer.analyze_input_distribution(self.function_name, "val")
        self.assertEqual(val_analysis["type"], "numeric")
        self.assertEqual(val_analysis["stats"]["count"], 60)
        
        # Test categorical input
        cat_analysis = self.analyzer.analyze_input_distribution(self.function_name, "cat")
        self.assertEqual(cat_analysis["type"], "categorical")
        self.assertEqual(cat_analysis["total_count"], 60)
        self.assertEqual(cat_analysis["unique_count"], 3)
        self.assertEqual(cat_analysis["frequencies"]["A"], 20)
        self.assertEqual(cat_analysis["frequencies"]["B"], 20)
        self.assertEqual(cat_analysis["frequencies"]["C"], 20)
        self.assertAlmostEqual(cat_analysis["distribution"]["A"], 0.33333333)

    def test_struct_class_inputs(self):
        """Test analytics with complex inputs (dicts/objects)"""
        calls = []
        # Simulate objects serialized as dictionaries
        user_states = [
            {"id": 1, "status": "active"},
            {"id": 2, "status": "inactive"},
            {"id": 1, "status": "active"} # Duplicate state
        ]
        
        for state in user_states:
            inputs = {"user": state}
            # Output is modified state
            output = state.copy()
            output["processed"] = True
            calls.append(create_mock_call(self.function_name, inputs, output, 5.0))
            
        self.mock_storage.load_calls.return_value = calls
        
        # Test input distribution for complex object
        # Should be treated as categorical (converted to string)
        input_analysis = self.analyzer.analyze_input_distribution(self.function_name, "user")
        self.assertEqual(input_analysis["type"], "categorical")
        self.assertEqual(input_analysis["total_count"], 3)
        # Unique counts should be 2 because two inputs are identical dicts
        self.assertEqual(input_analysis["unique_count"], 2)
        
        # Verify the string representation is being counted
        state_str = str({"id": 1, "status": "active"})
        self.assertEqual(input_analysis["frequencies"][state_str], 2)

    def test_error_function(self):
        """Test analytics when function errors out"""
        calls = []
        
        # 8 success calls
        for i in range(8):
            calls.append(create_mock_call(self.function_name, {"x": i}, "success", 5.0))
            
        # 2 error calls - output is an error dict
        error_output_1 = {"error": "ValueError", "message": "Invalid input"}
        error_output_2 = {"error": "TypeError", "message": "Wrong type"}
        
        calls.append(create_mock_call(self.function_name, {"x": -1}, error_output_1, 5.0))
        calls.append(create_mock_call(self.function_name, {"x": -2}, error_output_2, 5.0))
        
        self.mock_storage.load_calls.return_value = calls
        
        # Test output distribution
        # Mix of strings ("success") and dicts (error objects)
        # Should be treated as categorical
        output_analysis = self.analyzer.analyze_output_distribution(self.function_name)
        
        self.assertEqual(output_analysis["type"], "categorical")
        self.assertEqual(output_analysis["total_count"], 10)
        self.assertEqual(output_analysis["unique_count"], 3) # "success", error_1, error_2
        
        self.assertEqual(output_analysis["frequencies"]["success"], 8)
        self.assertEqual(output_analysis["frequencies"][str(error_output_1)], 1)

    @patch("matplotlib.pyplot.show")
    def test_plotting_calls(self, mock_show):
        """Verify plotting methods run without error"""
        calls = [create_mock_call(self.function_name, {"x": 1}, 1, 5.0)]
        self.mock_storage.load_calls.return_value = calls
        
        try:
            self.analyzer.plot_execution_time_distribution(self.function_name)
            self.analyzer.plot_input_distribution(self.function_name, "x")
            self.analyzer.plot_output_distribution(self.function_name)
        except Exception as e:
            self.fail(f"Plotting raised exception: {e}")
            
        # Ensure show was called (or savefig if we passed save_path)
        self.assertTrue(mock_show.called)


