"""
Analytics module for analyzing and visualizing function behavior and performance.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import logging
from collections import Counter

from .storage import LocalStorage, DatabaseStorage
from .types import TimeInterval, FunctionCall

# Optional dependencies
try:
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False


class AnalyticsError(Exception):
    """Base class for analytics errors"""
    pass


class MissingDependencyError(AnalyticsError):
    """Raised when required analytics libraries are missing"""
    pass


class FunctionAnalyzer:
    """
    Analyzer for function calls, providing statistical analysis and visualization.
    """

    def __init__(self, storage: Union[LocalStorage, DatabaseStorage]):
        """
        Initialize the analyzer with a storage backend.

        Args:
            storage: The storage instance to fetch calls from.
        """
        self.storage = storage
        if not HAS_ANALYTICS:
            logging.warning(
                "Analytics dependencies (numpy, scipy, matplotlib) are not installed. "
                "Analysis and plotting features will not work."
            )

    def _check_dependencies(self):
        """Ensure analytics dependencies are available"""
        if not HAS_ANALYTICS:
            raise MissingDependencyError(
                "Analytics dependencies are missing. "
                "Please install them with: pip install 'imitator[analytics]'"
            )

    def _get_calls(
        self, function_name: str, time_interval: Optional[TimeInterval] = None
    ) -> List[FunctionCall]:
        """Helper to fetch calls from storage"""
        return self.storage.load_calls(function_name, time_interval)

    def get_execution_time_stats(
        self, function_name: str, time_interval: Optional[TimeInterval] = None
    ) -> Dict[str, float]:
        """
        Calculate statistics for execution times.

        Args:
            function_name: Name of the function to analyze
            time_interval: Optional time range

        Returns:
            Dictionary with statistics (mean, median, std, min, max, p95, p99)
        """
        if not HAS_ANALYTICS:
            self._check_dependencies()

        calls = self._get_calls(function_name, time_interval)
        if not calls:
            return {}

        execution_times = [
            call.io_record.execution_time_ms 
            for call in calls 
            if call.io_record.execution_time_ms is not None
        ]

        if not execution_times:
            return {}

        return {
            "count": len(execution_times),
            "mean": float(np.mean(execution_times)),
            "median": float(np.median(execution_times)),
            "std": float(np.std(execution_times)),
            "min": float(np.min(execution_times)),
            "max": float(np.max(execution_times)),
            "p95": float(np.percentile(execution_times, 95)),
            "p99": float(np.percentile(execution_times, 99)),
        }

    def plot_execution_time_distribution(
        self,
        function_name: str,
        time_interval: Optional[TimeInterval] = None,
        bins: int = 30,
        show_kde: bool = True,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot the distribution of execution times.

        Args:
            function_name: Name of the function
            time_interval: Optional time range
            bins: Number of histogram bins
            show_kde: Whether to show Kernel Density Estimation overlay
            save_path: If provided, save plot to this path instead of showing
            figsize: Figure size (width, height)
        """
        self._check_dependencies()

        calls = self._get_calls(function_name, time_interval)
        execution_times = [
            call.io_record.execution_time_ms 
            for call in calls 
            if call.io_record.execution_time_ms is not None
        ]

        if not execution_times:
            logging.warning(f"No execution time data found for {function_name}")
            return

        plt.figure(figsize=figsize)
        
        # Plot histogram
        plt.hist(
            execution_times, 
            bins=bins, 
            density=True, 
            alpha=0.6, 
            color='skyblue', 
            edgecolor='black',
            label='Histogram'
        )

        if show_kde and len(execution_times) > 1:
            try:
                kde = stats.gaussian_kde(execution_times)
                x_range = np.linspace(min(execution_times), max(execution_times), 200)
                plt.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
            except Exception as e:
                logging.warning(f"Could not compute KDE: {e}")

        plt.title(f"Execution Time Distribution: {function_name}")
        plt.xlabel("Execution Time (ms)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def analyze_input_distribution(
        self,
        function_name: str,
        param_name: str,
        time_interval: Optional[TimeInterval] = None
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of a specific input parameter.

        Args:
            function_name: Name of the function
            param_name: Name of the parameter to analyze
            time_interval: Optional time range

        Returns:
            Dictionary with analysis results (type, stats for numeric, counts for categorical)
        """
        self._check_dependencies()

        calls = self._get_calls(function_name, time_interval)
        values = []

        for call in calls:
            inputs = call.io_record.inputs
            if param_name in inputs:
                val = inputs[param_name]
                # Handle potentially non-serializable objects that were converted to strings
                values.append(val)

        if not values:
            return {"error": f"Parameter '{param_name}' not found in calls"}

        # Determine if numeric or categorical
        try:
            numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
            is_numeric = len(numeric_values) == len(values) and len(values) > 0
        except (ValueError, TypeError):
            is_numeric = False

        if is_numeric:
            return {
                "type": "numeric",
                "stats": {
                    "count": len(numeric_values),
                    "mean": float(np.mean(numeric_values)),
                    "median": float(np.median(numeric_values)),
                    "std": float(np.std(numeric_values)),
                    "min": float(np.min(numeric_values)),
                    "max": float(np.max(numeric_values)),
                }
            }
        else:
            # Categorical analysis
            # Convert dicts/lists to string representation for counting
            str_values = [str(v) if isinstance(v, (dict, list)) else v for v in values]
            counts = Counter(str_values)
            total = len(values)
            return {
                "type": "categorical",
                "total_count": total,
                "unique_count": len(counts),
                "frequencies": {k: v for k, v in counts.most_common(10)}, # Top 10
                "distribution": {k: v/total for k, v in counts.most_common(10)}
            }

    def plot_input_distribution(
        self,
        function_name: str,
        param_name: str,
        time_interval: Optional[TimeInterval] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot the distribution of a specific input parameter.
        
        Automatically detects if the parameter is numeric (histogram/KDE) or 
        categorical (bar chart).

        Args:
            function_name: Name of the function
            param_name: Name of the parameter
            time_interval: Optional time range
            save_path: Output file path
            figsize: Figure size
        """
        self._check_dependencies()
        
        analysis = self.analyze_input_distribution(function_name, param_name, time_interval)
        
        if "error" in analysis:
            logging.warning(analysis["error"])
            return

        calls = self._get_calls(function_name, time_interval)
        
        plt.figure(figsize=figsize)

        if analysis["type"] == "numeric":
            values = [
                float(call.io_record.inputs[param_name]) 
                for call in calls 
                if param_name in call.io_record.inputs 
                and isinstance(call.io_record.inputs[param_name], (int, float))
            ]
            
            plt.hist(
                values, 
                bins=30, 
                density=True, 
                alpha=0.6, 
                color='lightgreen', 
                edgecolor='black',
                label='Histogram'
            )
            
            if len(values) > 1:
                try:
                    kde = stats.gaussian_kde(values)
                    x_range = np.linspace(min(values), max(values), 200)
                    plt.plot(x_range, kde(x_range), 'g-', lw=2, label='KDE')
                except Exception:
                    pass
            
            plt.xlabel(f"Parameter: {param_name}")
            plt.ylabel("Density")
            
        else:
            # Categorical
            freqs = analysis["frequencies"]
            labels = list(freqs.keys())
            counts = list(freqs.values())
            
            # Truncate long labels
            labels = [str(l)[:20] + "..." if len(str(l)) > 20 else str(l) for l in labels]
            
            plt.bar(labels, counts, color='lightcoral')
            plt.xlabel(f"Parameter: {param_name}")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        plt.title(f"Input Distribution: {function_name} ({param_name})")
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def analyze_output_distribution(
        self,
        function_name: str,
        time_interval: Optional[TimeInterval] = None
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of the function output.

        Args:
            function_name: Name of the function
            time_interval: Optional time range

        Returns:
            Dictionary with analysis results
        """
        self._check_dependencies()

        calls = self._get_calls(function_name, time_interval)
        values = []

        for call in calls:
            val = call.io_record.output
            if val is not None:
                # Handle potentially non-serializable objects
                values.append(val)

        if not values:
            return {"error": f"No outputs found for {function_name}"}

        # Determine if numeric or categorical
        try:
            numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
            is_numeric = len(numeric_values) == len(values) and len(values) > 0
        except (ValueError, TypeError):
            is_numeric = False

        if is_numeric:
            return {
                "type": "numeric",
                "stats": {
                    "count": len(numeric_values),
                    "mean": float(np.mean(numeric_values)),
                    "median": float(np.median(numeric_values)),
                    "std": float(np.std(numeric_values)),
                    "min": float(np.min(numeric_values)),
                    "max": float(np.max(numeric_values)),
                }
            }
        else:
            # Categorical analysis
            str_values = [str(v) if isinstance(v, (dict, list)) else v for v in values]
            counts = Counter(str_values)
            total = len(values)
            return {
                "type": "categorical",
                "total_count": total,
                "unique_count": len(counts),
                "frequencies": {k: v for k, v in counts.most_common(10)},
                "distribution": {k: v/total for k, v in counts.most_common(10)}
            }

    def plot_output_distribution(
        self,
        function_name: str,
        time_interval: Optional[TimeInterval] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot the distribution of the function output.

        Args:
            function_name: Name of the function
            time_interval: Optional time range
            save_path: Output file path
            figsize: Figure size
        """
        self._check_dependencies()
        
        analysis = self.analyze_output_distribution(function_name, time_interval)
        
        if "error" in analysis:
            logging.warning(analysis["error"])
            return

        calls = self._get_calls(function_name, time_interval)
        
        plt.figure(figsize=figsize)

        if analysis["type"] == "numeric":
            values = [
                float(call.io_record.output) 
                for call in calls 
                if call.io_record.output is not None 
                and isinstance(call.io_record.output, (int, float))
            ]
            
            plt.hist(
                values, 
                bins=30, 
                density=True, 
                alpha=0.6, 
                color='plum', 
                edgecolor='black',
                label='Histogram'
            )
            
            if len(values) > 1:
                try:
                    kde = stats.gaussian_kde(values)
                    x_range = np.linspace(min(values), max(values), 200)
                    plt.plot(x_range, kde(x_range), 'm-', lw=2, label='KDE')
                except Exception:
                    pass
            
            plt.xlabel("Output Value")
            plt.ylabel("Density")
            
        else:
            # Categorical
            freqs = analysis["frequencies"]
            labels = list(freqs.keys())
            counts = list(freqs.values())
            
            # Truncate long labels
            labels = [str(l)[:20] + "..." if len(str(l)) > 20 else str(l) for l in labels]
            
            plt.bar(labels, counts, color='mediumpurple')
            plt.xlabel("Output Value")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        plt.title(f"Output Distribution: {function_name}")
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

