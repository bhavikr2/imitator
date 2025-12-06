"""
Machine Learning integration for Imitator
"""

import logging
import threading
import time
import pickle
import atexit
from typing import Optional, Dict, List, Any, Set, Union
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.linear_model import SGDRegressor

from .types import FunctionCall, FunctionSignature, TimeInterval
from .storage import LocalStorage
from .monitor import monitor_function, FunctionMonitor


class ModelTrainer:
    """Manages training for a single function"""

    def __init__(self, signature: FunctionSignature, min_batch_size: int = 32):
        self.signature = signature
        self.min_batch_size = min_batch_size
        # Use SGDRegressor for online learning
        self.model = SGDRegressor(max_iter=1000, tol=1e-3)
        self.X_buffer: List[List[float]] = []
        self.y_buffer: List[float] = []
        # Store parameter names to ensure consistent order
        self.param_names = list(signature.parameters.keys())
        self.training_count = 0
        self.is_fitted = False

    def _extract_features(self, inputs: Dict[str, Any]) -> Optional[List[float]]:
        """Extract numerical features from inputs dict, handling lists/arrays and skipping non-numerics"""
        features = []
        
        def flatten(item):
            if isinstance(item, (int, float)):
                features.append(float(item))
            elif isinstance(item, (list, tuple, np.ndarray)):
                if isinstance(item, np.ndarray):
                    item = item.flatten()
                for subitem in item:
                    flatten(subitem)
            # Ignore other types (str, dict, objects) effectively filtering them out
            
        for name in self.param_names:
            if name not in inputs:
                continue
                
            val = inputs[name]
            flatten(val)
        
        if not features:
            return None
            
        return features

    def add_data(self, calls: List[FunctionCall]):
        """Add new data points and trigger training if buffer is full"""
        for call in calls:
            # Check for errors in output
            if isinstance(call.io_record.output, dict) and "error" in call.io_record.output:
                continue
                
            # Check output is scalar
            output = call.io_record.output
            if not isinstance(output, (int, float)):
                continue
                
            features = self._extract_features(call.io_record.inputs)
            if features:
                self.X_buffer.append(features)
                self.y_buffer.append(float(output))
                
        if len(self.X_buffer) >= self.min_batch_size:
            self._train_batch()
            
    def _train_batch(self):
        """Update model weights with buffered data"""
        try:
            X = np.array(self.X_buffer)
            y = np.array(self.y_buffer)
            
            # Check dimensions
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                
            self.model.partial_fit(X, y)
            self.is_fitted = True
            self.training_count += len(y)
            
            logging.info(
                f"Updated model for {self.signature.name}. "
                f"Batch: {len(y)}, Total samples: {self.training_count}"
            )
            
            # Clear buffers
            self.X_buffer = []
            self.y_buffer = []
        except Exception as e:
            logging.error(f"Training failed for {self.signature.name}: {e}")

    def predict(self, inputs: Dict[str, Any]) -> Optional[float]:
        """Predict output for given inputs"""
        if not self.is_fitted:
            return None
            
        features = self._extract_features(inputs)
        if not features:
            return None
            
        X = np.array([features])
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        try:
            return float(self.model.predict(X)[0])
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return None


class TrainingDaemon:
    """Background service to monitor storage and train models"""

    def __init__(self, storage=None, check_interval: float = 5.0):
        self.storage = storage or LocalStorage()
        self.check_interval = check_interval
        self.trainers: Dict[str, ModelTrainer] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Track last processed timestamp per function
        self._cursors: Dict[str, float] = {} 

    def register(self, signature: FunctionSignature, min_batch_size: int = 32):
        """Register a function for training"""
        if signature.name not in self.trainers:
            self.trainers[signature.name] = ModelTrainer(signature, min_batch_size)
            # Start from now (ignore old data? or maybe start from 0 to learn from history?)
            # Let's start from 0 to pick up existing logs.
            self._cursors[signature.name] = 0.0

    def start(self):
        """Start the daemon thread"""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logging.info("Training daemon started")
            
    def stop(self):
        """Stop the daemon thread"""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            logging.info("Training daemon stopped")
            
    def _run(self):
        """Main loop"""
        while not self._stop_event.is_set():
            try:
                for func_name, trainer in self.trainers.items():
                    last_ts = self._cursors.get(func_name, 0.0)
                    
                    # Create interval from last_ts to now
                    start_dt = datetime.fromtimestamp(last_ts)
                    interval = TimeInterval(start_time=start_dt)
                    
                    # Load calls
                    calls = self.storage.load_calls(func_name, interval)
                    
                    # Filter strictly newer calls
                    new_calls = []
                    max_ts = last_ts
                    
                    for call in calls:
                        ts = call.io_record.timestamp.timestamp()
                        # Use a small epsilon to avoid reloading the exact same timestamp
                        if ts > last_ts + 0.000001:
                            new_calls.append(call)
                            if ts > max_ts:
                                max_ts = ts
                                
                    if new_calls:
                        logging.debug(f"Found {len(new_calls)} new calls for {func_name}")
                        trainer.add_data(new_calls)
                        self._cursors[func_name] = max_ts
                        
            except Exception as e:
                logging.error(f"Error in training daemon: {e}")
                
            time.sleep(self.check_interval)

    def get_trainer(self, function_name: str) -> Optional[ModelTrainer]:
        return self.trainers.get(function_name)


# Global daemon instance
_daemon: Optional[TrainingDaemon] = None
_daemon_lock = threading.Lock()

def _cleanup_daemon():
    """Stop the global daemon on exit"""
    global _daemon
    if _daemon:
        _daemon.stop()

def get_training_daemon(storage=None) -> TrainingDaemon:
    """Get or create the global training daemon"""
    global _daemon
    with _daemon_lock:
        if _daemon is None:
            _daemon = TrainingDaemon(storage)
            _daemon.start()
            atexit.register(_cleanup_daemon)
        return _daemon


def train(min_batch_size: int = 32, monitor: Optional[FunctionMonitor] = None):
    """
    Decorator to enable training for a function.
    Automatically enables monitoring if not already present.
    
    Args:
        min_batch_size: Number of samples to accumulate before updating model
        monitor: Optional specific monitor instance to use
    """
    def decorator(func):
        # 1. Enable monitoring
        # We wrap with monitor_function. If monitor is provided, use it.
        monitored_func = monitor_function(func, monitor=monitor)
        
        # 2. Register with training daemon
        # We need to ensure the daemon uses the SAME storage as the monitor.
        
        # Resolve storage
        if monitor:
            storage = monitor.storage
        else:
            # If no monitor provided, monitor_function creates one with default LocalStorage.
            # We should try to use default LocalStorage for daemon too.
            # However, monitor_function creates a NEW monitor instance each time if passed as decorator factory?
            # Wait, monitor_function logic:
            # if monitor is None: effective_monitor = FunctionMonitor(...)
            
            # This means each decorated function gets its own monitor instance if we don't pass one.
            # But they all default to `LocalStorage()` which writes to `logs/` dir.
            # So as long as daemon uses `LocalStorage()`, it reads from `logs/`.
            # This is fine for file-based storage. 
            # For DB storage, user MUST pass the monitor or storage explicitly to ensure sharing.
            storage = None 

        daemon = get_training_daemon(storage)
        
        # Extract signature from original function
        # Note: func might already be a wrapper if composed with other decorators.
        # But monitor_function uses @wraps, so introspection might work.
        sig = FunctionSignature.from_function(func)
        
        daemon.register(sig, min_batch_size)
        
        return monitored_func
        
    return decorator

