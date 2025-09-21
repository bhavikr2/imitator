"""
Local storage backend for function I/O logs
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from collections import defaultdict
import threading

from .types import FunctionCall


class LocalStorage:
    """Local file-based storage for function call logs"""
    
    def __init__(self, log_dir="logs", format="jsonl", buffer_size=100, flush_interval=None):
        """
        Initialize local storage
        
        Args:
            log_dir: Directory to store log files
            format: File format ('jsonl' or 'json')
        """
        self.log_dir = Path(log_dir)
        self.format = format
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._buffer = defaultdict(list)
        self._lock = threading.Lock()
        self.log_dir.mkdir(exist_ok=True)
        # Optionally start a background flush thread if flush_interval is set
        # Register atexit handler to flush on shutdown

    def _get_log_file(self, function_name: str) -> Path:
        """Get log file path for a function"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"{function_name}_{timestamp}.{self.format}"
    
    def save_call(self, function_call: FunctionCall):
        """Save a function call to storage"""
        with self._lock:
            fn = function_call.function_signature.name
            self._buffer[fn].append(function_call)
            if len(self._buffer[fn]) >= self.buffer_size:
                self._flush_function(fn)

    def _flush_function(self, function_name):
        """Write all buffered calls for a function to disk, then clear buffer"""
        log_file = self._get_log_file(function_name)
        calls_to_write = self._buffer[function_name]
        self._buffer[function_name] = [] # Clear buffer after flushing

        if self.format == "jsonl":
            # Append to JSONL file
            with open(log_file, "a", encoding="utf-8") as f:
                for call in calls_to_write:
                    f.write(call.model_dump_json() + "\n")
                f.flush()  # Ensure data is written to OS buffer
                os.fsync(f.fileno())  # Force OS to write to disk
        else:
            # Read existing JSON array, append, and write back
            calls = []
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        calls = json.load(f)
                except json.JSONDecodeError:
                    calls = []
            
            calls.extend(call.model_dump() for call in calls_to_write)
            
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(calls, f, indent=2, default=str)
                f.flush()  # Ensure data is written to OS buffer
                os.fsync(f.fileno())  # Force OS to write to disk

    def flush(self):
        """Flush all buffers"""
        with self._lock:
            for function_name in list(self._buffer.keys()): # Iterate over a copy
                self._flush_function(function_name)

    def close(self):
        """Flush and clean up (stop background thread if any)"""
        self.flush()
        # No background thread to stop in this simple implementation

    def load_calls(self, function_name: str, date: Optional[str] = None) -> List[FunctionCall]:
        """
        Load function calls from storage
        
        Args:
            function_name: Name of the function
            date: Date in YYYYMMDD format, if None uses today
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        log_file = self.log_dir / f"{function_name}_{date}.{self.format}"
        
        if not log_file.exists():
            return []
        
        calls = []
        
        if self.format == "jsonl":
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        call_data = json.loads(line)
                        calls.append(FunctionCall.model_validate(call_data))
        else:
            with open(log_file, "r", encoding="utf-8") as f:
                call_data_list = json.load(f)
                for call_data in call_data_list:
                    calls.append(FunctionCall.model_validate(call_data))
        
        return calls
    
    def get_all_functions(self) -> List[str]:
        """Get list of all monitored functions"""
        functions = set()
        for file_path in self.log_dir.glob("*.json*"):
            # Extract function name from filename (before the date part)
            # Format is: {function_name}_{YYYYMMDD}.{format}
            stem = file_path.stem
            # Find the last underscore followed by 8 digits (date)
            import re
            match = re.match(r'(.+)_\d{8}$', stem)
            if match:
                function_name = match.group(1)
                functions.add(function_name)
        return list(functions) 