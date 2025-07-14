"""
Local storage backend for function I/O logs
"""

import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .types import FunctionCall


class LocalStorage:
    """Local file-based storage for function call logs"""
    
    def __init__(self, log_dir: str = "logs", format: str = "jsonl"):
        """
        Initialize local storage
        
        Args:
            log_dir: Directory to store log files
            format: File format ('jsonl' or 'json')
        """
        self.log_dir = Path(log_dir)
        self.format = format
        self.log_dir.mkdir(exist_ok=True)
        
    def _get_log_file(self, function_name: str) -> Path:
        """Get log file path for a function"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"{function_name}_{timestamp}.{self.format}"
    
    def save_call(self, function_call: FunctionCall) -> None:
        """Save a function call to storage"""
        log_file = self._get_log_file(function_call.function_signature.name)
        
        if self.format == "jsonl":
            # Append to JSONL file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(function_call.model_dump_json() + "\n")
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
            
            calls.append(function_call.model_dump())
            
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(calls, f, indent=2, default=str)
                f.flush()  # Ensure data is written to OS buffer
                os.fsync(f.fileno())  # Force OS to write to disk
    
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