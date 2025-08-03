"""
Local storage backend for function I/O logs
"""

import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import tempfile

from .types import FunctionCall


class LocalStorage:
    """Local file-based storage for function call logs"""
    
    def __init__(self, log_dir: str = "logs", format: str = "jsonl", 
                 max_file_size_mb: float = 50.0, rotation_batch_size: int = 50):
        """
        Initialize local storage
        
        Args:
            log_dir: Directory to store log files
            format: File format ('jsonl' or 'json')
            max_file_size_mb: Maximum file size in MB before rotation (default: 50MB)
            rotation_batch_size: Number of entries to remove when rotating (default: 50)
        """
        self.log_dir = Path(log_dir)
        self.format = format
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)  # Convert MB to bytes
        self.rotation_batch_size = rotation_batch_size
        self.log_dir.mkdir(exist_ok=True)
        
        # Counter to track new entries since last rotation check
        self._entry_counter = {}
        
    def _get_log_file(self, function_name: str) -> Path:
        """Get log file path for a function"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"{function_name}_{timestamp}.{self.format}"
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes"""
        return file_path.stat().st_size if file_path.exists() else 0
    
    def _count_jsonl_entries(self, file_path: Path) -> int:
        """Count the number of entries in a JSONL file"""
        if not file_path.exists():
            return 0
        
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Only count non-empty lines
                    count += 1
        return count
    
    def _rotate_jsonl_file(self, file_path: Path, entries_to_remove: int) -> None:
        """
        Rotate a JSONL file by removing the oldest entries
        
        Args:
            file_path: Path to the JSONL file
            entries_to_remove: Number of oldest entries to remove
        """
        if not file_path.exists():
            return
        
        # Read all lines
        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]  # Skip empty lines
        
        # If we have fewer lines than entries to remove, clear the file
        if len(lines) <= entries_to_remove:
            file_path.write_text("", encoding="utf-8")
            return
        
        # Keep only the lines after removing the oldest entries
        remaining_lines = lines[entries_to_remove:]
        
        # Write back to file atomically using a temporary file
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                       dir=file_path.parent, delete=False) as temp_file:
            for line in remaining_lines:
                temp_file.write(line)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = temp_file.name
        
        # Atomically replace the original file
        os.replace(temp_path, file_path)
    
    def _should_rotate(self, file_path: Path, function_name: str) -> bool:
        """
        Check if file rotation should be performed
        
        Args:
            file_path: Path to the log file
            function_name: Name of the function (for tracking entry count)
            
        Returns:
            True if rotation should be performed
        """
        # Check if file exceeds size limit
        if self._get_file_size(file_path) < self.max_file_size_bytes:
            return False
        
        # Initialize counter if not exists
        if function_name not in self._entry_counter:
            self._entry_counter[function_name] = 0
        
        # Increment counter for this new entry
        self._entry_counter[function_name] += 1
        
        # Check if we've reached the rotation batch size
        if self._entry_counter[function_name] >= self.rotation_batch_size:
            self._entry_counter[function_name] = 0  # Reset counter
            return True
        
        return False
    
    def save_call(self, function_call: FunctionCall) -> None:
        """Save a function call to storage"""
        log_file = self._get_log_file(function_call.function_signature.name)
        function_name = function_call.function_signature.name
        
        if self.format == "jsonl":
            # Check if rotation is needed before writing
            if self._should_rotate(log_file, function_name):
                self._rotate_jsonl_file(log_file, self.rotation_batch_size)
            
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
            
            # For JSON format, implement rotation by keeping only recent entries
            if len(calls) > self.rotation_batch_size * 2 and self._get_file_size(log_file) > self.max_file_size_bytes:
                calls = calls[-self.rotation_batch_size:]  # Keep only the most recent entries
            
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
    
    def get_log_file_info(self, function_name: str, date: Optional[str] = None) -> dict:
        """
        Get information about a log file
        
        Args:
            function_name: Name of the function
            date: Date in YYYYMMDD format, if None uses today
            
        Returns:
            Dictionary with file information including size, entry count, etc.
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        log_file = self.log_dir / f"{function_name}_{date}.{self.format}"
        
        info = {
            "file_path": str(log_file),
            "exists": log_file.exists(),
            "size_bytes": 0,
            "size_mb": 0.0,
            "entry_count": 0,
            "rotation_needed": False,
            "entries_until_rotation": self.rotation_batch_size
        }
        
        if log_file.exists():
            size_bytes = self._get_file_size(log_file)
            info.update({
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "entry_count": self._count_jsonl_entries(log_file) if self.format == "jsonl" else self._count_json_entries(log_file),
                "rotation_needed": size_bytes >= self.max_file_size_bytes,
                "entries_until_rotation": self.rotation_batch_size - self._entry_counter.get(function_name, 0)
            })
        
        return info
    
    def _count_json_entries(self, file_path: Path) -> int:
        """Count the number of entries in a JSON array file"""
        if not file_path.exists():
            return 0
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 1
        except (json.JSONDecodeError, FileNotFoundError):
            return 0
    
    def get_all_log_files_info(self) -> dict:
        """
        Get information about all log files
        
        Returns:
            Dictionary mapping function names to their log file information
        """
        functions = self.get_all_functions()
        return {func: self.get_log_file_info(func) for func in functions}
    
    def force_rotate_log(self, function_name: str, date: Optional[str] = None, entries_to_remove: Optional[int] = None) -> bool:
        """
        Force rotation of a specific log file
        
        Args:
            function_name: Name of the function
            date: Date in YYYYMMDD format, if None uses today
            entries_to_remove: Number of entries to remove, if None uses rotation_batch_size
            
        Returns:
            True if rotation was performed, False if file doesn't exist
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        log_file = self.log_dir / f"{function_name}_{date}.{self.format}"
        
        if not log_file.exists():
            return False
        
        entries_to_remove = entries_to_remove or self.rotation_batch_size
        
        if self.format == "jsonl":
            self._rotate_jsonl_file(log_file, entries_to_remove)
        else:
            # For JSON format, rotate by keeping only recent entries
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    calls = json.load(f)
                
                if len(calls) > entries_to_remove:
                    calls = calls[entries_to_remove:]
                    with open(log_file, "w", encoding="utf-8") as f:
                        json.dump(calls, f, indent=2, default=str)
                        f.flush()
                        os.fsync(f.fileno())
                else:
                    # Clear the file if we're removing all entries
                    log_file.write_text("[]", encoding="utf-8")
            except (json.JSONDecodeError, FileNotFoundError):
                return False
        
        # Reset entry counter for this function
        self._entry_counter[function_name] = 0
        
        return True
    
    def set_rotation_settings(self, max_file_size_mb: Optional[float] = None, 
                            rotation_batch_size: Optional[int] = None) -> None:
        """
        Update rotation settings
        
        Args:
            max_file_size_mb: New maximum file size in MB
            rotation_batch_size: New rotation batch size
        """
        if max_file_size_mb is not None:
            self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        
        if rotation_batch_size is not None:
            self.rotation_batch_size = rotation_batch_size
    
    def get_rotation_settings(self) -> dict:
        """
        Get current rotation settings
        
        Returns:
            Dictionary with current rotation settings
        """
        return {
            "max_file_size_mb": self.max_file_size_bytes / (1024 * 1024),
            "max_file_size_bytes": self.max_file_size_bytes,
            "rotation_batch_size": self.rotation_batch_size,
            "entry_counters": dict(self._entry_counter)
        } 