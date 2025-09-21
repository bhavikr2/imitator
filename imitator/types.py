"""
Type definitions for function monitoring and I/O logging
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
import inspect


class FunctionSignature(BaseModel):
    """Function signature information"""
    name: str
    parameters: Dict[str, str]  # param_name -> type_annotation
    return_type: Optional[str] = None
    
    @classmethod
    def from_function(cls, func) -> "FunctionSignature":
        """Extract signature from a function"""
        sig = inspect.signature(func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                parameters[param_name] = str(param.annotation)
            else:
                parameters[param_name] = "Any"
        
        return_type = None
        if sig.return_annotation != inspect.Signature.empty:
            return_type = str(sig.return_annotation)
            
        return cls(
            name=func.__name__,
            parameters=parameters,
            return_type=return_type
        )


class IORecord(BaseModel):
    """Input/Output record for a function call"""
    inputs: Dict[str, Any]
    output: Any
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_ms: Optional[float] = None
    input_modifications: Optional[Dict[str, Dict[str, Any]]] = None  # Track in-place modifications
    
    class Config:
        arbitrary_types_allowed = True


class FunctionCall(BaseModel):
    """Complete function call record with signature and I/O"""
    function_signature: FunctionSignature
    io_record: IORecord
    call_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    
    class Config:
        arbitrary_types_allowed = True 


class TimeInterval(BaseModel):
    """Time interval for querying stored function calls"""
    start_time: datetime
    end_time: Optional[datetime] = None
    time_zone: Optional[str] = None

    def normalized_bounds(self) -> Tuple[datetime, Optional[datetime]]:
        """Return start and end datetimes normalized to the specified time zone.

        If time_zone is provided, interpret naive datetimes as that zone and return
        zone-aware datetimes. If no time_zone is provided, return as-is.
        """
        if self.time_zone:
            tz = ZoneInfo(self.time_zone)
            def _to_tz(dt: Optional[datetime]) -> Optional[datetime]:
                if dt is None:
                    return None
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=tz)
                return dt.astimezone(tz)
            start = _to_tz(self.start_time)
            end = _to_tz(self.end_time) if self.end_time else None
            return start, end
        return self.start_time, self.end_time