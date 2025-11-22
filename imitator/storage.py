"""
Local storage backend for function I/O logs
"""

import json
import os
from pathlib import Path
from typing import List, Protocol, Optional, DefaultDict, Any
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import concurrent.futures
import logging

from .types import FunctionCall, TimeInterval, FunctionSignature, IORecord
from psycopg2 import sql


def _default_time_interval() -> TimeInterval:
    """Create a default TimeInterval starting 1 hour ago"""
    now = datetime.now()
    start_time = now - timedelta(hours=1)
    # Don't set end_time to allow for calls that happen after this interval is created
    return TimeInterval(start_time=start_time, end_time=None)


class DatabaseConnector(Protocol):
    """Protocol for database connectors"""

    def connect(self) -> bool:
        """Establish connection to the database
        
        Returns:
            True if connection was successful, False otherwise
        """
        ...

    def disconnect(self) -> bool:
        """Close database connection
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        ...

    def save_calls(self, calls: List[FunctionCall], function_name: str) -> None:
        """Save function calls to the database in batch"""
        ...

    def load_calls(
        self, function_name: str, time_interval: Optional[TimeInterval] = None
    ) -> List[FunctionCall]:
        """Load function calls from the database"""
        ...

    def get_all_functions(self) -> List[str]:
        """Get list of all monitored functions in the database"""
        ...


class LocalStorage:
    """Local file-based storage for function call logs"""

    def __init__(
        self,
        log_dir: str = "logs",
        format: str = "jsonl",
        buffer_size: int = 100,
        flush_interval: Optional[float] = None,
    ):
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
        self._buffer: DefaultDict[str, List[FunctionCall]] = defaultdict(list)
        self._lock = threading.Lock()
        self.log_dir.mkdir(exist_ok=True)
        # Optionally start a background flush thread if flush_interval is set
        # Register atexit handler to flush on shutdown

    def _get_log_file(self, function_name: str) -> Path:
        """Get log file path for a function"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"{function_name}_{timestamp}.{self.format}"

    def save_call(self, function_call: FunctionCall) -> None:
        """Save a function call to storage"""
        with self._lock:
            fn = function_call.function_signature.name
            self._buffer[fn].append(function_call)
            if len(self._buffer[fn]) >= self.buffer_size:
                self._flush_function(fn)

    def _flush_function(self, function_name: str) -> None:
        """Write all buffered calls for a function to disk, then clear buffer"""
        log_file = self._get_log_file(function_name)
        calls_to_write = self._buffer[function_name]
        self._buffer[function_name] = []  # Clear buffer after flushing

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

    def flush(self) -> None:
        """Flush all buffers"""
        with self._lock:
            for function_name in list(self._buffer.keys()):  # Iterate over a copy
                self._flush_function(function_name)

    def close(self) -> None:
        """Flush and clean up (stop background thread if any)"""
        self.flush()
        # No background thread to stop in this simple implementation

    def _get_dates_to_scan(self, time_interval: TimeInterval) -> set[str]:
        start, end = time_interval.normalized_bounds()
        # Generate all dates from start to end (inclusive)
        dates_to_scan = set()

        if end is None:
            # If no end date, just add the start date
            dates_to_scan.add(start.strftime("%Y%m%d"))
        else:
            # Use for loop with date range
            num_days = (end - start).days + 1
            for day_offset in range(num_days):
                current_date = start + timedelta(days=day_offset)
                dates_to_scan.add(current_date.strftime("%Y%m%d"))

        return dates_to_scan

    def _within_time_interval(
        self, call_id_timestamp: float, start_timestamp: float, end_timestamp: float
    ) -> bool:
        """Check if a call_id timestamp is within a time interval"""
        return (
            call_id_timestamp >= start_timestamp
            and call_id_timestamp <= end_timestamp
        )

    def load_calls(
        self, function_name: str, time_interval: Optional[TimeInterval] = None
    ) -> List[FunctionCall]:
        """
        Load function calls from storage

        Args:
            function_name: Name of the function
        """
        if time_interval is None:
            time_interval = _default_time_interval()

        start, end = time_interval.normalized_bounds()

        # Convert time_interval to timestamps for call_id filtering if provided
        start_timestamp = start.timestamp()
        end_timestamp = (
            end.timestamp() if end is not None else datetime.now().timestamp()
        )

        calls = []

        dates_to_scan = self._get_dates_to_scan(time_interval)
        for date_str in sorted(dates_to_scan):
            log_file = self.log_dir / f"{function_name}_{date_str}.{self.format}"
            if not log_file.exists():
                continue

            if self.format == "jsonl":
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            call_data = json.loads(line)
                            call_id_timestamp = float(call_data.get("call_id", "0"))
                            if self._within_time_interval(
                                call_id_timestamp, start_timestamp, end_timestamp
                            ):
                                calls.append(FunctionCall.model_validate(call_data))
            else:
                with open(log_file, "r", encoding="utf-8") as f:
                    call_data_list = json.load(f)
                    for call_data in call_data_list:
                        call_id_timestamp = float(call_data.get("call_id", "0"))
                        if self._within_time_interval(
                            call_id_timestamp, start_timestamp, end_timestamp
                        ):
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

            match = re.match(r"(.+)_\d{8}$", stem)
            if match:
                function_name = match.group(1)
                functions.add(function_name)
        return list(functions)


class DatabaseStorage(LocalStorage):
    """Database storage backend that uses database connectors with non-blocking
    operations"""

    def __init__(
        self, 
        connector: DatabaseConnector, 
        buffer_size: int = 100, 
        flush_interval: Optional[float] = None
    ) -> None:
        """
        Initialize database storage

        Args:
            connector: Database connector instance
            buffer_size: Number of calls to buffer before flushing
            flush_interval: Interval in seconds for automatic flushing (None for
                manual)
        """
        super().__init__(buffer_size=buffer_size, flush_interval=flush_interval)
        self.connector = connector
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._connected = False

    def _ensure_connected(self) -> None:
        """Ensure database connection is established
        
        Raises:
            ConnectionError: If connection cannot be established
        """
        if not self._connected:
            success = self.connector.connect()
            if success:
                self._connected = True
            else:
                raise ConnectionError(
                    "Failed to establish database connection. "
                    "Please check your connection settings."
                )

    def _flush_function(self, function_name: str) -> None:
        """Write buffered calls for a function to database in background thread"""
        calls_to_write = self._buffer[function_name]
        self._buffer[function_name] = []  # Clear buffer immediately

        if calls_to_write:
            # Submit database operation to background thread
            self._executor.submit(
                self._save_calls_async, calls_to_write, function_name
            )

    def _save_calls_async(
        self, calls: List[FunctionCall], function_name: str
    ) -> None:
        """Save calls to database in background thread (non-blocking)"""
        try:
            self._ensure_connected()
            self.connector.save_calls(calls, function_name)
        except Exception as e:
            # Log error but don't crash the application
            logging.error(f"Failed to save calls to database: {e}")
            # Optionally, you could implement retry logic here

    def flush(self) -> None:
        """Flush all buffers to database"""
        with self._lock:
            for function_name in list(self._buffer.keys()):
                if self._buffer[function_name]:
                    self._flush_function(function_name)

    def close(self) -> None:
        """Flush buffers and close database connection"""
        self.flush()
        self._executor.shutdown(wait=True)  # Wait for all background operations
        if self._connected:
            self.connector.disconnect()
            # Always reset connection flag to allow reconnection
            self._connected = False

    def load_calls(
        self, function_name: str, time_interval: Optional[TimeInterval] = None
    ) -> List[FunctionCall]:
        """Load function calls from database"""
        if time_interval is None:
            time_interval = _default_time_interval()
        self._ensure_connected()
        return self.connector.load_calls(function_name, time_interval=time_interval)

    def get_all_functions(self) -> List[str]:
        """Get list of all monitored functions from database"""
        self._ensure_connected()
        return self.connector.get_all_functions()


# Database Connector Implementations


class PostgreSQLConnector:
    """PostgreSQL database connector for function call logs"""

    def __init__(self, connection_string: str, table_name: str = "function_calls"):
        """
        Initialize PostgreSQL connector

        Args:
            connection_string: PostgreSQL connection string
            table_name: Name of the table to store function calls
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self._connection: Optional[Any] = None

    def connect(self) -> bool:
        """Establish connection to PostgreSQL database
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            import psycopg2

            self._connection = psycopg2.connect(self.connection_string)
            table_exists = self._ensure_table_exists()
            if not table_exists:
                # Connection established but table creation failed
                # Close connection and return False
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None
                return False
            return True
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. Install it with '"
                "pip install psycopg2-binary'"
            )
        except Exception as e:
            # Log the exception for debugging
            logging.debug(f"PostgreSQL connection failed: {e}")
            self._connection = None
            return False

    def _ensure_table_exists(self) -> bool:
        """Ensure the function calls table exists
        
        Returns:
            True if table exists or was created successfully, False otherwise
        """
        if self._connection is None:
            return False

        try:
            create_table_query = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    call_id VARCHAR(255) UNIQUE,
                    function_name VARCHAR(255) NOT NULL,
                    inputs JSONB NOT NULL,
                    output JSONB,
                    timestamp TIMESTAMP NOT NULL,
                    execution_time_ms FLOAT,
                    input_modifications JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            ).format(table=sql.Identifier(self.table_name))

            with self._connection.cursor() as cursor:
                cursor.execute(create_table_query)
                self._connection.commit()
            return True
        except Exception as e:
            # Log the exception for debugging
            logging.debug(f"Failed to ensure table exists: {e}")
            return False

    def disconnect(self) -> bool:
        """Close PostgreSQL connection
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
                return True
            except Exception:
                self._connection = None
                return False
        return True  # Already disconnected

    def save_calls(self, calls: List[FunctionCall], function_name: str) -> None:
        """Save function calls to PostgreSQL in batch"""
        if self._connection is None:
            raise ConnectionError("Database connection not established")

        insert_query = sql.SQL(
            """
            INSERT INTO {table} (call_id, function_name, inputs, output, timestamp,
            execution_time_ms, input_modifications)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (call_id) DO NOTHING
        """
        ).format(table=sql.Identifier(self.table_name))

        data = []
        for call in calls:
            data.append(
                (
                    call.call_id,
                    function_name,
                    json.dumps(call.io_record.inputs, default=str),
                    (
                        json.dumps(call.io_record.output, default=str)
                        if call.io_record.output is not None
                        else None
                    ),
                    call.io_record.timestamp,
                    call.io_record.execution_time_ms,
                    (
                        json.dumps(call.io_record.input_modifications, default=str)
                        if call.io_record.input_modifications
                        else None
                    ),
                )
            )

        with self._connection.cursor() as cursor:
            cursor.executemany(insert_query, data)
            self._connection.commit()

    def load_calls(
        self, function_name: str, time_interval: Optional[TimeInterval] = None
    ) -> List[FunctionCall]:
        """Load function calls from PostgreSQL"""
        if time_interval is None:
            time_interval = _default_time_interval()

        if self._connection is None:
            raise ConnectionError("Database connection not established")

        # Get normalized time bounds
        start, end = time_interval.normalized_bounds()

        # Build query with time filtering
        if end is not None:
            query = sql.SQL(
                """
                SELECT call_id, inputs, output, timestamp, execution_time_ms,
                input_modifications
                FROM {table}
                WHERE function_name = %s AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp
            """
            ).format(table=sql.Identifier(self.table_name))
            params = [function_name, start, end]
        else:
            query = sql.SQL(
                """
                SELECT call_id, inputs, output, timestamp, execution_time_ms,
                input_modifications
                FROM {table}
                WHERE function_name = %s AND timestamp >= %s
                ORDER BY timestamp
            """
            ).format(table=sql.Identifier(self.table_name))
            params = [function_name, start]

        with self._connection.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()

        calls = []
        for row in results:
            (
                call_id,
                inputs_json,
                output_json,
                timestamp,
                execution_time_ms,
                input_modifications_json,
            ) = row

            # Create FunctionSignature (simplified for loading)
            function_signature = FunctionSignature(
                name=function_name, parameters={}, return_type=None
            )

            # Create IORecord
            # Note: PostgreSQL JSONB columns are automatically deserialized by psycopg2
            io_record = IORecord(
                inputs=inputs_json,
                output=output_json,
                timestamp=timestamp,
                execution_time_ms=execution_time_ms,
                input_modifications=input_modifications_json,
            )

            calls.append(
                FunctionCall(
                    function_signature=function_signature,
                    io_record=io_record,
                    call_id=call_id,
                )
            )

        return calls

    def get_all_functions(self) -> List[str]:
        """Get list of all monitored functions from PostgreSQL"""
        if self._connection is None:
            raise ConnectionError("Database connection not established")

        query = sql.SQL(
            """
            SELECT DISTINCT function_name
            FROM {table}
            ORDER BY function_name
        """
        ).format(table=sql.Identifier(self.table_name))

        with self._connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()

        return [row[0] for row in results]


class MongoDBConnector:
    """MongoDB database connector for function call logs"""

    def __init__(
        self,
        connection_string: str,
        database_name: str = "function_monitor",
        collection_name: str = "function_calls",
    ):
        """
        Initialize MongoDB connector

        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database
            collection_name: Name of the collection to store function calls
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self._client: Optional[Any] = None
        self._database: Optional[Any] = None
        self._collection: Optional[Any] = None

    def connect(self) -> bool:
        """Establish connection to MongoDB database
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            import pymongo
            from pymongo.errors import ConnectionFailure

            self._client = pymongo.MongoClient(self.connection_string)
            # Verify connection
            self._client.admin.command("ping")
            self._database = self._client[self.database_name]
            self._collection = self._database[self.collection_name]

            # Create indexes for better query performance
            self._collection.create_index([("function_name", 1), ("timestamp", -1)])
            self._collection.create_index([("call_id", 1)], unique=True)
            return True

        except ImportError:
            raise ImportError(
                "pymongo is required for MongoDB support. Install it with '"
                "pip install pymongo'"
            )
        except ConnectionFailure:
            self._client = None
            self._database = None
            self._collection = None
            return False
        except Exception:
            self._client = None
            self._database = None
            self._collection = None
            return False

    def disconnect(self) -> bool:
        """Close MongoDB connection
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        if self._client:
            try:
                self._client.close()
                self._client = None
                self._database = None
                self._collection = None
                return True
            except Exception:
                self._client = None
                self._database = None
                self._collection = None
                return False
        return True  # Already disconnected

    def save_calls(self, calls: List[FunctionCall], function_name: str) -> None:
        """Save function calls to MongoDB in batch"""
        if self._collection is None:
            raise ConnectionError("Database connection not established")

        documents = []
        for call in calls:
            document = {
                "call_id": call.call_id,
                "function_name": function_name,
                "inputs": call.io_record.inputs,
                "output": call.io_record.output,
                "timestamp": call.io_record.timestamp,
                "execution_time_ms": call.io_record.execution_time_ms,
                "input_modifications": call.io_record.input_modifications,
                "created_at": datetime.now(),
            }
            documents.append(document)

        if documents:
            try:
                self._collection.insert_many(documents, ordered=False)
            except Exception as e:
                # If bulk insert fails, try individual inserts
                for doc in documents:
                    try:
                        self._collection.insert_one(doc)
                    except Exception:
                        # Skip duplicates and continue
                        continue

    def load_calls(
        self, function_name: str, time_interval: Optional[TimeInterval] = None
    ) -> List[FunctionCall]:
        """Load function calls from MongoDB"""
        if time_interval is None:
            time_interval = _default_time_interval()

        if self._collection is None:
            raise ConnectionError("Database connection not established")

        query: Dict[str, Any] = {"function_name": function_name}

        start, end = time_interval.normalized_bounds()
        if end is None:
            query["timestamp"] = {"$gte": start}
        else:
            query["timestamp"] = {"$gte": start, "$lte": end}

        cursor = self._collection.find(query).sort("timestamp", 1)

        calls = []
        for doc in cursor:
            # Create FunctionSignature (simplified for loading)
            function_signature = FunctionSignature(
                name=function_name, parameters={}, return_type=None
            )

            # Create IORecord
            io_record = IORecord(
                inputs=doc["inputs"],
                output=doc["output"],
                timestamp=doc["timestamp"],
                execution_time_ms=doc.get("execution_time_ms"),
                input_modifications=doc.get("input_modifications"),
            )

            calls.append(
                FunctionCall(
                    function_signature=function_signature,
                    io_record=io_record,
                    call_id=doc["call_id"],
                )
            )

        return calls

    def get_all_functions(self) -> List[str]:
        """Get list of all monitored functions from MongoDB"""
        if self._collection is None:
            raise ConnectionError("Database connection not established")

        if self._collection is not None:
            return list(self._collection.distinct("function_name"))
        return []


class CouchbaseConnector:
    """Couchbase database connector for function call logs"""

    def __init__(
        self,
        connection_string: str,
        bucket_name: str = "function_monitor",
        scope_name: str = "_default",
        collection_name: str = "function_calls",
    ):
        """
        Initialize Couchbase connector

        Args:
            connection_string: Couchbase connection string
            bucket_name: Name of the bucket
            scope_name: Name of the scope
            collection_name: Name of the collection to store function calls
        """
        self.connection_string = connection_string
        self.bucket_name = bucket_name
        self.scope_name = scope_name
        self.collection_name = collection_name
        self._cluster: Optional[Any] = None
        self._bucket: Optional[Any] = None
        self._collection: Optional[Any] = None

    def connect(self) -> bool:
        """Establish connection to Couchbase database
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            from couchbase.cluster import Cluster
            from couchbase.options import ClusterOptions
            from couchbase.auth import PasswordAuthenticator
            from couchbase.exceptions import CouchbaseException

            # Parse connection string (format:
            # couchbase://host1,host2?username=user&password=pass)
            import urllib.parse

            parsed = urllib.parse.urlparse(self.connection_string)
            hosts = parsed.netloc.split(",")

            # Extract credentials from query parameters
            query_params = urllib.parse.parse_qs(parsed.query)
            username = query_params.get("username", [""])[0]
            password = query_params.get("password", [""])[0]

            if not username or not password:
                raise ValueError(
                    "Username and password must be provided in connection string"
                )

            auth = PasswordAuthenticator(username, password)
            cluster_options = ClusterOptions(auth)

            self._cluster = Cluster(f"couchbase://{','.join(hosts)}", cluster_options)

            # Wait for cluster to be ready
            from datetime import timedelta

            self._cluster.wait_until_ready(timedelta(seconds=5))

            self._bucket = self._cluster.bucket(self.bucket_name)
            self._collection = self._bucket.scope(self.scope_name).collection(
                self.collection_name
            )
            return True

        except ImportError:
            raise ImportError(
                "couchbase is required for Couchbase support. Install it with '"
                "pip install couchbase'"
            )
        except CouchbaseException as e:
            self._cluster = None
            self._bucket = None
            self._collection = None
            return False
        except Exception:
            self._cluster = None
            self._bucket = None
            self._collection = None
            return False

    def disconnect(self) -> bool:
        """Close Couchbase connection
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        if self._cluster:
            try:
                self._cluster.close()
                self._cluster = None
                self._bucket = None
                self._collection = None
                return True
            except Exception:
                self._cluster = None
                self._bucket = None
                self._collection = None
                return False
        return True  # Already disconnected

    def save_calls(self, calls: List[FunctionCall], function_name: str) -> None:
        """Save function calls to Couchbase in batch"""
        if self._collection is None:
            raise ConnectionError("Database connection not established")

        from couchbase.exceptions import DocumentExistsException

        for call in calls:
            document_id = f"function_call::{call.call_id}"
            document = {
                "call_id": call.call_id,
                "function_name": function_name,
                "inputs": call.io_record.inputs,
                "output": call.io_record.output,
                "timestamp": call.io_record.timestamp.isoformat(),
                "execution_time_ms": call.io_record.execution_time_ms,
                "input_modifications": call.io_record.input_modifications,
                "created_at": datetime.now().isoformat(),
            }

            try:
                self._collection.insert(document_id, document)
            except DocumentExistsException:
                # Skip duplicates
                continue
            except Exception as e:
                # Log other errors but continue
                logging.error(f"Failed to save call {call.call_id} to Couchbase: {e}")

    def load_calls(
        self, function_name: str, time_interval: Optional[TimeInterval] = None
    ) -> List[FunctionCall]:
        """Load function calls from Couchbase"""
        if time_interval is None:
            time_interval = _default_time_interval()

        if self._collection is None:
            raise ConnectionError("Database connection not established")

        from datetime import datetime

        # Build N1QL query
        query = f"""
            SELECT META().id, call_id, inputs, output, timestamp, execution_time_ms,
            input_modifications
            FROM `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`
            WHERE function_name = $function_name
        """

        params = {"function_name": function_name}

        if time_interval is not None:
            start, end = time_interval.normalized_bounds()
            if end is None:
                query += " AND timestamp >= $start_time"
                params.update({"start_time": start.isoformat()})
            else:
                query += " AND timestamp >= $start_time AND timestamp <= $end_time"
                params.update(
                    {"start_time": start.isoformat(), "end_time": end.isoformat()}
                )

        query += " ORDER BY timestamp"

        calls = []
        try:
            if self._cluster is not None:
                result = self._cluster.query(query, params)
                rows = result.rows()
            else:
                raise ConnectionError("Database connection not established")

            for row in rows:
                (
                    document_id,
                    call_id,
                    inputs,
                    output,
                    timestamp_str,
                    execution_time_ms,
                    input_modifications,
                ) = row

                # Create FunctionSignature (simplified for loading)
                function_signature = FunctionSignature(
                    name=function_name, parameters={}, return_type=None
                )

                # Parse timestamp
                timestamp = datetime.fromisoformat(timestamp_str)

                # Create IORecord
                io_record = IORecord(
                    inputs=inputs,
                    output=output,
                    timestamp=timestamp,
                    execution_time_ms=execution_time_ms,
                    input_modifications=input_modifications,
                )

                calls.append(
                    FunctionCall(
                        function_signature=function_signature,
                        io_record=io_record,
                        call_id=call_id,
                    )
                )
        except Exception as e:
            raise ConnectionError(f"Failed to query Couchbase: {e}")

        return calls


class SQLiteConnector:
    """SQLite database connector for function call logs"""

    def __init__(self, db_path: str, table_name: str = "function_calls"):
        """
        Initialize SQLite connector

        Args:
            db_path: Path to the SQLite database file (e.g., "/tmp/imitator.db" or ":memory:")
            table_name: Name of the table to store function calls
        """
        self.db_path = db_path
        self.table_name = table_name
        self._connection: Optional[Any] = None

    def connect(self) -> bool:
        """Establish connection to SQLite database
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            import sqlite3

            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.execute("PRAGMA journal_mode=WAL;")
            table_exists = self._ensure_table_exists()
            return table_exists
        except Exception:
            self._connection = None
            return False

    def _ensure_table_exists(self) -> bool:
        """Ensure the function calls table exists
        
        Returns:
            True if table exists or was created successfully, False otherwise
        """
        if self._connection is None:
            return False

        try:
            create_table_query = (
                f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "call_id TEXT UNIQUE,"
                "function_name TEXT NOT NULL,"
                "inputs TEXT NOT NULL,"
                "output TEXT,"
                "timestamp_epoch REAL NOT NULL,"
                "execution_time_ms REAL,"
                "input_modifications TEXT,"
                "created_at REAL DEFAULT (strftime('%s','now'))"
                ")"
            )

            with self._connection:
                self._connection.execute(create_table_query)
            return True
        except Exception:
            return False

    def disconnect(self) -> bool:
        """Close SQLite connection
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
                return True
            except Exception:
                self._connection = None
                return False
        return True  # Already disconnected

    def save_calls(self, calls: List[FunctionCall], function_name: str) -> None:
        """Save function calls to SQLite in batch"""
        if self._connection is None:
            raise ConnectionError("Database connection not established")

        insert_query = (
            f"INSERT OR IGNORE INTO {self.table_name} "
            "(call_id, function_name, inputs, output, timestamp_epoch, "
            "execution_time_ms, input_modifications) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        )

        data = []
        for call in calls:
            timestamp_epoch = call.io_record.timestamp.timestamp()
            data.append(
                (
                    call.call_id,
                    function_name,
                    json.dumps(call.io_record.inputs, default=str),
                    (
                        json.dumps(call.io_record.output, default=str)
                        if call.io_record.output is not None
                        else None
                    ),
                    timestamp_epoch,
                    call.io_record.execution_time_ms,
                    (
                        json.dumps(call.io_record.input_modifications, default=str)
                        if call.io_record.input_modifications
                        else None
                    ),
                )
            )

        with self._connection:
            self._connection.executemany(insert_query, data)

    def load_calls(
        self, function_name: str, time_interval: Optional[TimeInterval] = None
    ) -> List[FunctionCall]:
        """Load function calls from SQLite"""
        if time_interval is None:
            time_interval = _default_time_interval()

        if self._connection is None:
            raise ConnectionError("Database connection not established")

        # Get normalized time bounds
        start, end = time_interval.normalized_bounds()
        start_ts = start.timestamp()
        end_ts = end.timestamp() if end is not None else datetime.now().timestamp()

        if end is not None:
            query = (
                "SELECT call_id, inputs, output, timestamp_epoch, "
                "execution_time_ms, input_modifications "
                f"FROM {self.table_name} "
                "WHERE function_name = ? AND timestamp_epoch >= ? "
                "AND timestamp_epoch <= ? ORDER BY timestamp_epoch"
            )
            params: List[Any] = [function_name, start_ts, end_ts]
        else:
            query = (
                "SELECT call_id, inputs, output, timestamp_epoch, "
                "execution_time_ms, input_modifications "
                f"FROM {self.table_name} "
                "WHERE function_name = ? AND timestamp_epoch >= ? "
                "ORDER BY timestamp_epoch"
            )
            params = [function_name, start_ts]

        cursor = self._connection.execute(query, params)
        rows = cursor.fetchall()

        calls: List[FunctionCall] = []
        for row in rows:
            (
                call_id,
                inputs_json,
                output_json,
                timestamp_epoch,
                execution_time_ms,
                input_modifications_json,
            ) = row

            function_signature = FunctionSignature(
                name=function_name, parameters={}, return_type=None
            )

            timestamp_dt = datetime.fromtimestamp(float(timestamp_epoch))

            io_record = IORecord(
                inputs=json.loads(inputs_json),
                output=json.loads(output_json) if output_json is not None else None,
                timestamp=timestamp_dt,
                execution_time_ms=execution_time_ms,
                input_modifications=
                    json.loads(input_modifications_json)
                    if input_modifications_json is not None
                    else None,
            )

            calls.append(
                FunctionCall(
                    function_signature=function_signature,
                    io_record=io_record,
                    call_id=call_id,
                )
            )

        return calls

    def get_all_functions(self) -> List[str]:
        """Get list of all monitored functions from SQLite"""
        if self._connection is None:
            raise ConnectionError("Database connection not established")

        query = (
            f"SELECT DISTINCT function_name FROM {self.table_name} "
            "ORDER BY function_name"
        )
        cursor = self._connection.execute(query)
        return [row[0] for row in cursor.fetchall()]
