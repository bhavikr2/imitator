"""
Test MongoDB database connector with physics simulation
"""

import pytest
import numpy as np
from imitator import monitor_function, DatabaseStorage, MongoDBConnector
from imitator.types import FunctionCall, FunctionSignature
from datetime import datetime
from imitator.types import TimeInterval


class TestMongoDBConnector:
    """Test MongoDB database connector with physics simulation"""
    
    @pytest.fixture
    def mongodb_connector(self):
        """Fixture to provide MongoDB connector for testing"""
        # Use test database connection
        connector = MongoDBConnector(
            connection_string="mongodb://admin:password@localhost:27017/",
            database_name="test_function_monitor",
            collection_name="test_calls"
        )
        return connector
    
    @pytest.fixture
    def mongodb_storage(self, mongodb_connector):
        """Fixture to provide DatabaseStorage with MongoDB connector"""
        return DatabaseStorage(mongodb_connector, buffer_size=5)
    
    def test_mongodb_connection(self, mongodb_connector):
        """Test MongoDB connection establishment"""
        connected = mongodb_connector.connect()
        assert connected is True
        assert mongodb_connector._client is not None
        assert mongodb_connector._collection is not None
        disconnected = mongodb_connector.disconnect()
        assert disconnected is True
    
    def test_physics_simulation_mongodb(self, mongodb_storage, clean_logs):
        """Test physics simulation with MongoDB database streaming"""

        start_time = datetime.now()
        
        # Quantum mechanics simulation functions
        @monitor_function(storage=mongodb_storage)
        def calculate_traffic_flow(density: float, speed: float) -> float:
            """Calculate traffic flow: Flow = Density * Speed"""
            return density * speed
        
        @monitor_function(storage=mongodb_storage)
        def calculate_traffic_density(num_vehicles: int, road_length_km: float) -> float:
            """Calculate traffic density: Density = Number of Vehicles / Road Length"""
            if road_length_km == 0:
                raise ValueError("Road length cannot be zero.")
            return num_vehicles / road_length_km
        
        @monitor_function(storage=mongodb_storage)
        def calculate_travel_time(distance_km: float, speed_kmph: float) -> float:
            """Calculate travel time: Time = Distance / Speed"""
            if speed_kmph <= 0:
                raise ValueError("Speed must be greater than zero.")
            return distance_km / speed_kmph

        # Run traffic simulation
        results = []
        
        # Test traffic flow calculations
        for density, speed in [(20.0, 60.0), (30.0, 40.0), (10.0, 80.0)]:
            flow = calculate_traffic_flow(density, speed)
            results.append(flow)
            expected = density * speed
            assert abs(flow - expected) < 1e-9
        
        # Test traffic density calculations
        for num_vehicles, road_length in [(100, 5.0), (150, 10.0), (50, 2.5)]:
            density = calculate_traffic_density(num_vehicles, road_length)
            results.append(density)
            expected = num_vehicles / road_length
            assert abs(density - expected) < 1e-9
        
        # Test travel time calculations
        travel_time_results = []
        for distance, speed in [(100.0, 80.0), (50.0, 60.0), (200.0, 100.0)]:
            time = calculate_travel_time(distance, speed)
            travel_time_results.append(time)
            
            assert time > 0
        
        # Flush buffers to ensure all calls are saved
        mongodb_storage.flush()
        mongodb_storage.close()
        
        # Verify calls were logged to MongoDB
        connector = mongodb_storage.connector
        connected = connector.connect()
        assert connected is True
        
        time_interval = TimeInterval(start_time=start_time)

        # Check traffic flow calls
        flow_calls = connector.load_calls("calculate_traffic_flow", time_interval=time_interval)
        assert len(flow_calls) == 3, f"Expected 3 traffic flow calls, got {len(flow_calls)}"
        
        # Check traffic density calls
        density_calls = connector.load_calls("calculate_traffic_density", time_interval=time_interval)
        assert len(density_calls) == 3, f"Expected 3 traffic density calls, got {len(density_calls)}"
        
        # Check travel time calls
        travel_time_calls = connector.load_calls("calculate_travel_time", time_interval=time_interval)
        assert len(travel_time_calls) == 3, f"Expected 3 travel time calls, got {len(travel_time_calls)}"
        
        # Verify function names
        functions = connector.get_all_functions()
        expected_functions = {"calculate_traffic_flow", "calculate_traffic_density", "calculate_travel_time"}
        assert expected_functions.issubset(functions)
        
        disconnected = connector.disconnect()
        assert disconnected is True
        
        return results, travel_time_results
