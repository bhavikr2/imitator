"""
Test PostgreSQL database connector with physics simulation
"""

import pytest
import numpy as np
from imitator import monitor_function, DatabaseStorage, PostgreSQLConnector
from imitator.types import FunctionCall, FunctionSignature
from datetime import datetime
from imitator.types import TimeInterval

class TestPostgreSQLConnector:
    """Test PostgreSQL database connector with physics simulation"""
    
    @pytest.fixture
    def postgres_connector(self):
        """Fixture to provide PostgreSQL connector for testing"""
        # Use test database connection
        connector = PostgreSQLConnector(
            connection_string="postgresql://postgres:password@localhost:5432/postgres",
            table_name="test_function_calls"
        )
        return connector
    
    @pytest.fixture
    def postgres_storage(self, postgres_connector):
        """Fixture to provide DatabaseStorage with PostgreSQL connector"""
        return DatabaseStorage(postgres_connector, buffer_size=5)
    
    def test_postgres_connection(self, postgres_connector):
        """Test PostgreSQL connection establishment"""
        connected = postgres_connector.connect()
        assert connected is True
        assert postgres_connector._connection is not None
        disconnected = postgres_connector.disconnect()
        assert disconnected is True
    
    def test_physics_simulation_postgres(self, postgres_storage, clean_logs):
        """Test physics simulation with PostgreSQL database streaming"""
        
        start_time = datetime.now()

        # Simple physics simulation functions
        @monitor_function(storage=postgres_storage)
        def calculate_kinetic_energy(mass: float, velocity: float) -> float:
            """Calculate kinetic energy: KE = 0.5 * mass * velocity^2"""
            return 0.5 * mass * (velocity ** 2)
        
        @monitor_function(storage=postgres_storage) 
        def calculate_pendulum_period(length: float, gravity: float = 9.81) -> float:
            """Calculate pendulum period: T = 2π * sqrt(L/g)"""
            return 2 * np.pi * np.sqrt(length / gravity)
        
        @monitor_function(storage=postgres_storage)
        def simulate_projectile_motion(initial_velocity: float, angle_degrees: float, 
                                     gravity: float = 9.81) -> dict:
            """Simulate projectile motion and return trajectory data"""
            angle_rad = np.radians(angle_degrees)
            
            # Calculate components
            vx = initial_velocity * np.cos(angle_rad)
            vy = initial_velocity * np.sin(angle_rad)
            
            # Time of flight
            time_of_flight = (2 * vy) / gravity
            
            # Range
            range_val = vx * time_of_flight
            
            # Maximum height
            max_height = (vy ** 2) / (2 * gravity)
            
            return {
                "initial_velocity": initial_velocity,
                "angle_degrees": angle_degrees,
                "velocity_x": vx,
                "velocity_y": vy,
                "time_of_flight": time_of_flight,
                "range": range_val,
                "max_height": max_height
            }
        
        # Run physics simulations
        results = []
        
        # Test kinetic energy calculations
        for mass, velocity in [(1.0, 10.0), (2.0, 5.0), (0.5, 20.0)]:
            ke = calculate_kinetic_energy(mass, velocity)
            results.append(ke)
            expected = 0.5 * mass * (velocity ** 2)
            assert abs(ke - expected) < 1e-10
        
        # Test pendulum period calculations
        for length in [1.0, 2.0, 0.5]:
            period = calculate_pendulum_period(length)
            results.append(period)
            expected = 2 * np.pi * np.sqrt(length / 9.81)
            assert abs(period - expected) < 1e-10
        
        # Test projectile motion simulations
        projectile_results = []
        for velocity, angle in [(20.0, 45.0), (30.0, 30.0), (15.0, 60.0)]:
            trajectory = simulate_projectile_motion(velocity, angle)
            projectile_results.append(trajectory)
            
            # Verify basic physics principles
            assert trajectory["velocity_x"] > 0
            assert trajectory["max_height"] > 0
            assert trajectory["range"] > 0
        
        # Flush buffers to ensure all calls are saved
        postgres_storage.flush()
        postgres_storage.close()
        
        # Verify calls were logged to PostgreSQL
        connector = postgres_storage.connector
        connected = connector.connect()
        assert connected is True
        
        time_interval = TimeInterval(start_time=start_time)

        # Check kinetic energy calls
        ke_calls = connector.load_calls("calculate_kinetic_energy", time_interval=time_interval)
        assert len(ke_calls) == 3, f"Expected 3 kinetic energy calls, got {len(ke_calls)}"
        
        # Check pendulum calls
        pendulum_calls = connector.load_calls("calculate_pendulum_period", time_interval=time_interval)
        assert len(pendulum_calls) == 3, f"Expected 3 pendulum calls, got {len(pendulum_calls)}"
        
        # Check projectile calls
        projectile_calls = connector.load_calls("simulate_projectile_motion", time_interval=time_interval)
        assert len(projectile_calls) == 3, f"Expected 3 projectile calls, got {len(projectile_calls)}"
        
        # Verify function names
        functions = connector.get_all_functions()
        expected_functions = {"calculate_kinetic_energy", "calculate_pendulum_period", "simulate_projectile_motion"}
        assert expected_functions.issubset(functions)
        
        disconnected = connector.disconnect()
        assert disconnected is True
        
        return results, projectile_results
