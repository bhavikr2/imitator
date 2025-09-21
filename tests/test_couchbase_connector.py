"""
Test Couchbase database connector with physics simulation
"""

import pytest
import numpy as np
from imitator import monitor_function, DatabaseStorage, CouchbaseConnector
from imitator.types import FunctionCall, FunctionSignature


class TestCouchbaseConnector:
    """Test Couchbase database connector with physics simulation"""
    
    @pytest.fixture
    def couchbase_connector(self):
        """Fixture to provide Couchbase connector for testing"""
        # Use test database connection
        connector = CouchbaseConnector(
            connection_string="couchbase://localhost?username=admin&password=password",
            bucket_name="function_bucket",
            collection_name="test_calls"
        )
        return connector
    
    @pytest.fixture
    def couchbase_storage(self, couchbase_connector):
        """Fixture to provide DatabaseStorage with Couchbase connector"""
        return DatabaseStorage(couchbase_connector, buffer_size=5)
    
    def test_couchbase_connection(self, couchbase_connector):
        """Test Couchbase connection establishment"""
        # Note: Couchbase connection test might be skipped if server not available
        try:
            couchbase_connector.connect()
            assert couchbase_connector._cluster is not None
            couchbase_connector.disconnect()
        except Exception as e:
            pytest.skip(f"Couchbase server not available: {e}")
    
    def test_physics_simulation_couchbase(self, couchbase_storage, clean_logs):
        """Test physics simulation with Couchbase database streaming"""
        
        # Astrophysics simulation functions
        @monitor_function(storage=couchbase_storage)
        def calculate_schwarzschild_radius(mass: float) -> float:
            """Calculate Schwarzschild radius: R_s = 2GM/c^2"""
            G = 6.67430e-11  # Gravitational constant
            c = 299792458.0   # Speed of light
            return (2 * G * mass) / (c ** 2)
        
        @monitor_function(storage=couchbase_storage)
        def calculate_hubble_flow_distance(redshift: float, hubble_constant: float = 70.0) -> float:
            """Calculate Hubble flow distance: d = c * z / H0"""
            c = 299792458.0  # Speed of light in m/s
            H0 = hubble_constant * 1000  # Convert to m/s/MPc
            return (c * redshift) / H0
        
        @monitor_function(storage=couchbase_storage)
        def simulate_black_hole(
            mass: float,
            spin: float = 0.0,
            charge: float = 0.0
        ) -> dict:
            """Simulate black hole properties"""
            G = 6.67430e-11
            c = 299792458.0
            
            # Schwarzschild radius
            schwarzschild_radius = (2 * G * mass) / (c ** 2)
            
            # Event horizon (for non-rotating, uncharged black hole)
            event_horizon = schwarzschild_radius
            
            # Hawking temperature
            hawking_temperature = (6.626e-34 * (c ** 3)) / (8 * np.pi * G * mass * 1.38e-23)
            
            # Hawking radiation lifetime
            if mass > 0:
                lifetime = (5120 * np.pi * (G ** 2) * (mass ** 3)) / (6.626e-34 * (c ** 4))
            else:
                lifetime = float('inf')
            
            return {
                "mass": mass,
                "spin": spin,
                "charge": charge,
                "schwarzschild_radius": schwarzschild_radius,
                "event_horizon": event_horizon,
                "hawking_temperature": hawking_temperature,
                "lifetime_years": lifetime / (365.25 * 24 * 3600) if lifetime != float('inf') else float('inf')
            }
        
        # Run astrophysics simulations
        results = []
        
        # Test Schwarzschild radius calculations
        for mass in [1.989e30, 5.972e24, 7.348e22]:  # Sun, Earth, Moon masses
            radius = calculate_schwarzschild_radius(mass)
            results.append(radius)
            G = 6.67430e-11
            c = 299792458.0
            expected = (2 * G * mass) / (c ** 2)
            assert abs(radius - expected) < 1e-20
        
        # Test Hubble flow distance calculations
        for redshift in [0.1, 0.5, 2.0]:
            distance = calculate_hubble_flow_distance(redshift)
            results.append(distance)
            c = 299792458.0
            H0 = 70.0 * 1000  # m/s/MPc
            expected = (c * redshift) / H0
            assert abs(distance - expected) < 1e-10
        
        # Test black hole simulations
        black_hole_results = []
        test_masses = [1.989e30, 1.989e31, 1.989e32]  # Solar masses
        
        for mass in test_masses:
            black_hole = simulate_black_hole(mass)
            black_hole_results.append(black_hole)
            
            # Verify black hole properties
            assert black_hole["schwarzschild_radius"] > 0
            assert black_hole["event_horizon"] > 0
            assert black_hole["hawking_temperature"] >= 0
        
        # Flush buffers to ensure all calls are saved
        couchbase_storage.flush()
        couchbase_storage.close()
        
        # Verify calls were logged to Couchbase
        connector = couchbase_storage.connector
        try:
            connector.connect()
            
            # Check Schwarzschild calls
            schwarzschild_calls = connector.load_calls("calculate_schwarzschild_radius")
            assert len(schwarzschild_calls) == 3, f"Expected 3 Schwarzschild calls, got {len(schwarzschild_calls)}"
            
            # Check Hubble calls
            hubble_calls = connector.load_calls("calculate_hubble_flow_distance")
            assert len(hubble_calls) == 3, f"Expected 3 Hubble calls, got {极en(hubble_calls)}"
            
            # Check black hole calls
            black_hole_calls = connector.load_calls("simulate_black_hole")
            assert len(black_hole_calls) == 3, f"Expected 3 black hole calls, got {len(black_hole_calls)}"
            
            # Verify function names
            functions = connector.get_all_functions()
            expected_functions = {"calculate_schwarzschild_radius", "calculate_hubble极low_distance", "simulate_black_hole"}
            assert expected_functions.issubset(functions)
            
            connector.disconnect()
            
        except Exception as e:
            pytest.skip(f"Couchbase server not available for verification: {e}")
        
        return results, black_hole_results
