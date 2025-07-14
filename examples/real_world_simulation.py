#!/usr/bin/env python3
"""
Real-World Simulation Example - Imitator Framework

This example simulates real-world scenarios where function monitoring
would be valuable, including:
- E-commerce order processing
- User authentication system
- Data analytics pipeline
- Machine learning model inference
- File processing system
"""

import sys
import os
import random
import time
import hashlib
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imitator import monitor_function, LocalStorage


# Custom storage for simulation
simulation_storage = LocalStorage(log_dir="simulation_logs", format="jsonl")


# Enums for structured data
class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


# Data classes for structured data
@dataclass
class Product:
    id: str
    name: str
    price: float
    category: str
    stock: int


@dataclass
class User:
    id: str
    username: str
    email: str
    role: UserRole
    created_at: datetime


@dataclass
class Order:
    id: str
    user_id: str
    products: List[Dict[str, Any]]
    total: float
    status: OrderStatus
    created_at: datetime


# Simulation 1: E-commerce Order Processing System
class ECommerceSystem:
    """Simulates an e-commerce order processing system."""
    
    def __init__(self):
        self.products = self._initialize_products()
        self.users = self._initialize_users()
        self.orders = {}
        self.order_counter = 1
    
    def _initialize_products(self) -> Dict[str, Product]:
        """Initialize sample products."""
        products = [
            Product("P001", "Laptop", 999.99, "Electronics", 10),
            Product("P002", "Mouse", 29.99, "Electronics", 50),
            Product("P003", "Keyboard", 79.99, "Electronics", 25),
            Product("P004", "Monitor", 299.99, "Electronics", 15),
            Product("P005", "Headphones", 199.99, "Electronics", 30),
            Product("P006", "Book", 19.99, "Books", 100),
            Product("P007", "Notebook", 5.99, "Stationery", 200),
            Product("P008", "Pen", 2.99, "Stationery", 500),
        ]
        return {p.id: p for p in products}
    
    def _initialize_users(self) -> Dict[str, User]:
        """Initialize sample users."""
        users = [
            User("U001", "john_doe", "john@example.com", UserRole.USER, datetime.now()),
            User("U002", "jane_smith", "jane@example.com", UserRole.USER, datetime.now()),
            User("U003", "admin", "admin@example.com", UserRole.ADMIN, datetime.now()),
            User("U004", "guest", "guest@example.com", UserRole.GUEST, datetime.now()),
        ]
        return {u.id: u for u in users}
    
    @monitor_function(storage=simulation_storage)
    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product information by ID."""
        if product_id not in self.products:
            return None
        
        product = self.products[product_id]
        return {
            "id": product.id,
            "name": product.name,
            "price": product.price,
            "category": product.category,
            "stock": product.stock,
            "available": product.stock > 0
        }
    
    @monitor_function(storage=simulation_storage)
    def search_products(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for products by name or category."""
        results = []
        query_lower = query.lower()
        
        for product in self.products.values():
            name_match = query_lower in product.name.lower()
            category_match = category is None or product.category.lower() == category.lower()
            
            if name_match and category_match:
                results.append({
                    "id": product.id,
                    "name": product.name,
                    "price": product.price,
                    "category": product.category,
                    "stock": product.stock
                })
        
        return results
    
    @monitor_function(storage=simulation_storage)
    def calculate_order_total(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate order total with taxes and shipping."""
        if not items:
            return {"subtotal": 0, "tax": 0, "shipping": 0, "total": 0, "error": "No items"}
        
        subtotal = 0
        invalid_items = []
        
        for item in items:
            product_id = item.get("product_id")
            quantity = item.get("quantity", 1)
            
            if product_id not in self.products:
                invalid_items.append(product_id)
                continue
            
            product = self.products[product_id]
            if product.stock < quantity:
                invalid_items.append(f"{product_id} (insufficient stock)")
                continue
            
            subtotal += product.price * quantity
        
        if invalid_items:
            return {"error": f"Invalid items: {', '.join(invalid_items)}"}
        
        # Calculate tax (8.5%)
        tax = subtotal * 0.085
        
        # Calculate shipping (free over $100)
        shipping = 0 if subtotal >= 100 else 9.99
        
        total = subtotal + tax + shipping
        
        return {
            "subtotal": round(subtotal, 2),
            "tax": round(tax, 2),
            "shipping": round(shipping, 2),
            "total": round(total, 2)
        }
    
    @monitor_function(storage=simulation_storage)
    def create_order(self, user_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a new order."""
        if user_id not in self.users:
            return {"error": "Invalid user ID"}
        
        # Calculate order total
        total_info = self.calculate_order_total(items)
        if "error" in total_info:
            return total_info
        
        # Create order
        order_id = f"ORD{self.order_counter:06d}"
        self.order_counter += 1
        
        order = Order(
            id=order_id,
            user_id=user_id,
            products=items,
            total=total_info["total"],
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.orders[order_id] = order
        
        # Update product stock
        for item in items:
            product_id = item["product_id"]
            quantity = item["quantity"]
            self.products[product_id].stock -= quantity
        
        return {
            "order_id": order_id,
            "total": total_info["total"],
            "status": order.status.value,
            "created_at": order.created_at.isoformat()
        }
    
    @monitor_function(storage=simulation_storage)
    def update_order_status(self, order_id: str, new_status: str) -> Dict[str, Any]:
        """Update order status."""
        if order_id not in self.orders:
            return {"error": "Order not found"}
        
        try:
            status = OrderStatus(new_status)
        except ValueError:
            return {"error": f"Invalid status: {new_status}"}
        
        old_status = self.orders[order_id].status
        self.orders[order_id].status = status
        
        return {
            "order_id": order_id,
            "old_status": old_status.value,
            "new_status": status.value,
            "updated_at": datetime.now().isoformat()
        }


# Simulation 2: User Authentication System
class AuthenticationSystem:
    """Simulates a user authentication system."""
    
    def __init__(self):
        self.users = {}
        self.sessions = {}
        self.failed_attempts = defaultdict(int)
        self.locked_accounts = set()
    
    @monitor_function(storage=simulation_storage)
    def hash_password(self, password: str) -> str:
        """Hash a password for secure storage."""
        # Simple hash simulation (in real world, use proper hashing)
        salt = "secure_salt_123"
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    
    @monitor_function(storage=simulation_storage)
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new user."""
        # Validate input
        if not username or not email or not password:
            return {"error": "All fields required"}
        
        if username in self.users:
            return {"error": "Username already exists"}
        
        if len(password) < 6:
            return {"error": "Password must be at least 6 characters"}
        
        # Create user
        user_id = f"U{len(self.users) + 1:03d}"
        hashed_password = self.hash_password(password)
        
        self.users[username] = {
            "id": user_id,
            "username": username,
            "email": email,
            "password": hashed_password,
            "role": UserRole.USER.value,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "user_id": user_id,
            "username": username,
            "status": "registered"
        }
    
    @monitor_function(storage=simulation_storage)
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate a user login."""
        # Check if account is locked
        if username in self.locked_accounts:
            return {"error": "Account locked due to multiple failed attempts"}
        
        # Check if user exists
        if username not in self.users:
            self.failed_attempts[username] += 1
            return {"error": "Invalid username or password"}
        
        # Verify password
        hashed_password = self.hash_password(password)
        if self.users[username]["password"] != hashed_password:
            self.failed_attempts[username] += 1
            
            # Lock account after 3 failed attempts
            if self.failed_attempts[username] >= 3:
                self.locked_accounts.add(username)
                return {"error": "Account locked due to multiple failed attempts"}
            
            return {"error": "Invalid username or password"}
        
        # Success - reset failed attempts
        self.failed_attempts[username] = 0
        
        # Create session
        session_id = hashlib.md5(f"{username}{datetime.now()}".encode()).hexdigest()
        self.sessions[session_id] = {
            "user_id": self.users[username]["id"],
            "username": username,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24)
        }
        
        return {
            "session_id": session_id,
            "user_id": self.users[username]["id"],
            "username": username,
            "role": self.users[username]["role"]
        }
    
    @monitor_function(storage=simulation_storage)
    def validate_session(self, session_id: str) -> Dict[str, Any]:
        """Validate a user session."""
        if session_id not in self.sessions:
            return {"error": "Invalid session"}
        
        session = self.sessions[session_id]
        if datetime.now() > session["expires_at"]:
            del self.sessions[session_id]
            return {"error": "Session expired"}
        
        return {
            "valid": True,
            "user_id": session["user_id"],
            "username": session["username"],
            "expires_at": session["expires_at"].isoformat()
        }


# Simulation 3: Data Analytics Pipeline
class DataAnalytics:
    """Simulates a data analytics pipeline."""
    
    @monitor_function(storage=simulation_storage)
    def clean_data(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Clean and validate raw data."""
        if not raw_data:
            return {"cleaned_data": [], "issues": ["No data provided"]}
        
        cleaned_data = []
        issues = []
        
        for i, record in enumerate(raw_data):
            cleaned_record = {}
            record_issues = []
            
            # Validate and clean each field
            for field, value in record.items():
                if value is None:
                    record_issues.append(f"Missing value for {field}")
                    continue
                
                # Clean string fields
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    if not cleaned_value:
                        record_issues.append(f"Empty string for {field}")
                        continue
                    cleaned_record[field] = cleaned_value
                
                # Validate numeric fields
                elif isinstance(value, (int, float)):
                    if value < 0 and field in ["price", "quantity", "age"]:
                        record_issues.append(f"Negative value for {field}")
                        continue
                    cleaned_record[field] = value
                
                else:
                    cleaned_record[field] = value
            
            if record_issues:
                issues.append(f"Record {i}: {', '.join(record_issues)}")
            
            if cleaned_record:  # Only include if not empty
                cleaned_data.append(cleaned_record)
        
        return {
            "cleaned_data": cleaned_data,
            "original_count": len(raw_data),
            "cleaned_count": len(cleaned_data),
            "issues": issues
        }
    
    @monitor_function(storage=simulation_storage)
    def calculate_metrics(self, data: List[Dict[str, Any]], metric_field: str) -> Dict[str, Any]:
        """Calculate statistical metrics for a field."""
        if not data:
            return {"error": "No data provided"}
        
        values = []
        for record in data:
            if metric_field in record:
                try:
                    value = float(record[metric_field])
                    values.append(value)
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return {"error": f"No valid numeric values found for field '{metric_field}'"}
        
        # Calculate metrics
        total = sum(values)
        count = len(values)
        mean = total / count
        
        sorted_values = sorted(values)
        median = sorted_values[count // 2] if count % 2 == 1 else \
                (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2
        
        variance = sum((x - mean) ** 2 for x in values) / count
        std_dev = variance ** 0.5
        
        return {
            "field": metric_field,
            "count": count,
            "sum": total,
            "mean": mean,
            "median": median,
            "min": min(values),
            "max": max(values),
            "std_dev": std_dev,
            "variance": variance
        }
    
    @monitor_function(storage=simulation_storage)
    def generate_report(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive data report."""
        if not data:
            return {"error": "No data provided"}
        
        # Basic statistics
        record_count = len(data)
        
        # Field analysis
        field_stats = {}
        all_fields = set()
        
        for record in data:
            all_fields.update(record.keys())
        
        for field in all_fields:
            field_values = [record.get(field) for record in data]
            non_null_values = [v for v in field_values if v is not None]
            
            field_stats[field] = {
                "total_count": len(field_values),
                "non_null_count": len(non_null_values),
                "null_count": len(field_values) - len(non_null_values),
                "null_percentage": (len(field_values) - len(non_null_values)) / len(field_values) * 100
            }
            
            # Type analysis
            if non_null_values:
                types = set(type(v).__name__ for v in non_null_values)
                field_stats[field]["types"] = list(types)
        
        return {
            "record_count": record_count,
            "field_count": len(all_fields),
            "fields": list(all_fields),
            "field_statistics": field_stats,
            "generated_at": datetime.now().isoformat()
        }


# Simulation 4: Machine Learning Model Inference
class MLModelSimulator:
    """Simulates machine learning model inference."""
    
    def __init__(self):
        self.model_loaded = True
        self.prediction_count = 0
    
    @monitor_function(storage=simulation_storage)
    def preprocess_features(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess features for model input."""
        if not raw_features:
            return {"error": "No features provided"}
        
        processed_features = {}
        
        for feature_name, value in raw_features.items():
            # Simulate feature preprocessing
            if isinstance(value, str):
                # String encoding simulation
                processed_features[f"{feature_name}_encoded"] = len(value)
            elif isinstance(value, (int, float)):
                # Numeric normalization simulation
                processed_features[f"{feature_name}_normalized"] = min(max(value / 100, 0), 1)
            else:
                processed_features[feature_name] = value
        
        return {
            "features": processed_features,
            "feature_count": len(processed_features),
            "original_features": list(raw_features.keys())
        }
    
    @monitor_function(storage=simulation_storage)
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the ML model."""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        if not features:
            return {"error": "No features provided"}
        
        # Simulate model inference
        time.sleep(0.01)  # Simulate processing time
        
        # Mock prediction logic
        feature_sum = sum(v for v in features.values() if isinstance(v, (int, float)))
        
        # Simulate different prediction scenarios
        if feature_sum > 50:
            prediction = "high_risk"
            confidence = 0.85 + random.random() * 0.1
        elif feature_sum > 20:
            prediction = "medium_risk"
            confidence = 0.70 + random.random() * 0.15
        else:
            prediction = "low_risk"
            confidence = 0.60 + random.random() * 0.25
        
        self.prediction_count += 1
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "feature_count": len(features),
            "prediction_id": f"pred_{self.prediction_count:06d}",
            "timestamp": datetime.now().isoformat()
        }
    
    @monitor_function(storage=simulation_storage)
    def batch_predict(self, batch_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make predictions for a batch of feature sets."""
        if not batch_features:
            return {"error": "No batch features provided"}
        
        predictions = []
        for i, features in enumerate(batch_features):
            try:
                prediction = self.predict(features)
                predictions.append(prediction)
            except Exception as e:
                predictions.append({
                    "error": str(e),
                    "feature_index": i
                })
        
        return {
            "predictions": predictions,
            "batch_size": len(batch_features),
            "successful_predictions": len([p for p in predictions if "error" not in p])
        }


def run_ecommerce_simulation():
    """Run e-commerce system simulation."""
    print("\n🛒 E-commerce System Simulation")
    print("=" * 50)
    
    ecommerce = ECommerceSystem()
    
    # Product searches
    print("Product searches:")
    laptop_results = ecommerce.search_products("laptop")
    print(f"  Search 'laptop': {len(laptop_results)} results")
    
    electronics = ecommerce.search_products("", "Electronics")
    print(f"  Electronics category: {len(electronics)} products")
    
    # Order creation
    print("\nOrder processing:")
    order_items = [
        {"product_id": "P001", "quantity": 1},
        {"product_id": "P002", "quantity": 2}
    ]
    
    order_result = ecommerce.create_order("U001", order_items)
    if "error" not in order_result:
        print(f"  Order created: {order_result['order_id']}, Total: ${order_result['total']}")
        
        # Update order status
        update_result = ecommerce.update_order_status(order_result['order_id'], "processing")
        print(f"  Order updated: {update_result['old_status']} → {update_result['new_status']}")
    
    # Test error cases
    invalid_order = ecommerce.create_order("U999", order_items)
    print(f"  Invalid user order: {invalid_order}")


def run_auth_simulation():
    """Run authentication system simulation."""
    print("\n🔐 Authentication System Simulation")
    print("=" * 50)
    
    auth = AuthenticationSystem()
    
    # User registration
    print("User registration:")
    reg_result = auth.register_user("test_user", "test@example.com", "securepass123")
    print(f"  Registration: {reg_result}")
    
    # Authentication attempts
    print("\nAuthentication attempts:")
    
    # Successful login
    login_result = auth.authenticate_user("test_user", "securepass123")
    if "session_id" in login_result:
        print(f"  Login successful: {login_result['username']}")
        
        # Validate session
        session_result = auth.validate_session(login_result['session_id'])
        print(f"  Session validation: {session_result['valid']}")
    
    # Failed login attempts
    for i in range(3):
        failed_result = auth.authenticate_user("test_user", "wrongpass")
        print(f"  Failed attempt {i+1}: {failed_result['error']}")


def run_analytics_simulation():
    """Run data analytics simulation."""
    print("\n📊 Data Analytics Simulation")
    print("=" * 50)
    
    analytics = DataAnalytics()
    
    # Sample data with some issues
    raw_data = [
        {"name": "John", "age": 30, "salary": 50000, "city": "New York"},
        {"name": "Jane", "age": 25, "salary": 60000, "city": ""},  # Empty city
        {"name": "", "age": 35, "salary": 45000, "city": "Boston"},  # Empty name
        {"name": "Bob", "age": -5, "salary": 55000, "city": "Chicago"},  # Invalid age
        {"name": "Alice", "age": 28, "salary": None, "city": "Seattle"},  # Missing salary
    ]
    
    # Clean data
    print("Data cleaning:")
    clean_result = analytics.clean_data(raw_data)
    print(f"  Cleaned {clean_result['original_count']} → {clean_result['cleaned_count']} records")
    print(f"  Issues found: {len(clean_result['issues'])}")
    
    # Calculate metrics
    print("\nMetrics calculation:")
    if clean_result['cleaned_data']:
        salary_metrics = analytics.calculate_metrics(clean_result['cleaned_data'], "salary")
        if "error" not in salary_metrics:
            print(f"  Salary metrics: mean=${salary_metrics['mean']:.0f}, median=${salary_metrics['median']:.0f}")
    
    # Generate report
    print("\nReport generation:")
    report = analytics.generate_report(clean_result['cleaned_data'])
    print(f"  Report: {report['record_count']} records, {report['field_count']} fields")


def run_ml_simulation():
    """Run machine learning simulation."""
    print("\n🤖 Machine Learning Simulation")
    print("=" * 50)
    
    ml_model = MLModelSimulator()
    
    # Feature preprocessing
    print("Feature preprocessing:")
    raw_features = {
        "income": 75000,
        "age": 35,
        "education": "Bachelor",
        "employment_years": 8
    }
    
    processed = ml_model.preprocess_features(raw_features)
    print(f"  Processed {processed['feature_count']} features")
    
    # Single prediction
    print("\nSingle prediction:")
    prediction = ml_model.predict(processed['features'])
    print(f"  Prediction: {prediction['prediction']} (confidence: {prediction['confidence']})")
    
    # Batch prediction
    print("\nBatch prediction:")
    batch_features = [
        {"income_normalized": 0.3, "age_normalized": 0.35, "education_encoded": 8},
        {"income_normalized": 0.6, "age_normalized": 0.45, "education_encoded": 7},
        {"income_normalized": 0.8, "age_normalized": 0.55, "education_encoded": 10}
    ]
    
    batch_result = ml_model.batch_predict(batch_features)
    print(f"  Batch results: {batch_result['successful_predictions']}/{batch_result['batch_size']} successful")


def analyze_simulation_logs():
    """Analyze logs from all simulations."""
    print("\n📈 Simulation Log Analysis")
    print("=" * 50)
    
    functions = simulation_storage.get_all_functions()
    print(f"Total monitored functions: {len(functions)}")
    
    # Group functions by system
    systems = {
        "E-commerce": ["get_product_by_id", "search_products", "calculate_order_total", "create_order", "update_order_status"],
        "Authentication": ["hash_password", "register_user", "authenticate_user", "validate_session"],
        "Analytics": ["clean_data", "calculate_metrics", "generate_report"],
        "ML": ["preprocess_features", "predict", "batch_predict"]
    }
    
    for system_name, system_functions in systems.items():
        print(f"\n🔧 {system_name} System:")
        system_calls = 0
        system_errors = 0
        
        for func_name in system_functions:
            if func_name in functions:
                calls = simulation_storage.load_calls(func_name)
                errors = sum(1 for call in calls if isinstance(call.io_record.output, dict) and "error" in call.io_record.output)
                system_calls += len(calls)
                system_errors += errors
                
                if calls:
                    times = [call.io_record.execution_time_ms for call in calls if call.io_record.execution_time_ms]
                    avg_time = sum(times) / len(times) if times else 0
                    print(f"  {func_name}: {len(calls)} calls, {errors} errors, {avg_time:.2f}ms avg")
        
        print(f"  Total: {system_calls} calls, {system_errors} errors")
    
    print(f"\n📁 All simulation logs stored in: {simulation_storage.log_dir}")


def main():
    """Run all real-world simulations."""
    print("🌟 Imitator Framework - Real-World Simulation")
    print("=" * 60)
    
    # Run all simulations
    run_ecommerce_simulation()
    run_auth_simulation()
    run_analytics_simulation()
    run_ml_simulation()
    
    print("\n" + "=" * 60)
    print("✅ All simulations completed!")
    
    # Analyze all logs
    analyze_simulation_logs()


if __name__ == "__main__":
    main() 