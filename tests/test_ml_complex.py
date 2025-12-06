import time
import pytest
import shutil
import numpy as np
from pathlib import Path
from imitator import train, LocalStorage, FunctionMonitor
import imitator.ml

@pytest.fixture
def clean_ml_env():
    """Fixture to manage ML test environment"""
    def _setup(dir_name):
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
        if imitator.ml._daemon:
            imitator.ml._daemon.stop()
        imitator.ml._daemon = None
        storage = LocalStorage(log_dir=dir_name, buffer_size=1)
        monitor = FunctionMonitor(storage=storage)
        return monitor, storage

    yield _setup

    # Cleanup after test
    if imitator.ml._daemon:
        imitator.ml._daemon.stop()

def test_image_classification_simulation(clean_ml_env):
    """
    Simulates a function that determines if an image is 'bright' (sum of pixels > threshold).
    """
    monitor, storage = clean_ml_env("tests/test_logs_image")
    
    @train(min_batch_size=20, monitor=monitor)
    def classify_brightness(image_matrix: list, image_id: str) -> float:
        # image_id is an irrelevant string argument that should be ignored
        total = np.sum(image_matrix)
        # Threshold for 100 pixels with values 0-1 is 50
        return 1.0 if total > 50.0 else 0.0

    # Generate data
    import random
    
    for i in range(100):
        # Create 10x10 image
        is_bright = random.random() > 0.5
        if is_bright:
            # Generate bright image (avg pixel > 0.5)
            img = [[random.uniform(0.6, 1.0) for _ in range(10)] for _ in range(10)]
        else:
            # Generate dark image (avg pixel < 0.4)
            img = [[random.uniform(0.0, 0.4) for _ in range(10)] for _ in range(10)]
            
        classify_brightness(img, f"img_{i}")

    monitor.wait_for_all_saves()
    storage.flush()
    
    # Train
    daemon = imitator.ml.get_training_daemon(monitor.storage)
    daemon.check_interval = 0.1
    
    # Wait for trainer
    start = time.time()
    trainer = None
    while time.time() - start < 10:
        trainer = daemon.get_trainer("classify_brightness")
        if trainer and trainer.is_fitted and trainer.training_count >= 80:
            break
        time.sleep(0.1)
        
    assert trainer is not None
    assert trainer.is_fitted
    
    # Test Prediction
    bright_test = [[0.9] * 10] * 10
    dark_test = [[0.1] * 10] * 10
    
    pred_bright = trainer.predict({"image_matrix": bright_test, "image_id": "test_1"})
    pred_dark = trainer.predict({"image_matrix": dark_test, "image_id": "test_2"})
    
    # Check if predictions are directionally correct
    assert pred_bright > 0.7
    assert pred_dark < 0.3
    
    if Path("tests/test_logs_image").exists():
        shutil.rmtree("tests/test_logs_image")

def test_math_matrix_processing(clean_ml_env):
    """
    Simulates a function that computes the trace of a 2x2 matrix.
    """
    monitor, storage = clean_ml_env("tests/test_logs_math")
    
    @train(min_batch_size=20, monitor=monitor)
    def compute_trace(matrix: list) -> float:
        return matrix[0][0] + matrix[1][1]

    # Generate data
    import random
    for _ in range(100):
        m = [
            [random.uniform(0, 10), random.uniform(0, 10)],
            [random.uniform(0, 10), random.uniform(0, 10)]
        ]
        compute_trace(m)

    monitor.wait_for_all_saves()
    storage.flush()
    
    daemon = imitator.ml.get_training_daemon(monitor.storage)
    daemon.check_interval = 0.1
    
    start = time.time()
    trainer = None
    while time.time() - start < 5:
        trainer = daemon.get_trainer("compute_trace")
        if trainer and trainer.is_fitted and trainer.training_count >= 80:
            break
        time.sleep(0.1)
        
    # Test
    # Identity matrix trace should be 2
    test_m = [[1.0, 0.0], [0.0, 1.0]]
    pred = trainer.predict({"matrix": test_m})
    
    assert abs(pred - 2.0) < 0.2
    
    if Path("tests/test_logs_math").exists():
        shutil.rmtree("tests/test_logs_math")

def test_robustness_filtering(clean_ml_env):
    """
    Tests filtering of errors and robust feature extraction
    """
    monitor, storage = clean_ml_env("tests/test_logs_robust")
    
    @train(min_batch_size=5, monitor=monitor)
    def flaky_function(val: float, metadata: dict) -> float:
        if val < 0:
            raise ValueError("Negative input")
        return val * 2

    # Generate mixed data
    # 1. Valid data
    for i in range(10):
        flaky_function(float(i), {"id": i})
        
    # 2. Error data (should be ignored by trainer)
    try:
        flaky_function(-1.0, {"id": "error"})
    except ValueError:
        pass
        
    # 3. Data with extra irrelevant metadata (should be handled by feature extractor)
    for i in range(10):
        flaky_function(float(i), {"id": i, "extra": "stuff"})

    monitor.wait_for_all_saves()
    storage.flush()
    
    daemon = imitator.ml.get_training_daemon(monitor.storage)
    daemon.check_interval = 0.1
    
    start = time.time()
    trainer = None
    while time.time() - start < 5:
        trainer = daemon.get_trainer("flaky_function")
        if trainer and trainer.is_fitted:
            break
        time.sleep(0.1)
        
    assert trainer is not None
    # We generated 20 valid calls and 1 error call.
    # The error call should be filtered out.
    # The dictionary 'metadata' should be ignored in features.
    
    pred = trainer.predict({"val": 5.0, "metadata": {"some": "thing"}})
    
    assert abs(pred - 10.0) < 0.5
    
    if Path("tests/test_logs_robust").exists():
        shutil.rmtree("tests/test_logs_robust")

