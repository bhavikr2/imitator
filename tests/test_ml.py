import time
import pytest
import shutil
from pathlib import Path
from imitator import train, LocalStorage, FunctionMonitor
import imitator.ml

@pytest.fixture
def setup_clean_env():
    """Setup clean environment for ML tests"""
    test_log_dir = "tests/test_logs_ml"
    if Path(test_log_dir).exists():
        shutil.rmtree(test_log_dir)
        
    storage = LocalStorage(log_dir=test_log_dir, buffer_size=1) 
    monitor = FunctionMonitor(storage=storage)
    
    # Reset global daemon
    if imitator.ml._daemon:
        imitator.ml._daemon.stop()
    imitator.ml._daemon = None
    
    yield monitor, storage, test_log_dir
    
    # Cleanup
    if imitator.ml._daemon:
        imitator.ml._daemon.stop()
    if Path(test_log_dir).exists():
        shutil.rmtree(test_log_dir)

def test_training_flow(setup_clean_env):
    monitor, storage, _ = setup_clean_env
    
    # Define function
    @train(min_batch_size=10, monitor=monitor)
    def linear_func(x: float, y: float) -> float:
        return 2.0 * x + 3.0 * y
        
    # Generate data
    import random
    for _ in range(50):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        linear_func(x, y)
        
    # Wait for monitor to process async saves
    monitor.wait_for_all_saves(timeout=2.0)
    
    # Flush storage to disk
    storage.flush()
    
    # Get daemon and speed it up
    daemon = imitator.ml.get_training_daemon(monitor.storage)
    daemon.check_interval = 0.1
    
    # Wait for training
    start = time.time()
    trainer = None
    while time.time() - start < 5:
        trainer = daemon.get_trainer("linear_func")
        if trainer and trainer.is_fitted and trainer.training_count >= 40:
            break
        time.sleep(0.1)
        
    assert trainer is not None
    assert trainer.is_fitted
    assert trainer.training_count >= 10
    
    # Test prediction
    pred = trainer.predict({"x": 2.0, "y": 2.0})
    assert pred is not None
    
    # Allow some error margin for SGD
    assert abs(pred - 10.0) < 1.0

