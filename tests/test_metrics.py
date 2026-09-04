"""Tests for metric calculations."""
import numpy as np
from src.utils.metrics_calculator import compute_metrics

def test_perfect_predictions():
    y = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y, y)
    assert abs(m["f1_weighted"] - 100.0) < 0.01

def test_random_predictions():
    np.random.seed(42)
    y_true = np.random.randint(0, 5, 100)
    y_pred = np.random.randint(0, 5, 100)
    m = compute_metrics(y_true, y_pred)
    assert 0 <= m["f1_weighted"] <= 100
