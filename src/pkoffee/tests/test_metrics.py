import numpy as np
import pytest

def test_import():
    from pkoffee import metrics
    
def test_size_mismatch_valid():
    from pkoffee.metrics import check_size_match

    array_a = np.array([1, 2, 3])
    array_b = np.array([4, 5, 6])
    # Should not raise an error
    check_size_match(array_a, array_b)
    
def test_size_mismatch_invalid():
    from pkoffee.metrics import check_size_match, SizeMismatchError
    
    a = np.zeros(5)
    b = np.zeros(3)
    with pytest.raises(SizeMismatchError, match="Arrays must have same length"):
        check_size_match(a, b)

def test_compute_r2_fixed_value():
    from pkoffee.metrics import compute_r2

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    expected_r2 = 0.992
    assert np.isclose(compute_r2(y_true, y_pred), expected_r2, atol=1e-3)
    
def test_compute_r2_properties():
    from pkoffee.metrics import compute_r2
    rng = np.random.default_rng()
    y_true = rng.normal(size=10)
    
    # Perfect prediction
    assert np.isclose(compute_r2(y_true, y_true), 1.0)
    
    # Mean prediction
    y_mean = np.full_like(y_true, np.mean(y_true))
    assert np.isclose(compute_r2(y_true, y_mean), 0.0)
    
def test_rmse():
    from pkoffee.metrics import compute_rmse
    
    rng = np.random.default_rng()
    y_true = rng.normal(size=10)
    
    # Perfect prediction
    assert compute_rmse(y_true, y_true) == 0.0
    
def test_mae():
    from pkoffee.metrics import compute_mae
    
    rng = np.random.default_rng()
    y_true = rng.normal(size=10)
    
    # Perfect prediction
    assert compute_mae(y_true, y_true) == 0.0