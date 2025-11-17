#  tests1.py
import numpy as np
from linalg_interp import gauss_iter_solve, spline_function
from scipy.interpolate import UnivariateSpline

# --------------------------
# Tests for gauss_iter_solve()
# --------------------------

def test_gauss_iter_solve_single_rhs():
    """Test solving a system with a single RHS vector."""
    A = np.array([[4.0, -1.0, 0.0],
                  [-1.0, 4.0, -1.0],
                  [0.0, -1.0, 3.0]])
    b = np.array([15.0, 10.0, 10.0])
    
    x_seidel = gauss_iter_solve(A, b, alg='seidel')
    x_jacobi = gauss_iter_solve(A, b, alg='jacobi')
    x_expected = np.linalg.solve(A, b)
    
    assert np.allclose(x_seidel, x_expected, atol=1e-6), "Gauss-Seidel solution incorrect."
    assert np.allclose(x_jacobi, x_expected, atol=1e-6), "**Jacobi solution incorrect.**"

def test_gauss_iter_solve_inverse_matrix():
    """Test solving for multiple RHS vectors (inverse of A)."""
    A = np.array([[4.0, -1.0, 0.0],
                  [-1.0, 4.0, -1.0],
                  [0.0, -1.0, 3.0]])
    I = np.eye(3)
    A_inv = gauss_iter_solve(A, I, alg='seidel')
    identity_check = A @ A_inv
    
    assert np.allclose(identity_check, I, atol=1e-6), "Inverse matrix solution incorrect."

def test_gauss_iter_solve_invalid_input():
    """Test gauss_iter_solve() raises ValueError on invalid inputs."""
    A_non_square = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([1, 2, 3])
    
    try:
        gauss_iter_solve(A_non_square, b)
        assert False, "Did not raise ValueError for non-square A."
    except ValueError:
        pass

    A = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    try:
        gauss_iter_solve(A, b, alg='invalid')
        assert False, "Did not raise ValueError for invalid alg."
    except ValueError:
        pass

# --------------------------
# Tests for spline_function()
# --------------------------

def test_spline_function_linear_quadratic_cubic():
    """Test spline_function() for order 1,2,3 using known polynomials."""
    xd = np.linspace(0, 5, 6)
    
    # Linear data
    yd_lin = 2 * xd + 1
    f_lin1 = spline_function(xd, yd_lin, order=1)
    f_lin2 = spline_function(xd, yd_lin, order=2)
    f_lin3 = spline_function(xd, yd_lin, order=3)
    x_test = np.linspace(0, 5, 50)
    
    assert np.allclose(f_lin1(x_test), 2*x_test+1, atol=1e-6), "Order 1 linear spline failed."
    
    # Quadratic data
    yd_quad = xd**2
    f_quad2 = spline_function(xd, yd_quad, order=2)
    f_quad3 = spline_function(xd, yd_quad, order=3)
    assert np.allclose(f_quad2(x_test), x_test**2, atol=1e-5), "Order 2 quadratic spline failed."
    assert np.allclose(f_quad3(x_test), x_test**2, atol=1e-5), "Order 3 quadratic spline failed."

    # Cubic data
    yd_cubic = xd**3
    f_cubic3 = spline_function(xd, yd_cubic, order=3)
    assert np.allclose(f_cubic3(x_test), x_test**3, atol=1e-5), "Order 3 cubic spline failed."

def test_spline_function_vs_scipy():
    """Compare custom spline_function() with scipy's UnivariateSpline."""
    xd = np.linspace(0, 5, 6)
    yd = np.exp(0.2 * xd)
    
    f_custom = spline_function(xd, yd, order=3)
    f_scipy = UnivariateSpline(xd, yd, k=3, s=0, ext='raise')
    
    x_test = np.linspace(0, 5, 50)
    assert np.allclose(f_custom(x_test), f_scipy(x_test), atol=1e-6), "Custom spline vs scipy failed."

def test_spline_function_exceptions():
    """Test that spline_function() raises ValueError for invalid inputs."""
    xd = np.array([0, 1, 2, 2])
    yd = np.array([0, 1, 4, 8])
    
    # Repeated xd
    try:
        spline_function(xd, yd)
        assert False, "Did not raise ValueError for repeated xd."
    except ValueError:
        pass

    # Mismatched lengths
    try:
        spline_function(np.array([0,1,2]), np.array([0,1]))
        assert False, "Did not raise ValueError for mismatched lengths."
    except ValueError:
        pass

    # Unsorted xd
    try:
        spline_function(np.array([0,3,2]), np.array([0,9,4]))
        assert False, "Did not raise ValueError for unsorted xd."
    except ValueError:
        pass

    # Invalid order
    try:
        spline_function(np.array([0,1,2]), np.array([0,1,4]), order=5)
        assert False, "Did not raise ValueError for invalid order."
    except ValueError:
        pass

    # Out-of-range evaluation
    f = spline_function(np.array([0,1,2]), np.array([0,1,4]), order=2)
    try:
        f(-1)
        assert False, "Did not raise ValueError for x < xmin."
    except ValueError:
        pass
    try:
        f(3)
        assert False, "Did not raise ValueError for x > xmax."
    except ValueError:
        pass

# --------------------------
# Run all tests
# --------------------------

if __name__ == "__main__":
    tests = [
        test_gauss_iter_solve_single_rhs,
        test_gauss_iter_solve_inverse_matrix,
        test_gauss_iter_solve_invalid_input,
        test_spline_function_linear_quadratic_cubic,
        test_spline_function_vs_scipy,
        test_spline_function_exceptions
    ]
    
    for test in tests:
        try:
            test()
            print(f"{test.__name__}: PASS")
        except AssertionError as e:
            print(f"{test.__name__}: FAIL ({e})")
        except Exception as e:
            print(f"{test.__name__}: ERROR ({e})")