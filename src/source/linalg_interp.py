# linalg_interp.py

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import warnings


def gauss_iter_solve(A, b, x0=None, tol=1e-8, alg='seidel'):
    """
    Solve a linear system Ax = b using the Gauss-Seidel or Jacobi iterative method.

    Parameters
    ----------
    A : array_like
        Coefficient matrix (must be square).
    b : array_like
        Right-hand-side vector or matrix.
    x0 : array_like, optional
        Initial guess for x. If None, initializes with zeros.
    tol : float, optional
        Relative error tolerance for convergence (default=1e-8).
    alg : str, optional
        Algorithm flag, 'seidel' or 'jacobi' (case-insensitive).

    Returns
    -------
    numpy.ndarray
        Solution vector or matrix x. This will have the same shape as b.
    """

    # Convert inputs to numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Basic dimension checks
    if A.ndim != 2 or b.ndim > 2:
        raise ValueError("A must be 2D and b must be 1D or 2D array-like.")
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix A must be square.")
    if b.shape[0] != n:
        raise ValueError("Matrix A and vector b must have the same number of rows.")

    # Initialize x0
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x0 = np.array(x0, dtype=float)
        if x0.ndim == 1:
            x0 = x0.reshape(-1, 1)
        if x0.shape[0] != n:
            raise ValueError("Initial guess x0 has incompatible dimensions.")
        # Repeat column if b has multiple RHS
        if b.ndim == 2 and x0.shape[1] == 1:
            x = np.tile(x0, (1, b.shape[1]))
        elif b.ndim == 2 and x0.shape[1] != b.shape[1]:
            raise ValueError("x0 and b have incompatible column dimensions.")
        else:
            x = x0.copy()

    # Validate algorithm flag
    alg_flag = alg.strip().lower()
    if alg_flag not in ['seidel', 'jacobi']:
        raise ValueError("Invalid algorithm flag. Use 'seidel' or 'jacobi'.")

    max_iter = 10000
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("Matrix A has zero(s) on its diagonal; cannot iterate.")

    # ----------------------------
    # Iterative process
    # ----------------------------
    for k in range(max_iter):
        x_old = x.copy()

        if alg_flag == 'jacobi':
            # **Corrected Jacobi update rule**
            if b.ndim == 1:
                x = (b - (A - np.diagflat(D)) @ x_old)
                x = x / D
            else:
                x = (b - (A - np.diagflat(D)) @ x_old) / D[:, None]
            #x = (b - (A - np.diagflat(D)) @ x_old) / D[:, None]
            #x = (b - (A - np.diagflat(D)) @ x_old) / D[:, None]
        else:
            # Gauss-Seidel update rule
            for i in range(n):
                s1 = np.dot(A[i, :i], x[:i])
                s2 = np.dot(A[i, i+1:], x_old[i+1:])
                x[i] = (b[i] - s1 - s2) / A[i, i]

        # Check convergence (relative error)
        if np.linalg.norm(x - x_old) / np.linalg.norm(x) < tol:
            return x.squeeze()

    warnings.warn("Solution did not converge after maximum iterations.", RuntimeWarning)
    return x.squeeze()


def spline_function(xd, yd, order=3):
    """
    Generate a spline interpolation function.

    Parameters
    ----------
    xd : array_like
        Independent variable values (must be increasing).
    yd : array_like
        Dependent variable values (same shape as xd).
    order : int, optional
        Spline order: 1 (linear), 2 (quadratic), or 3 (cubic). Default is 3.

    Returns
    -------
    function
        A callable spline function f(x).
    """

    xd = np.array(xd, dtype=float).flatten()
    yd = np.array(yd, dtype=float).flatten()

    # Validation checks
    if xd.shape[0] != yd.shape[0]:
        raise ValueError("xd and yd must have the same length.")
    if np.unique(xd).shape[0] != xd.shape[0]:
        raise ValueError("xd contains repeated values.")
    if not np.allclose(np.sort(xd), xd):
        raise ValueError("xd must be in strictly increasing order.")
    if order not in [1, 2, 3]:
        raise ValueError("Spline order must be 1, 2, or 3.")

    xmin, xmax = xd.min(), xd.max()

    # Create spline function based on order
    if order == 1:
        spline = interp1d(xd, yd, kind='linear', bounds_error=True)
    else:
        spline = UnivariateSpline(xd, yd, k=order, s=0, ext='raise')

    # Define wrapped function that checks for range
    def f(x):
        x = np.array(x, dtype=float)
        if np.any((x < xmin) | (x > xmax)):
            raise ValueError(f"Input x is outside the range [{xmin}, {xmax}].")
        return spline(x)

    return f
