# goph419-f2025-lab02-stFC
Overview

This project implements a Python module for solving linear systems using iterative methods (Gauss-Seidel and Jacobi) and generating spline interpolation functions for datasets. The goal is to analyze air and water density variations over temperature ranges as part of a larger PDE solver project for multi-phase fluid flow through granular materials.

The project demonstrates:

Iterative linear solvers for single and multiple right-hand-side systems.

Linear, quadratic, and cubic spline interpolation functions.

Unit testing of custom functions and validation against standard libraries (numpy.linalg.solve and scipy.interpolate.UnivariateSpline).

Visualization of spline fits for air and water density data.

Project Structure
project_root/
│
├── src/source
│          ├── linalg_interp.py       # Module: gauss_iter_solve and spline_function
│          ├── driver.py              # Script: loads data, performs analysis, plots results
│          ├── tests.py               # Unit tests for module functions
│       └── data/                     # Contains air and water density text files
│
├       ── figures/                   # Generated plots
└── README.md

Installation & Requirements

Python 3.8 or higher

NumPy

SciPy

Matplotlib

Install dependencies:

pip install numpy scipy matplotlib