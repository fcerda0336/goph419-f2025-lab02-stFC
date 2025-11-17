# driver.py
import numpy as np
import matplotlib.pyplot as plt
from linalg_interp import spline_function

def main():
    # File paths
    water_file = "../src/data/water_density_vs_temp_usgs.txt"
    air_file = "../src/data/air_density_vs_temp_eng_toolbox.txt"

    # -------------------------------
    # Load data
    # -------------------------------
    try:
        T_water, rho_water = np.loadtxt(water_file, unpack=True)
        T_air, rho_air = np.loadtxt(air_file, unpack=True)
    except OSError as e:
        print(f"Error loading data files: {e}")
        return

    # -------------------------------
    # Generate spline functions
    # -------------------------------
    orders = [1, 2, 3]

    splines_water = [spline_function(T_water, rho_water, order=o) for o in orders]
    splines_air = [spline_function(T_air, rho_air, order=o) for o in orders]

    # -------------------------------
    # Interpolate using 100 points
    # -------------------------------
    T_water_fine = np.linspace(T_water.min(), T_water.max(), 100)
    T_air_fine = np.linspace(T_air.min(), T_air.max(), 100)

    interp_water = [spline(T_water_fine) for spline in splines_water]
    interp_air = [spline(T_air_fine) for spline in splines_air]

    # -------------------------------
    # Plotting
    # -------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    # Water plots
    for i, y in enumerate(interp_water):
        ax = axes[i*2]
        ax.plot(T_water, rho_water, 'o', label='Data')
        ax.plot(T_water_fine, y, '-', label=f'Spline order {orders[i]}')
        ax.set_title(f'Water Density - Order {orders[i]}')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Density (g/cm³)')
        ax.legend()
        ax.grid(True)

    # Air plots
    for i, y in enumerate(interp_air):
        ax = axes[i*2 + 1]
        ax.plot(T_air, rho_air, 'o', label='Data')
        ax.plot(T_air_fine, y, '-', label=f'Spline order {orders[i]}')
        ax.set_title(f'Air Density - Order {orders[i]}')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Density (kg/m³)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    # -------------------------------
    # Save figures
    # -------------------------------
    import os
    figures_dir = "../figures"
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "density_splines.png")
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()