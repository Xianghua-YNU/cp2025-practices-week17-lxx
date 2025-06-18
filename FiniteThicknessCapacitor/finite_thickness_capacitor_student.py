#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    Solve 2D Laplace equation using SOR method for finite thickness parallel plate capacitor.
    
    Args:
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        plate_thickness (int): Thickness of conductor plates in grid points
        plate_separation (int): Separation between plates in grid points
        omega (float): Relaxation factor (1.0 < omega < 2.0)
        max_iter (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        
    Returns:
        np.ndarray: 2D electric potential distribution
    """
    # Initialize potential grid
    potential = np.zeros((ny, nx))
    
    # Calculate plate positions
    lower_plate_start = (ny - plate_separation - 2 * plate_thickness) // 2
    lower_plate_end = lower_plate_start + plate_thickness
    upper_plate_start = lower_plate_end + plate_separation
    upper_plate_end = upper_plate_start + plate_thickness
    
    # Set boundary conditions
    potential[:, 0] = 0.0    # Left boundary (ground)
    potential[:, -1] = 0.0   # Right boundary (ground)
    potential[0, :] = 0.0    # Top boundary (ground)
    potential[-1, :] = 0.0   # Bottom boundary (ground)
    
    # Set plate potentials
    potential[lower_plate_start:lower_plate_end, :] = -100.0  # Lower plate
    potential[upper_plate_start:upper_plate_end, :] = 100.0   # Upper plate
    
    # Create mask for fixed potential points
    fixed_mask = np.zeros_like(potential, dtype=bool)
    fixed_mask[:, 0] = True
    fixed_mask[:, -1] = True
    fixed_mask[0, :] = True
    fixed_mask[-1, :] = True
    fixed_mask[lower_plate_start:lower_plate_end, :] = True
    fixed_mask[upper_plate_start:upper_plate_end, :] = True
    
    # SOR iteration
    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if not fixed_mask[i, j]:
                    old_val = potential[i, j]
                    # Update using SOR scheme
                    potential[i, j] = (1-omega)*old_val + omega*0.25*(
                        potential[i-1, j] + potential[i+1, j] + 
                        potential[i, j-1] + potential[i, j+1]
                    )
                    max_diff = max(max_diff, abs(potential[i, j] - old_val))
        
        # Check convergence
        if max_diff < tolerance:
            break
    
    return potential

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation.
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution
    """
    # Calculate Laplacian using finite differences
    laplacian = np.zeros_like(potential_grid)
    
    # Central difference scheme
    laplacian[1:-1, 1:-1] = (
        (potential_grid[1:-1, 2:] - 2*potential_grid[1:-1, 1:-1] + potential_grid[1:-1, :-2]) / dx**2 +
        (potential_grid[2:, 1:-1] - 2*potential_grid[1:-1, 1:-1] + potential_grid[:-2, 1:-1]) / dy**2
    )
    
    # Apply Poisson equation
    charge_density = -laplacian / (4 * np.pi)
    
    return charge_density

def plot_results(potential, charge_density, x_coords, y_coords):
    """
    Create visualization of potential and charge density distributions.
    
    Args:
        potential (np.ndarray): 2D electric potential distribution
        charge_density (np.ndarray): Charge density distribution
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
    """
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Plot potential distribution
    plt.subplot(1, 2, 1)
    X, Y = np.meshgrid(x_coords, y_coords)
    contour = plt.contourf(X, Y, potential, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Electric Potential (V)')
    plt.title('Electric Potential Distribution')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # Plot charge density distribution
    plt.subplot(1, 2, 2)
    charge_contour = plt.contourf(X, Y, charge_density, levels=20, cmap='RdBu')
    plt.colorbar(charge_contour, label='Charge Density (C/m^2)')
    plt.title('Charge Density Distribution')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    nx = 100
    ny = 100
    plate_thickness = 5
    plate_separation = 20
    omega = 1.9
    
    # Solve Laplace equation
    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega)
    
    # Calculate charge density
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # Create coordinate arrays
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    
    # Plot results
    plot_results(potential, charge_density, x_coords, y_coords)
