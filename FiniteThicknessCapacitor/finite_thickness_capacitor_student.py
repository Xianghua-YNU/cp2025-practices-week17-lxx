#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor with 3D Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import laplace

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.85, max_iter=10000, tolerance=1e-6):
    """Solve 2D Laplace equation using SOR method"""
    # Initialize potential grid
    U = np.zeros((ny, nx))
    
    # Calculate plate positions (centered vertically)
    y_center = ny // 2
    upper_start = y_center + plate_separation // 2
    upper_end = upper_start + plate_thickness
    lower_end = y_center - plate_separation // 2
    lower_start = lower_end - plate_thickness
    
    # Set conductor potentials
    U[upper_start:upper_end, :] = 100.0  # Upper plate
    U[lower_start:lower_end, :] = -100.0  # Lower plate
    
    # Boundary conditions
    U[:, 0] = 0.0    # Left boundary
    U[:, -1] = 0.0   # Right boundary
    U[0, :] = 0.0    # Top boundary
    U[-1, :] = 0.0   # Bottom boundary
    
    # SOR iteration
    for _ in range(max_iter):
        max_error = 0.0
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                # Skip conductor regions
                if (upper_start <= i < upper_end) or (lower_start <= i < lower_end):
                    continue
                
                old_val = U[i, j]
                new_val = (1-omega)*old_val + omega*0.25*(
                    U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1]
                )
                U[i,j] = new_val
                max_error = max(max_error, abs(new_val - old_val))
        
        if max_error < tolerance:
            break
    
    return U

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using finite difference Laplacian.
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution
    """
    # Initialize laplacian array
    laplacian = np.zeros_like(potential_grid)
    
    # Central difference approximation (excluding boundaries)
    laplacian[1:-1,1:-1] = (
        (potential_grid[1:-1,2:] - 2*potential_grid[1:-1,1:-1] + potential_grid[1:-1,:-2])/dx**2 +
        (potential_grid[2:,1:-1] - 2*potential_grid[1:-1,1:-1] + potential_grid[:-2,1:-1])/dy**2
    )
    
    # Calculate charge density from Poisson equation
    charge_density = -laplacian / (4 * np.pi)
    
    # Set conductor interior charge density to zero
    charge_density[np.abs(potential_grid) == 100.0] = 0.0
    
    return charge_density

def plot_3d_results(potential, charge_density, x_coords, y_coords):
    """
    Create advanced 3D visualizations with multiple viewing angles.
    """
    fig = plt.figure(figsize=(18, 8))
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # ================
    # 3D Potential Plot
    # ================
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, potential, cmap='viridis',
                          rstride=2, cstride=2, alpha=0.8,
                          linewidth=0, antialiased=True)
    ax1.contour(X, Y, potential, zdir='z', offset=potential.min(), 
               cmap='viridis', linestyles="solid")
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Electric Potential (V)')
    ax1.set_title('3D Potential Distribution', pad=20)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Potential (V)')
    ax1.view_init(elev=30, azim=45)
    
    # =====================
    # 3D Charge Density Plot
    # =====================
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, charge_density, cmap='RdBu_r',
                           rstride=2, cstride=2, alpha=0.8,
                           linewidth=0.5, antialiased=True)
    levels = np.linspace(charge_density.min(), charge_density.max(), 15)
    ax2.contour(X, Y, charge_density, levels=levels, 
               zdir='z', offset=charge_density.min(), 
               colors='k', linewidths=0.5)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_zlabel('Charge Density (C/m²)')
    ax2.set_title('3D Charge Density Distribution', pad=20)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Charge Density (C/m²)')
    ax2.view_init(elev=25, azim=-45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    nx, ny = 120, 100
    plate_thickness = 8
    plate_separation = 40
    omega = 1.88
    
    # Solve Laplace equation
    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega)
    
    # Create coordinate system
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    
    # Calculate charge density
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # Generate 3D visualizations
    plot_3d_results(potential, charge_density, x_coords, y_coords)
