import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.8, max_iter=10000, tolerance=1e-6):
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
    
    # Set boundary conditions
    potential[:, 0] = 0.0    # Left boundary
    potential[:, -1] = 0.0   # Right boundary
    potential[0, :] = 0.0    # Top boundary
    potential[-1, :] = 0.0   # Bottom boundary
    
    # Calculate plate positions
    top_plate_start = (ny - plate_separation) // 2 - plate_thickness
    top_plate_end = (ny - plate_separation) // 2
    bottom_plate_start = (ny + plate_separation) // 2
    bottom_plate_end = (ny + plate_separation) // 2 + plate_thickness
    
    # Set plate potentials
    potential[top_plate_start:top_plate_end, :] = 100.0
    potential[bottom_plate_start:bottom_plate_end, :] = -100.0
    
    # Create mask for conductor regions (not to be updated)
    conductor_mask = np.zeros_like(potential, dtype=bool)
    conductor_mask[top_plate_start:top_plate_end, :] = True
    conductor_mask[bottom_plate_start:bottom_plate_end, :] = True
    
    # SOR iteration
    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if conductor_mask[i, j]:
                    continue
                
                old_value = potential[i, j]
                new_value = (1-omega)*old_value + omega*0.25*(
                    potential[i+1, j] + potential[i-1, j] + 
                    potential[i, j+1] + potential[i, j-1]
                )
                
                potential[i, j] = new_value
                max_diff = max(max_diff, abs(new_value - old_value))
        
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
    # Calculate the Laplacian using finite differences
    laplacian = np.zeros_like(potential_grid)
    
    # Central difference approximation
    laplacian[1:-1, 1:-1] = (
        (potential_grid[1:-1, 2:] - 2*potential_grid[1:-1, 1:-1] + potential_grid[1:-1, :-2]) / dx**2 +
        (potential_grid[2:, 1:-1] - 2*potential_grid[1:-1, 1:-1] + potential_grid[:-2, 1:-1]) / dy**2
    )
    
    # Calculate charge density
    charge_density = -laplacian / (4 * np.pi)
    
    # Set charge density inside conductors to zero
    charge_density[np.abs(potential_grid) == 100.0] = 0.0
    
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
    plt.figure(figsize=(14, 6))
    
    # Plot potential distribution
    plt.subplot(1, 2, 1)
    contour = plt.contourf(x_coords, y_coords, potential, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Electric Potential (V)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Electric Potential Distribution')
    
    # Plot charge density distribution
    plt.subplot(1, 2, 2)
    charge_contour = plt.contourf(x_coords, y_coords, charge_density, levels=20, cmap='RdBu_r')
    plt.colorbar(charge_contour, label='Charge Density (C/m$^2$)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Charge Density Distribution')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    nx = ny = 100
    plate_thickness = 5
    plate_separation = 30
    omega = 1.8
    
    # Solve for potential
    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega)
    
    # Create coordinate arrays
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    
    # Calculate charge density
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # Plot results
    plot_results(potential, charge_density, x_coords, y_coords)
