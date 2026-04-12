import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D toolbox
from scipy.interpolate import RegularGridInterpolator

# --- 1. Procedural Terrain Generator ---
def generate_urban_terrain(grid_size=150, area_dim=500, num_hills=3, num_buildings=10):
    """
    Creates a smooth, continuous terrain with random hills and buildings.
    The output is differentiable (C2), perfect for OpenMDAO/Dymos.
    """
    # Create the 2D grid: From -area_dim to +area_dim
    x = np.linspace(-area_dim, area_dim, grid_size)
    y = np.linspace(-area_dim, area_dim, grid_size)
    X, Y = np.meshgrid(x, y, indexing='ij')  # Meshgrid creates 2D arrays
    Z = np.zeros_like(X)

    # Use a fixed seed for reproducible 'randomness' during testing
    np.random.seed(42)

    # --- Add Rolling Hills (Wide Gaussians) ---
    for _ in range(num_hills):
        cx, cy = np.random.uniform(-area_dim, area_dim, 2)
        height = np.random.uniform(20, 50)  # Moderate height
        width = np.random.uniform(100, 200)   # Wide spread
        # The classic Gaussian "bell curve" formula
        Z += height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * width**2))

    # --- Add 'Buildings' (Steep, Narrow Gaussians) ---
    for _ in range(num_buildings):
        cx, cy = np.random.uniform(-area_dim*0.8, area_dim*0.8, 2)
        height = np.random.uniform(10, 45) # Variable building heights
        width = np.random.uniform(10, 20)   # Narrow width creates "steep" sides
        # Building = Narrow Gaussian
        Z += height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * width**2))

    return X, Y, Z

# Generate the data: X, Y (the grid), Z (the map)
X, Y, z_map = generate_urban_terrain()

# --- 2. Plotting the Result ---
# Create the figure and a 3D axes object
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
# - rstride/cstride: Step size (lower is smoother, slower)
# - cmap: Colormap ('terrain', 'gist_earth', 'viridis' are good)
surf = ax.plot_surface(X, Y, z_map,
                       rstride=2, cstride=2,
                       cmap='terrain',  # The classic map look
                       edgecolor='none', # Turn off grid lines for smoothness
                       alpha=0.9,       # Slight transparency
                       antialiased=True)

# --- 3. Customizing the Plot ---
# Add a colorbar to interpret the heights
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Elevation (m)')

# Set labels for axes
ax.set_xlabel('Horizontal distance (X) [m]')
ax.set_ylabel('Horizontal distance (Y) [m]')
ax.set_zlabel('Terrain Elevation (Z_ground) [m]')
ax.set_title('Procedural Urban Terrain: Hills and Buildings (Smoothed for Optimization)')

# Set specific view angle (Elev, Azim)
# This angle helps distinguish "buildings" from "hills"
ax.view_init(elev=30, azim=135)

# Optional: Tighten the limits if needed
# ax.set_xlim(-500, 500)
# ax.set_ylim(-500, 500)
# ax.set_zlim(0, np.max(z_map) + 20)

# Render the plot
plt.tight_layout()
plt.show() # This might open an interactive window depending on your Python setup

