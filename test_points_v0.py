import numpy as np
import random

def generate_random_points(n, x_range=(0, 50), y_range=(0, 50), z_range=(0, 0)):
    """
    Generates a list of N random (x, y, z) tuples.
    """
    points = []
    for _ in range(n):
        x = round(random.uniform(*x_range), 2)
        y = round(random.uniform(*y_range), 2)
        z = round(random.uniform(*z_range), 2)
        points.append((x, y, z))
    return points

def generate_grid_points(rows, cols, spacing=10, z=0):
    """
    Generates a list of points in a structured grid pattern.
    """
    points = []
    for r in range(rows):
        for c in range(cols):
            points.append((float(c * spacing), float(r * spacing), float(z)))
    return points

def generate_numpy_array(n):
    """
    Generates points using NumPy (faster for very large datasets).
    Then converts to a list of tuples.
    """
    # Create an Nx3 array
    coords = np.zeros((n, 3))
    coords[:, 0] = np.random.uniform(0, 50, n) # X
    coords[:, 1] = np.random.uniform(0, 50, n) # Y
    coords[:, 2] = 0                           # Z
    
    # Convert to list of tuples to match your requested format
    return [tuple(row) for row in coords]

# --- Examples of use ---

n_points = 10 

# 1. Random Points (like obstacles in a field)
avoid_points_random = generate_random_points(n_points)
print("Random Points:")
print(avoid_points_random)

# 2. Grid Pattern (like a forest or pillars)
avoid_points_grid = generate_grid_points(3, 3, spacing=15)
print("\nGrid Points:")
print(avoid_points_grid)

# 3. Simple List Comprehension (for a line or custom math)
# Example: A diagonal line of 5 points
avoid_points_line = [(i * 10, i * 10, 0) for i in range(1, 6)]
print("\nLine Points:")
print(avoid_points_line)

