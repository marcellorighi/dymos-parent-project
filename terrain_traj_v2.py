import numpy as np
import openmdao.api as om
import dymos as dy
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_urban_terrain(seed = None, grid_size=150, area_dim=500, num_hills=12, num_buildings=0, hill_min_height = 20, hill_max_height=160, hill_min_width = 20., hill_max_width = 320.):
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
    np.random.seed(seed)

    # --- Add Rolling Hills (Wide Gaussians) ---
    for _ in range(num_hills):
        cx, cy = np.random.uniform(-0.5 * area_dim, 0.5 * area_dim, 2)
        height = np.random.uniform(hill_min_height, hill_max_height)  # Moderate height
        width = np.random.uniform(hill_min_width, hill_max_width)   # Wide spread
        # The classic Gaussian "bell curve" formula
        Z += height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * width**2))

    # --- Add 'Buildings' (Steep, Narrow Gaussians) ---
    for _ in range(num_buildings):
        cx, cy = np.random.uniform(-area_dim*0.8, area_dim*0.8, 2)
        height = np.random.uniform(10, 45) # Variable building heights
        width = np.random.uniform(10, 20)   # Narrow width creates "steep" sides
        # Building = Narrow Gaussian
        Z += height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * width**2))

    return x, y, X, Y, Z


class DroneTerrainODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('terrain_interp', types=RegularGridInterpolator)
        self.options.declare('avoid_points', types=list)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States and Controls
        self.add_input('x', shape=(nn,), units='m')
        self.add_input('y', shape=(nn,), units='m')
        self.add_input('z', shape=(nn,), units='m')
        self.add_input('vx', shape=(nn,), units='m/s')
        self.add_input('vy', shape=(nn,), units='m/s')
        self.add_input('vz', shape=(nn,), units='m/s')
        self.add_input('ax', shape=(nn,), units='m/s**2')
        self.add_input('ay', shape=(nn,), units='m/s**2')
        self.add_input('az', shape=(nn,), units='m/s**2')

        # Outputs: Rates (Kinematics)
        self.add_output('x_dot', shape=(nn,), units='m/s')
        self.add_output('y_dot', shape=(nn,), units='m/s')
        self.add_output('z_dot', shape=(nn,), units='m/s')
        self.add_output('vx_dot', shape=(nn,), units='m/s**2')
        self.add_output('vy_dot', shape=(nn,), units='m/s**2')
        self.add_output('vz_dot', shape=(nn,), units='m/s**2')
        
        # Output: Constraint Variable
        self.add_output('clearance', shape=(nn,), units='m')
        self.add_output('avoid_dist2', shape=(nn,), units='m')

        # Complex Step for all partials
        # self.declare_partials('*', '*', method='cs')
        # self.declare_partials('clearance', ['x', 'y', 'z'], method='fd')

        self.declare_partials('x_dot', 'vx', method='cs')
        self.declare_partials('y_dot', 'vy', method='cs')
        self.declare_partials('z_dot', 'vz', method='cs')
        
        self.declare_partials('vx_dot', 'ax', method='cs')
        self.declare_partials('vy_dot', 'ay', method='cs')
        self.declare_partials('vz_dot', 'az', method='cs')

        # Use FD for the terrain and the avoid points
        self.declare_partials('clearance', ['x', 'y', 'z'], method='fd')
        self.declare_partials('avoid_dist2', ['x', 'y', 'z'], method='fd')

    def compute(self, inputs, outputs):
        # 1. Kinematics
        outputs['x_dot'] = inputs['vx']
        outputs['y_dot'] = inputs['vy']
        outputs['z_dot'] = inputs['vz']
        
        # 2. Dynamics (Simple Point Mass for now)
        outputs['vx_dot'] = inputs['ax']
        outputs['vy_dot'] = inputs['ay']
        outputs['vz_dot'] = inputs['az'] - 9.81 # Including gravity

        # 3. Terrain Clearance
        pts = np.column_stack((inputs['x'], inputs['y']))
        z_ground = self.options['terrain_interp'](pts)
        outputs['clearance'] = inputs['z'] - z_ground

        # 4. Avoid points 
        points = self.options['avoid_points']
        dist2 = np.zeros_like(inputs['x']) 
        for px, py, pz in points:
            dist2 += 1. / ( (inputs['x'] - px)**2 + (inputs['y'] - py)**2 + (inputs['z'] - pz)**2 + 1.e-2)
        outputs['avoid_dist2'] = 1.e3 * dist2

# --- SEED --- 
cmd_seed = np.random.randint(0, 100000)

# --- AREA SIZE --- 
margin = 50. 
area_dim = 1200. 

# --- TERRAIN --- 
x_coords, y_coords, X_mesh, Y_mesh, Z = generate_urban_terrain(seed = cmd_seed, area_dim = area_dim + margin) 

# interp = RegularGridInterpolator((x_coords, y_coords), Z, method='cubic')
interp = RegularGridInterpolator((x_coords, y_coords), Z, 
                                 method='cubic', 
                                 bounds_error=False, 
                                 fill_value=None)

# --- AVOID ponts ---  

# 1. Generate your random (x, y) coordinates
num_points = 50
area_limit = area_dim - margin #400
x_rand = np.random.uniform(-area_limit, area_limit, num_points)
y_rand = np.random.uniform(-area_limit, area_limit, num_points)

# 2. Prepare the points for the interpolator
# We stack them into the (N, 2) format Scipy expects
query_pts = np.column_stack((x_rand, y_rand))

# 3. Query the terrain interpolator (interp) to get the heights
z_terrain = interp(query_pts)

# 4. Combine into a list of (x, y, z) tuples
avoid_points = list(zip(x_rand, y_rand, z_terrain))

# --- 2. Create Problem and Trajectory ---
prob = om.Problem(model=om.Group())

prob.driver = om.pyOptSparseDriver(optimizer='IPOPT')
prob.driver.opt_settings['print_level'] = 5
prob.driver.opt_settings['max_iter'] = 400


traj = prob.model.add_subsystem('traj', dy.Trajectory())
phase = traj.add_phase('phase0', dy.Phase(ode_class=DroneTerrainODE, 
                                          ode_init_kwargs={'terrain_interp': interp, 'avoid_points': avoid_points},
                                          transcription=dy.GaussLobatto(num_segments=15, order=3)))

# --- 3. Set Time Options (Objective: Minimize Time) ---
phase.set_time_options(fix_initial=True, duration_bounds=(1, 500))
phase.add_objective('time', loc='final', ref=10.0)

# --- 4. Set State Options ---
# Initial and Final Position: Traveling from (-400, -400, 20) to (400, 400, 20)
phase.add_state('x', fix_initial=True, fix_final=True, 
                #lower=-495, upper=495,  # Keep it on the map!
                lower=-area_dim + 5, upper= area_dim - 5,  # Keep it on the map!
                rate_source='x_dot', units='m', ref=50)

phase.add_state('y', fix_initial=True, fix_final=True, 
                # lower=-495, upper=495, 
                lower=-area_dim + 5, upper= area_dim - 5,  # Keep it on the map!
                rate_source='y_dot', units='m', ref=50)
phase.add_state('z', fix_initial=True, fix_final=True, rate_source='z_dot', units='m', ref=100)
phase.add_path_constraint('z', lower=0, upper=120., ref=100.0)

# Velocities: Start and end at a hover (0 m/s)
phase.add_state('vx', fix_initial=True, fix_final=True, rate_source='vx_dot', units='m/s', ref=10)
phase.add_state('vy', fix_initial=True, fix_final=True, rate_source='vy_dot', units='m/s', ref=10)
phase.add_state('vz', fix_initial=True, fix_final=True, rate_source='vz_dot', units='m/s', ref=10)

# --- 5. Set Controls ---
phase.add_control('ax', lower=-19, upper=19, units='m/s**2', ref=5)
phase.add_control('ay', lower=-19, upper=19, units='m/s**2', ref=5)
phase.add_control('az', lower=-2, upper=20, units='m/s**2', ref=10) # az must fight gravity

# --- 6. Path Constraint: Avoid Terrain ---
phase.add_path_constraint('clearance', lower= 5.0, ref= 5.0)
phase.add_path_constraint('avoid_dist2', upper= 2.000, ref= 2.000) 

# --- 7. Run Optimization ---
prob.setup(check=True)

# Set initial guesses for states
prob.set_val('traj.phase0.t_initial', 0.0)
prob.set_val('traj.phase0.t_duration', 50.0)
prob.set_val('traj.phase0.states:x', phase.interp('x', xs =[0, 0.5, 1], ys= [-area_dim+100, 0., area_dim - 100]))
prob.set_val('traj.phase0.states:y', phase.interp('y', xs =[0, 0.5, 1], ys= [-area_dim + 100, area_dim - 100., area_dim - 100]))

# z_start = interp((-400.,-400.))
# z_end = interp((400.,400.)) 
z_start = interp((-area_dim+100,-area_dim + 100)) 
z_end = interp((area_dim - 100, area_dim - 100))
prob.set_val('traj.phase0.states:z', phase.interp('z', xs = [0., 0.5, 1.0], ys = [z_start + 10., 10., z_end + 10.]))

prob.run_driver()

# --- 8. Post-Processing ---
sim_data = traj.simulate()
# You can use your previous z_map plotting code here to overlay the 'sim_data' trajectory!

# --- 1. Extract Data ---
t = prob.get_val('traj.phase0.timeseries.time')
x = prob.get_val('traj.phase0.timeseries.x')
y = prob.get_val('traj.phase0.timeseries.y')
z = prob.get_val('traj.phase0.timeseries.z')

vx = prob.get_val('traj.phase0.timeseries.vx')
vy = prob.get_val('traj.phase0.timeseries.vy')
vz = prob.get_val('traj.phase0.timeseries.vz')

Ax = prob.get_val('traj.phase0.timeseries.ax')
Ay = prob.get_val('traj.phase0.timeseries.ay')
Az = prob.get_val('traj.phase0.timeseries.az')

# --- terrain map for plotting --- 

# area_dim = 400.
# grid_size = 20 

# xx = np.linspace(-area_dim, area_dim, grid_size)
# yy = np.linspace(-area_dim, area_dim, grid_size)
# X, Y = np.meshgrid(xx, yy, indexing='ij')  # Meshgrid creates 2D arrays
# points = np.column_stack((X.ravel(), Y.ravel()))
# z_values = interp(points)
# Z = z_values.reshape(X.shape)

# --- 2. Plotting the Result ---
# Create the figure and a 3D axes object
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
# - rstride/cstride: Step size (lower is smoother, slower)
# - cmap: Colormap ('terrain', 'gist_earth', 'viridis' are good)
surf = ax.plot_surface(X_mesh, Y_mesh, Z,     
                       rstride=2, cstride=2,
                       cmap='terrain',  # The classic map look
                       edgecolor='none', # Turn off grid lines for smoothness
                       alpha=0.9,       # Slight transparency
                       antialiased=True)

# --- 3. Customizing the Plot ---
# Add a colorbar to interpret the heights
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Elevation (m)')

# drone trajectory 

ax.plot(x, y, z, 'b-', label='Path')
ax.scatter(x[0], y[0], z[0], color='g', label='Start')
ax.scatter(x[-1], y[-1], z[-1], color='r', label='End')

# avoid points
ax.scatter(x_rand, y_rand, z_terrain, color='red', s=20, label='Avoid Points')

# Set labels for axes
ax.set_xlabel('Horizontal distance (X) [m]')
ax.set_ylabel('Horizontal distance (Y) [m]')
ax.set_zlabel('Terrain Elevation (Z_ground) [m]')
ax.set_title('Procedural Urban Terrain: Hills and Buildings (Smoothed for Optimization)')

# Set specific view angle (Elev, Azim)
# This angle helps distinguish "buildings" from "hills"
ax.view_init(elev=30, azim=135)
ax.axis('equal')

# Optional: Tighten the limits if needed
# ax.set_xlim(-500, 500)
# ax.set_ylim(-500, 500)
# ax.set_zlim(0, np.max(z_map) + 20)

# Render the plot
plt.tight_layout()
plt.show() # This might open an interactive window depending on your Python setup


