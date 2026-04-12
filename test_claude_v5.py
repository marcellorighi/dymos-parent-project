import openmdao.api as om
import dymos as dm
import numpy as np

class DroneODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('avoid_points', types=list)
        self.options.declare('penalty_strength', types=float, default=100.0)
        self.options.declare('penalty_radius', types=float, default=3.0)
    
    def setup(self):
        nn = self.options['num_nodes']
        
        # Inputs - States
        self.add_input('x', shape=(nn,), units='m')
        self.add_input('y', shape=(nn,), units='m')
        self.add_input('z', shape=(nn,), units='m')
        self.add_input('vx', shape=(nn,), units='m/s')
        self.add_input('vy', shape=(nn,), units='m/s')
        self.add_input('vz', shape=(nn,), units='m/s')
        
        # Inputs - Controls
        self.add_input('ax', shape=(nn,), units='m/s**2')
        self.add_input('ay', shape=(nn,), units='m/s**2')
        self.add_input('az', shape=(nn,), units='m/s**2')
        
        # Outputs - State rates
        self.add_output('x_dot', shape=(nn,), units='m/s')
        self.add_output('y_dot', shape=(nn,), units='m/s')
        self.add_output('z_dot', shape=(nn,), units='m/s')
        self.add_output('vx_dot', shape=(nn,), units='m/s**2')
        self.add_output('vy_dot', shape=(nn,), units='m/s**2')
        self.add_output('vz_dot', shape=(nn,), units='m/s**2')
        
        # Outputs - Constraints
        self.add_output('v_squared', shape=(nn,), units='m**2/s**2')
        self.add_output('obstacle_penalty', shape=(nn,), units=None)  # Fixed units
        
        n_avoid = len(self.options['avoid_points'])
        self.add_output('dist_to_avoid', shape=(nn, n_avoid), units='m')

    def setup_partials(self):  # ← INDENTED (inside class)
        """Declare partial derivatives (Jacobian structure)"""
        nn = self.options['num_nodes']
        n_avoid = len(self.options['avoid_points'])
        
        # Create index arrays for diagonal Jacobians
        ar = np.arange(nn)
        
        # Partials for simple ODEs (constant = 1.0)
        self.declare_partials('x_dot', 'vx', rows=ar, cols=ar, val=1.0)
        self.declare_partials('y_dot', 'vy', rows=ar, cols=ar, val=1.0)
        self.declare_partials('z_dot', 'vz', rows=ar, cols=ar, val=1.0)
        self.declare_partials('vx_dot', 'ax', rows=ar, cols=ar, val=1.0)
        self.declare_partials('vy_dot', 'ay', rows=ar, cols=ar, val=1.0)
        self.declare_partials('vz_dot', 'az', rows=ar, cols=ar, val=1.0)
        
        # Partials for v_squared = vx^2 + vy^2 + vz^2
        self.declare_partials('v_squared', 'vx', rows=ar, cols=ar)
        self.declare_partials('v_squared', 'vy', rows=ar, cols=ar)
        self.declare_partials('v_squared', 'vz', rows=ar, cols=ar)
        
        # Partials for dist_to_avoid (2D output)
        all_rows = []
        all_cols = []
        
        for i in range(n_avoid):
            rows_i = np.arange(nn) * n_avoid + i
            cols_i = np.arange(nn)
            all_rows.append(rows_i)
            all_cols.append(cols_i)
        
        all_rows = np.concatenate(all_rows)
        all_cols = np.concatenate(all_cols)
        
        self.declare_partials('dist_to_avoid', 'x', rows=all_rows, cols=all_cols)
        self.declare_partials('dist_to_avoid', 'y', rows=all_rows, cols=all_cols)
        self.declare_partials('dist_to_avoid', 'z', rows=all_rows, cols=all_cols)
        
        # Partials for obstacle_penalty
        self.declare_partials('obstacle_penalty', 'x', rows=ar, cols=ar)
        self.declare_partials('obstacle_penalty', 'y', rows=ar, cols=ar)
        self.declare_partials('obstacle_penalty', 'z', rows=ar, cols=ar)
    
    def compute(self, inputs, outputs):  # ← INDENTED (inside class)
        """Compute outputs from inputs"""
        
        # State derivatives (ODEs)
        outputs['x_dot'] = inputs['vx']
        outputs['y_dot'] = inputs['vy']
        outputs['z_dot'] = inputs['vz']
        outputs['vx_dot'] = inputs['ax']
        outputs['vy_dot'] = inputs['ay']
        outputs['vz_dot'] = inputs['az']
        
        # Velocity squared
        outputs['v_squared'] = (inputs['vx']**2 + 
                               inputs['vy']**2 + 
                               inputs['vz']**2)
        
        # Distance to obstacles and penalty
        avoid_points = self.options['avoid_points']
        penalty_strength = self.options['penalty_strength']
        penalty_radius = self.options['penalty_radius']
        
        total_penalty = np.zeros(self.options['num_nodes'])
        
        for i, (ax, ay, az) in enumerate(avoid_points):
            dx = inputs['x'] - ax
            dy = inputs['y'] - ay
            dz = inputs['z'] - az
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            outputs['dist_to_avoid'][:, i] = dist
            
            # Penalty function
            penalty = penalty_strength * np.exp(-dist / penalty_radius)
            total_penalty += penalty
        
        outputs['obstacle_penalty'] = total_penalty

    def compute_partials(self, inputs, partials):  # ← INDENTED (inside class)
        """Compute partial derivatives (Jacobians)"""
        
        # Partials for v_squared = vx^2 + vy^2 + vz^2
        partials['v_squared', 'vx'] = 2.0 * inputs['vx']
        partials['v_squared', 'vy'] = 2.0 * inputs['vy']
        partials['v_squared', 'vz'] = 2.0 * inputs['vz']
        
        # Partials for dist_to_avoid and obstacle_penalty
        avoid_points = self.options['avoid_points']
        penalty_strength = self.options['penalty_strength']
        penalty_radius = self.options['penalty_radius']
        nn = self.options['num_nodes']
        n_avoid = len(avoid_points)
        
        # Initialize arrays for flattened dist_to_avoid partials
        ddist_dx_flat = np.zeros(nn * n_avoid)
        ddist_dy_flat = np.zeros(nn * n_avoid)
        ddist_dz_flat = np.zeros(nn * n_avoid)
        
        # Initialize penalty gradients
        dpenalty_dx = np.zeros(nn)
        dpenalty_dy = np.zeros(nn)
        dpenalty_dz = np.zeros(nn)
        
        for i, (ax, ay, az) in enumerate(avoid_points):
            dx = inputs['x'] - ax
            dy = inputs['y'] - ay
            dz = inputs['z'] - az
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Avoid division by zero
            dist_safe = np.maximum(dist, 1e-10)
            
            # ∂dist/∂x = dx/dist, etc.
            ddist_dx = dx / dist_safe
            ddist_dy = dy / dist_safe
            ddist_dz = dz / dist_safe
            
            # Store in flattened array
            flat_indices = np.arange(nn) * n_avoid + i
            ddist_dx_flat[flat_indices] = ddist_dx
            ddist_dy_flat[flat_indices] = ddist_dy
            ddist_dz_flat[flat_indices] = ddist_dz
            
            # Penalty derivatives
            penalty_exp = penalty_strength * np.exp(-dist / penalty_radius)
            factor = -penalty_exp / penalty_radius
            
            dpenalty_dx += factor * ddist_dx
            dpenalty_dy += factor * ddist_dy
            dpenalty_dz += factor * ddist_dz
        
        # Assign flattened partials
        partials['dist_to_avoid', 'x'] = ddist_dx_flat
        partials['dist_to_avoid', 'y'] = ddist_dy_flat
        partials['dist_to_avoid', 'z'] = ddist_dz_flat
        
        partials['obstacle_penalty', 'x'] = dpenalty_dx
        partials['obstacle_penalty', 'y'] = dpenalty_dy
        partials['obstacle_penalty', 'z'] = dpenalty_dz


    
# Create problem
p = om.Problem()
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'IPOPT'
p.driver.opt_settings['max_iter'] = 600
p.driver.opt_settings['print_level'] = 5

traj = dm.Trajectory()
p.model.add_subsystem('traj', traj)

# Define obstacles
# avoid_points = [(10, 10, 4), (20, 15, 8), (30, 5, 0), (48,48,19.2) ]
avoid_points = [(10, 10, 0), (20, 15, 0), (30, 5, 0), (48,48,0) ]

# Create phase with penalty parameters
phase = dm.Phase(
    ode_class=DroneODE,
    ode_init_kwargs={
        'avoid_points': avoid_points,
        'penalty_strength': 100.0,  # Tune this: higher = stronger avoidance
        'penalty_radius': 5.0       # Tune this: smaller = sharper penalty
    },
    transcription=dm.GaussLobatto(num_segments=30, order=5)
)
traj.add_phase('phase0', phase)

phase.set_time_options(fix_initial=True, fix_duration=False, 
                       duration_bounds=(10.0, 100.0))

# States with scaling
phase.add_state('x', fix_initial=True, fix_final=True, 
                rate_source='x_dot', units='m',
                ref=50.0, defect_ref=10.0)

phase.add_state('y', fix_initial=True, fix_final=True,
                rate_source='y_dot', units='m',
                ref=50.0, defect_ref=10.0)

phase.add_state('z', fix_initial=True, fix_final=True,
                rate_source='z_dot', units='m',
                ref=20.0, defect_ref=5.0)

phase.add_state('vx', fix_initial=False, fix_final=False,
                rate_source='vx_dot', units='m/s',
                lower=-5, upper=5, ref=3.0, defect_ref=1.0)

phase.add_state('vy', fix_initial=False, fix_final=False,
                rate_source='vy_dot', units='m/s',
                lower=-5, upper=5, ref=3.0, defect_ref=1.0)

phase.add_state('vz', fix_initial=False, fix_final=False,
                rate_source='vz_dot', units='m/s',
                lower=-5, upper=5, ref=2.0, defect_ref=1.0)
# Controls
phase.add_control('ax', lower=-5, upper=5, units='m/s**2', ref=3.0)
phase.add_control('ay', lower=-5, upper=5, units='m/s**2', ref=3.0)
phase.add_control('az', lower=-5, upper=5, units='m/s**2', ref=3.0)

# Keep velocity constraint (this is feasible)
# phase.add_path_constraint('v_squared', upper=25.0, units='m**2/s**2', ref=25.0)

# REMOVE hard obstacle constraints:
# phase.add_path_constraint('dist_to_avoid', lower=2.0, units='m')  # DELETE THIS

# Add timeseries output to integrate penalty over trajectory
phase.add_timeseries_output('obstacle_penalty')


# OPTION A: Minimize weighted combination of time + penalty
# This requires creating a combined objective


# To add penalty to objective, we need to integrate it over time
# Add this component to the trajectory level:

class PenaltyIntegrator(om.ExplicitComponent):
    """Integrate penalty over the trajectory"""
    def setup(self):
        self.add_input('time', shape_by_conn=True, units='s')
        self.add_input('penalty', shape_by_conn=True)
        self.add_output('total_penalty', units='s')  # Integral of penalty over time
    
    def compute(self, inputs, outputs):
        # Trapezoidal integration
        time = inputs['time'].flatten()
        penalty = inputs['penalty'].flatten()
        outputs['total_penalty'] = np.trapezoid(penalty, time)

# Add integrator
p.model.add_subsystem('penalty_integrator', PenaltyIntegrator())
p.model.connect('traj.phase0.timeseries.time', 'penalty_integrator.time')
p.model.connect('traj.phase0.timeseries.obstacle_penalty', 'penalty_integrator.penalty')

# First, add state to integrate penalty over time
phase.add_state('penalty_integral', 
                rate_source='obstacle_penalty',
                fix_initial=True,
                fix_final=False,
                units=None,
                ref=1000.0)

# Create combined objective component
class CombinedObjective(om.ExplicitComponent):
    def setup(self):
        self.add_input('time_final', units='s')
        self.add_input('penalty_final', units=None)
        self.add_output('J', units='s')
        self.declare_partials('*', '*')
    
    def compute(self, inputs, outputs):
        # Minimize time + weighted penalty
        penalty_weight = 0.5  # Tune this!
        outputs['J'] = inputs['time_final'] + penalty_weight * inputs['penalty_final']
    
    def compute_partials(self, inputs, partials):
        partials['J', 'time_final'] = 1.0
        partials['J', 'penalty_final'] = 0.5

# Add the component after creating the problem
p.model.add_subsystem('combined_obj', CombinedObjective())

# After p.setup(), connect the values
p.model.connect('traj.phase0.timeseries.time', 
                'combined_obj.time_final', src_indices=[-1])
p.model.connect('traj.phase0.timeseries.penalty_integral', 
                'combined_obj.penalty_final', src_indices=[-1])

# Minimize combined objective
p.model.add_objective('combined_obj.J')

# Setup
p.setup()

# Don't forget to initialize penalty integral to zero
# Initial values
p.set_val('traj.phase0.states:penalty_integral', 0.0)








p.set_val('traj.phase0.t_initial', 0.0)
p.set_val('traj.phase0.t_duration', 25.0)

p.set_val('traj.phase0.states:x', phase.interp('x', [0, 50]))
p.set_val('traj.phase0.states:y', phase.interp('y', [0, 50]))
p.set_val('traj.phase0.states:z', phase.interp('z', [0, 00]))

p.set_val('traj.phase0.states:vx', phase.interp('vx', [2, 2]))
p.set_val('traj.phase0.states:vy', phase.interp('vy', [2, 2]))
p.set_val('traj.phase0.states:vz', phase.interp('vz', [0.4, 0.4]))

# Run
p.run_driver()

# Check results
print(f"\\nOptimal time: {p.get_val('traj.phase0.timeseries.time')[-1].item():.2f} s")
print(f"Total penalty accumulated: {p.get_val('penalty_integrator.total_penalty').item():.2f}")



# Access solution directly (no database needed)
import matplotlib.pyplot as plt

# Get the simulation results
sim_out = traj.simulate()

# Get time values
t = p.get_val('traj.phase0.timeseries.time')

# Get states
x = p.get_val('traj.phase0.timeseries.x')
y = p.get_val('traj.phase0.timeseries.y')
z = p.get_val('traj.phase0.timeseries.z')

# Get velocities
vx = p.get_val('traj.phase0.timeseries.vx')
vy = p.get_val('traj.phase0.timeseries.vy')
vz = p.get_val('traj.phase0.timeseries.vz')

# Get controls (accelerations)
# Get controls from phase (NOT from timeseries)
ax = p.get_val('traj.phase0.control_values:ax')
ay = p.get_val('traj.phase0.control_values:ay')
az = p.get_val('traj.phase0.control_values:az')

print("AX SHAPE!!!",ax.shape)

# Compute magnitudes
v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
a_mag = np.sqrt(ax**2 + ay**2 + az**2)



# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# 1. 3D Trajectory
from mpl_toolkits.mplot3d import Axes3D
ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')
ax1.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
ax1.scatter(x[-1], y[-1], z[-1], c='blue', s=100, marker='s', label='End')

# Plot obstacle(s)
for obs_x, obs_y, obs_z in avoid_points:
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    obs_radius = 4.0  # Visualization radius
    ox = obs_x + obs_radius * np.outer(np.cos(u), np.sin(v))
    oy = obs_y + obs_radius * np.outer(np.sin(u), np.sin(v))
    oz = obs_z + obs_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(ox, oy, oz, color='red', alpha=0.3)

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.legend()
ax1.set_title('3D Trajectory')

plt.tight_layout()
plt.show()

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# 1. 3D Trajectory
from mpl_toolkits.mplot3d import Axes3D
ax1 = plt.subplot(3, 2, 1, projection='3d')
ax1.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')
ax1.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
ax1.scatter(x[-1], y[-1], z[-1], c='blue', s=100, marker='s', label='End')

# Plot obstacle(s)
for obs_x, obs_y, obs_z in avoid_points:
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    obs_radius = 4.0  # Visualization radius
    ox = obs_x + obs_radius * np.outer(np.cos(u), np.sin(v))
    oy = obs_y + obs_radius * np.outer(np.sin(u), np.sin(v))
    oz = obs_z + obs_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(ox, oy, oz, color='red', alpha=0.3)

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.legend()
ax1.set_title('3D Trajectory')

# 2. Velocity Magnitude
ax2 = plt.subplot(3, 2, 2)
ax2.plot(t, v_mag, 'b-', linewidth=2, label='Velocity magnitude')
ax2.axhline(y=5.0, color='r', linestyle='--', linewidth=1, label='Max velocity (5 m/s)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_title('Velocity Magnitude')

# 3. Velocity Components
ax3 = plt.subplot(3, 2, 3)
ax3.plot(t, vx, 'r-', linewidth=2, label='vx')
ax3.plot(t, vy, 'g-', linewidth=2, label='vy')
ax3.plot(t, vz, 'b-', linewidth=2, label='vz')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Velocity (m/s)')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_title('Velocity Components')

# 4. Acceleration Magnitude
ax4 = plt.subplot(3, 2, 4)
ax4.plot(t, a_mag, 'r-', linewidth=2, label='Acceleration magnitude')
ax4.axhline(y=3.0, color='k', linestyle='--', linewidth=1, label='Max accel (3 m/s²)')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Acceleration (m/s²)')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_title('Acceleration Magnitude (Control)')

# 5. Acceleration Components
ax5 = plt.subplot(3, 2, 5)
ax5.plot(t, ax, 'r-', linewidth=2, label='ax')
ax5.plot(t, ay, 'g-', linewidth=2, label='ay')
ax5.plot(t, az, 'b-', linewidth=2, label='az')
ax5.axhline(y=3.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax5.axhline(y=-3.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Acceleration (m/s²)')
ax5.grid(True, alpha=0.3)
ax5.legend()
ax5.set_title('Acceleration Components (Controls)')

# 6. Distance to Obstacles
# ax6 = plt.subplot(3, 2, 6)
# dists = p.get_val('traj.phase0.timeseries.dist_to_avoid')
# for i in range(dists.shape[1]):  # For each obstacle
#     ax6.plot(t, dists[:, i], linewidth=2, label=f'Obstacle {i+1}')
# ax6.axhline(y=4.0, color='r', linestyle='--', linewidth=1, label='Min distance constraint')
# ax6.set_xlabel('Time (s)')
# ax6.set_ylabel('Distance (m)')
# ax6.grid(True, alpha=0.3)
# ax6.legend()
# ax6.set_title('Distance to Obstacles')

plt.tight_layout()
plt.savefig('drone_trajectory_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Print statistics
print(f"\\n{'='*60}")
print(f"TRAJECTORY STATISTICS")
print(f"{'='*60}")
print(f"Total time: {t[-1].item():.2f} s")
print(f"\\nVelocity:")
print(f"  Mean magnitude: {v_mag.mean():.2f} m/s")
print(f"  Max magnitude: {v_mag.max():.2f} m/s")
print(f"  Min magnitude: {v_mag.min():.2f} m/s")
print(f"\\nAcceleration:")
print(f"  Mean magnitude: {a_mag.mean():.2f} m/s²")
print(f"  Max magnitude: {a_mag.max():.2f} m/s²")
print(f"\\nObstacle avoidance:")
# print(f"  Minimum distance: {dists.min():.2f} m")
# print(f"  Constraint satisfied: {'YES ✓' if dists.min() >= 4.0 else 'NO ✗'}")
print(f"{'='*60}\\n")




# CORRECTED print statements
#print(f"\\nOptimal flight time: {t[-1].item():.2f} seconds")
#print(f"Final position: ({x[-1].item():.2f}, {y[-1].item():.2f}, {z[-1].item():.2f})")




