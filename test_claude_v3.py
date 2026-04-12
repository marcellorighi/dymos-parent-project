import openmdao.api as om
import dymos as dm
import numpy as np

class DroneODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('avoid_points', types=list)
        self.options.declare('penalty_strength', types=float, default=100.0)
        self.options.declare('penalty_radius', types=float, default=5.0)
    
    def setup(self):
        nn = self.options['num_nodes']
        
        # States
        self.add_input('x', shape=(nn,), units='m')
        self.add_input('y', shape=(nn,), units='m')
        self.add_input('z', shape=(nn,), units='m')
        self.add_input('vx', shape=(nn,), units='m/s')
        self.add_input('vy', shape=(nn,), units='m/s')
        self.add_input('vz', shape=(nn,), units='m/s')
        
        # Controls
        self.add_input('ax', shape=(nn,), units='m/s**2')
        self.add_input('ay', shape=(nn,), units='m/s**2')
        self.add_input('az', shape=(nn,), units='m/s**2')
        
        # State rates
        self.add_output('x_dot', shape=(nn,), units='m/s')
        self.add_output('y_dot', shape=(nn,), units='m/s')
        self.add_output('z_dot', shape=(nn,), units='m/s')
        self.add_output('vx_dot', shape=(nn,), units='m/s**2')
        self.add_output('vy_dot', shape=(nn,), units='m/s**2')
        self.add_output('vz_dot', shape=(nn,), units='m/s**2')
        
        # Velocity squared
        self.add_output('v_squared', shape=(nn,), units='m**2/s**2')
        
        # ADD: Obstacle penalty (instantaneous)
        self.add_output('obstacle_penalty', shape=(nn,), units=None)
        
        # Distance for information (optional, not used in constraints)
        n_avoid = len(self.options['avoid_points'])
        self.add_output('dist_to_avoid', shape=(nn, n_avoid), units='m')
    
    def compute(self, inputs, outputs):
        # ODEs (unchanged)
        outputs['x_dot'] = inputs['vx']
        outputs['y_dot'] = inputs['vy']
        outputs['z_dot'] = inputs['vz']
        outputs['vx_dot'] = inputs['ax']
        outputs['vy_dot'] = inputs['ay']
        outputs['vz_dot'] = inputs['az']
        
        # Velocity squared (unchanged)
        outputs['v_squared'] = (inputs['vx']**2 + 
                               inputs['vy']**2 + 
                               inputs['vz']**2)
        
        # Compute obstacle penalty
        avoid_points = self.options['avoid_points']
        penalty_strength = self.options['penalty_strength']
        penalty_radius = self.options['penalty_radius']
        
        total_penalty = np.zeros(self.options['num_nodes'])
        
        for i, (ax, ay, az) in enumerate(avoid_points):
            # Distance to this obstacle
            dx = inputs['x'] - ax
            dy = inputs['y'] - ay
            dz = inputs['z'] - az
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Store distance (for plotting/debugging)
            outputs['dist_to_avoid'][:, i] = dist
            
            # Penalty function: exponential decay with distance
            # Penalty is high when close, zero when far
            # Option 1: Exponential penalty
            penalty = penalty_strength * np.exp(-dist / penalty_radius)
            
            # Option 2: Inverse distance penalty (alternative)
            # penalty = penalty_strength * np.maximum(0, 1 - dist / penalty_radius)**2
            
            # Option 3: Gaussian penalty (alternative)
            # penalty = penalty_strength * np.exp(-0.5 * (dist / penalty_radius)**2)
            
            total_penalty += penalty
        
        outputs['obstacle_penalty'] = total_penalty


# Create problem
p = om.Problem()
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'IPOPT'
p.driver.opt_settings['max_iter'] = 500
p.driver.opt_settings['print_level'] = 5

traj = dm.Trajectory()
p.model.add_subsystem('traj', traj)

# Define obstacles
avoid_points = [(10, 10, 4), (20, 15, 8), (30, 5, 10)]

# Create phase with penalty parameters
phase = dm.Phase(
    ode_class=DroneODE,
    ode_init_kwargs={
        'avoid_points': avoid_points,
        'penalty_strength': 10000.0,  # Tune this: higher = stronger avoidance
        'penalty_radius': 5.0       # Tune this: smaller = sharper penalty
    },
    transcription=dm.GaussLobatto(num_segments=10, order=3)
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
phase.add_path_constraint('v_squared', upper=25.0, units='m**2/s**2', ref=25.0)

# REMOVE hard obstacle constraints:
# phase.add_path_constraint('dist_to_avoid', lower=2.0, units='m')  # DELETE THIS

# Add timeseries output to integrate penalty over trajectory
phase.add_timeseries_output('obstacle_penalty')

# OPTION A: Minimize weighted combination of time + penalty
# This requires creating a combined objective
phase.add_objective('time', loc='final', ref=30.0)

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

# Setup
p.setup()

# Initial values
p.set_val('traj.phase0.t_initial', 0.0)
p.set_val('traj.phase0.t_duration', 25.0)

p.set_val('traj.phase0.states:x', phase.interp('x', [0, 50]))
p.set_val('traj.phase0.states:y', phase.interp('y', [0, 50]))
p.set_val('traj.phase0.states:z', phase.interp('z', [0, 20]))

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

# Plot trajectory
fig = plt.figure(figsize=(12, 5))

# 3D trajectory
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x, y, z, 'b-', linewidth=2, label='Drone path')

# Plot avoid points
for i, (ax_pos, ay_pos, az_pos) in enumerate(avoid_points):
    ax1.scatter(ax_pos, ay_pos, az_pos, c='red', s=200, marker='X', 
                label=f'Obstacle {i+1}' if i == 0 else '')

ax1.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
ax1.scatter(x[-1], y[-1], z[-1], c='blue', s=100, marker='s', label='End')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.legend()
ax1.set_title('3D Trajectory')

# Velocity magnitude over time
ax2 = fig.add_subplot(122)
v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
ax2.plot(t, v_mag, 'b-', linewidth=2)
ax2.axhline(y=5.0, color='r', linestyle='--', label='Max velocity')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity magnitude (m/s)')
ax2.legend()
ax2.set_title('Velocity Profile')
ax2.grid(True)

plt.tight_layout()
plt.show()

# CORRECTED print statements
print(f"\\nOptimal flight time: {t[-1].item():.2f} seconds")
print(f"Final position: ({x[-1].item():.2f}, {y[-1].item():.2f}, {z[-1].item():.2f})")




