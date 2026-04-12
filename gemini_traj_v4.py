import openmdao.api as om
import dymos as dm
import numpy as np
import random
from scipy.interpolate import RegularGridInterpolator

# --- Step 1: Pre-generate a "Frozen" Turbulence Field ---
# This would ideally follow the Dryden scales, but here is a functional grid:
grid_x = np.linspace(0, 500, 10)
grid_y = np.linspace(0, 500, 10)
grid_z = np.linspace(0, 50, 5)

# Create a random (but smoothed) field for wx, wy, wz
# In a real Dryden model, you'd use a Filtered White Noise process here.
data_wx = np.random.randn(10, 10, 5) * 3.0  # 2 m/s intensity
data_wy = np.random.randn(10, 10, 5) * 3.0  # 2 m/s intensity
data_wz = np.random.randn(10, 10, 5) * 1.0  # 2 m/s intensity

# Create the Interpolator (Standard for Dymos ODEs)
get_wind_x = RegularGridInterpolator((grid_x, grid_y, grid_z), data_wx, method='linear')
get_wind_y = RegularGridInterpolator((grid_x, grid_y, grid_z), data_wy, method='linear')
get_wind_z = RegularGridInterpolator((grid_x, grid_y, grid_z), data_wz, method='linear')

class DroneODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('avoid_points', types=list)
        self.options.declare('penalty_strength', types=float, default=100.0)
        self.options.declare('v_ref', default=5.0)  # Wind speed at ref height
        self.options.declare('z_ref', default=20.0) # Reference height

    def setup(self):
        nn = self.options['num_nodes']
        # Inputs: States & Controls
        for var in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']:
            self.add_input(var, shape=(nn,), units=None)

        # Outputs: Rates & Instantaneous Penalty
        for var in ['x_dot', 'y_dot', 'z_dot', 'vx_dot', 'vy_dot', 'vz_dot', 'inst_penalty', 'acc_mag2', 'energy', 'wind_x', 'wind_y', 'wind_z']:
            self.add_output(var, shape=(nn,), units=None)

        # Use finite difference for partials to keep the script simple
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']
        v_ref = self.options['v_ref']
        z_ref = self.options['z_ref']

        # Ensure coordinates stay within the lookup table bounds
        # (This is CRITICAL for robustness)
        x_c = np.clip(x, 0, 500)
        y_c = np.clip(y, 0, 500)
        z_c = np.clip(z, 0, 50)

        # Step 2: Query the Turbulence Field
        # We combine the coordinates into a (N, 3) array for the interpolator
        points = np.stack((x_c, y_c, z_c), axis=-1)
        
        # Local wind from the "Frozen" field
        # outputs['wind_x'] = get_wind_x(points)
        w_x = get_wind_x(points)
        w_y = get_wind_y(points)
        w_z = get_wind_z(points)

        # 1. Spatially Varying Wind (Logarithmic or Power Law Shear)
        # Wind is stronger at higher altitudes, zero at ground.
        # We cap z at a small positive value to avoid log(0)
        # z_safe = np.maximum(z, 0.1)
        # wind_speed = v_ref * (z_safe / z_ref)**0.15 # 1/7th power law
        
        # Assume wind is blowing primarily in the X direction
        # w_x = wind_speed 
        # w_y = np.zeros_like(z)
        # w_z = np.zeros_like(z)
        
        outputs['wind_x'] = w_x
        outputs['wind_y'] = w_y
        outputs['wind_z'] = w_z

        # Kinematics
        # 2. Kinematics: Ground Velocity = Air Velocity + Wind
        # Here, vx, vy, vz are treated as the drone's velocity relative to AIR
        outputs['x_dot'] = inputs['vx'] + w_x
        outputs['y_dot'] = inputs['vy'] + w_y
        outputs['z_dot'] = inputs['vz'] + w_z

        # 3. Dynamics: Accelerations change the Air Velocity
        outputs['vx_dot'] = inputs['ax']
        outputs['vy_dot'] = inputs['ay']
        outputs['vz_dot'] = inputs['az']

        # outputs['x_dot'], outputs['y_dot'], outputs['z_dot'] = inputs['vx'], inputs['vy'], inputs['vz']
        # outputs['vx_dot'], outputs['vy_dot'], outputs['vz_dot'] = inputs['ax'], inputs['ay'], inputs['az']

        # Penalty calculation: 1 / dist^2
        avoid_points = self.options['avoid_points']
        penalty_strength = self.options['penalty_strength']
        total_penalty = np.zeros(self.options['num_nodes'])

        for (px, py, pz) in avoid_points:
            dist2 = (inputs['x'] - px)**2 + (inputs['y'] - py)**2 + (inputs['z'] - pz)**2
            # Avoid division by zero with a small epsilon
            total_penalty += penalty_strength / (dist2 + 1e-4)
        
        outputs['inst_penalty'] = total_penalty
        outputs['acc_mag2'] = inputs['ax']**2 + inputs['ay']**2 + inputs['az']**2
        #v_mag3 = inputs['vx']**3 + inputs['vy']**3 + inputs['vz']**3
        eps = 1e-6 
        v_mag2 = inputs['vx']**2 + inputs['vy']**2 + inputs['vz']**2
        outputs['energy'] = np.power(v_mag2 + eps, 1.5)
        #outputs['energy'] = inputs['vx']**2 + inputs['vy']**2 + inputs['vz']**2

# --- Setup Problem ---
p = om.Problem()
p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
p.driver.opt_settings['print_level'] = 5
# p.driver.opt_settings['delta'] = 1e-1
p.driver.opt_settings['max_iter'] = 800

# Generate Obstacles
avoid_points = [(random.uniform(150, 450), random.uniform(150, 450), 0) for _ in range(80)]

traj = dm.Trajectory()
phase = dm.Phase(ode_class=DroneODE, 
                 ode_init_kwargs={'avoid_points': avoid_points},
                 transcription=dm.GaussLobatto(num_segments=15, order=3))
p.model.add_subsystem('traj', traj)
traj.add_phase('phase0', phase)

# Time, States, and Controls
phase.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=(5, 200))

# States: x, y, z are fixed at start (0,0,0) and end (500,500,10)
#for s in ['x', 'y', 'z']:
for s in ['x', 'y']:
    phase.add_state(s, fix_initial=True, fix_final=True, ref=500.0, rate_source=f'{s}_dot')

phase.add_state('z', fix_initial=True, lower = 0, upper = 50., ref=5.0, fix_final=True, rate_source=f'z_dot')

# Velocities: Fixed at 0 at start/end
for v in ['vx', 'vy', 'vz']:
    phase.add_state(v, fix_initial=True, fix_final=True, rate_source=f'{v}_dot')

phase.add_state('acc_integral', rate_source='acc_mag2', lower = 0, ref=5.0, fix_initial=True)

phase.add_state('energy_spent', 
                rate_source='energy', 
                fix_initial=True, 
                fix_final=False,
                lower = 0, 
                ref=500.0, 
                units=None)

# Accelerations as Controls
for a in ['ax', 'ay', 'az']:
    phase.add_control(a, lower=-8.0, upper=8.0, rate_continuity=True, 
                  rate2_continuity=False)

# INTEGRATE PENALTY: This creates the "Accumulated Reward" automatically
phase.add_state('total_penalty', rate_source='inst_penalty', ref=5000.0, fix_initial=True, fix_final=False)



phase.add_timeseries_output('wind_x')
phase.add_timeseries_output('wind_y')
phase.add_timeseries_output('wind_z')



# Objective: Minimize Time + Penalty_Integral
class ObjectiveComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('time', units=None)
        self.add_input('penalty', units=None)
        self.add_input('acc_integral', units=None)
        self.add_input('energy_final', units=None) # New input
        self.add_output('J')
        self.declare_partials('*', '*', method='fd')
    def compute(self, inputs, outputs):
        # outputs['J'] = inputs['time'] + 20.0 * inputs['penalty']
        outputs['J'] = 0.003 * ( inputs['time'] + 6.0 * inputs['penalty'] + 0.0 * inputs['acc_integral'] + 0.001 * inputs['energy_final'] )
        # outputs['J'] = 0.003 * ( inputs['time'] + 6.0 * inputs['penalty'] + 5.0 * inputs['acc_integral'] + 0.001 * inputs['energy_final'] )
        # outputs['J'] = inputs['time'] + 4.0 * inputs['penalty'] + 0.02 * inputs['energy_final']

p.model.add_subsystem('obj_comp', ObjectiveComp())
p.model.connect('traj.phase0.timeseries.time', 'obj_comp.time', src_indices=[-1])
p.model.connect('traj.phase0.timeseries.total_penalty', 'obj_comp.penalty', src_indices=[-1])
p.model.connect('traj.phase0.timeseries.acc_integral', 'obj_comp.acc_integral', src_indices=[-1])
p.model.connect('traj.phase0.timeseries.energy_spent', 
                'obj_comp.energy_final', src_indices=[-1])
p.model.add_objective('obj_comp.J')

p.setup()

# --- Initial Guesses ---
p.set_val('traj.phase0.t_duration', 30.0)
p.set_val('traj.phase0.states:x', phase.interp('x', [0, 500.]))
p.set_val('traj.phase0.states:y', phase.interp('y', [0, 450.]))
p.set_val('traj.phase0.states:z', phase.interp('z', [0, 10]))
p.set_val('traj.phase0.states:total_penalty', 0.0)
p.set_val('traj.phase0.states:energy_spent', phase.interp('energy_spent', [0, 100]))
p.set_val('traj.phase0.states:acc_integral', phase.interp('energy_spent', [0, 100]))

p.run_driver()

# Extracting the values from the objective component
final_time = p.get_val('obj_comp.time')[0]
final_penalty = p.get_val('obj_comp.penalty')[0]
final_energy = p.get_val('obj_comp.energy_final')[0]
acc_integral = p.get_val('obj_comp.acc_integral')[0]
final_total_J = p.get_val('obj_comp.J')[0]

print(f"\n{'='*30}")
print(f"OPTIMIZATION RESULTS")
print(f"{'='*30}")
print(f"Final Time:         {final_time:.4f} s")
print(f"Obstacle Penalty:   {final_penalty:.4f}")
print(f"Energy Expenditure: {final_energy:.4f}")
print(f"Acceleration integ: {acc_integral:.4f}")
print(f"Total Objective J:  {final_total_J:.4f}")
print(f"{'='*30}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Extract Data ---
t = p.get_val('traj.phase0.timeseries.time')
x = p.get_val('traj.phase0.timeseries.x')
y = p.get_val('traj.phase0.timeseries.y')
z = p.get_val('traj.phase0.timeseries.z')

vx = p.get_val('traj.phase0.timeseries.vx')
vy = p.get_val('traj.phase0.timeseries.vy')
vz = p.get_val('traj.phase0.timeseries.vz')

ax = p.get_val('traj.phase0.timeseries.ax')
ay = p.get_val('traj.phase0.timeseries.ay')
az = p.get_val('traj.phase0.timeseries.az')

# Extract the wind components calculated by the ODE
wx = p.get_val('traj.phase0.timeseries.wind_x')
wy = p.get_val('traj.phase0.timeseries.wind_y')
wz = p.get_val('traj.phase0.timeseries.wind_z')

# --- 2. Plotting ---
fig = plt.figure(figsize=(15, 10))

# Plot 1: 3D Trajectory
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(x, y, z, 'b-', label='Path')
ax1.scatter(x[0], y[0], z[0], color='g', label='Start')
ax1.scatter(x[-1], y[-1], z[-1], color='r', label='End')

# Plot Obstacles (as simple points for clarity)
obs_x, obs_y, obs_z = zip(*avoid_points)
ax1.scatter(obs_x, obs_y, obs_z, color='k', marker='x', alpha=0.5, label='Obstacles')

ax1.set_title("3D Drone Trajectory")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.legend()

# Plot 2: Velocity Components
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t, vx, label='Vx')
ax2.plot(t, vy, label='Vy')
ax2.plot(t, vz, label='Vz')
ax2.set_title("Velocity Components")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("m/s")
ax2.legend()
ax2.grid(True)

# Plot 3: Acceleration (Controls)
ax3 = fig.add_subplot(2, 2, 3)
ax3.step(t, ax, where='post', label='ax')
ax3.step(t, ay, where='post', label='ay')
ax3.step(t, az, where='post', label='az')
ax3.set_title("Acceleration (Controls)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("m/s²")
ax3.legend()
ax3.grid(True)

# Plot 4: Altitude (Z) over time
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(t, z, color='purple')
ax4.set_title("Altitude Profile")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Z (m)")
ax4.grid(True)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot A: The Wind Profile (Physics check)
# This shows how wind changes with height (Z)
ax1.plot(wx, z, 'b-', label='Wind X (Shear)')
ax1.set_xlabel('Wind Speed (m/s)')
ax1.set_ylabel('Altitude Z (m)')
ax1.set_title('Wind Profile vs. Altitude')
ax1.grid(True)
ax1.legend()

# Plot B: Wind Experienced Over Time
# This shows the "Timeline" of the disturbances
ax2.plot(t, wx, label='Wind X')
ax2.plot(t, wy, label='Wind Y')
ax2.plot(t, wz, label='Wind Z')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Wind Speed (m/s)')
ax2.set_title('Wind Experienced during Flight')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()


