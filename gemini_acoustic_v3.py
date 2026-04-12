import openmdao.api as om
import dymos as dm
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_acoustic_map(points, start_pos=(0,0), end_pos=(500,500)):
    px = [p[0] for p in points]
    py = [p[1] for p in points]
    p_lim = [p[2] for p in points]
    p_weight = [p[3] * 50 for p in points] # Scale size by weight for visibility

    plt.figure(figsize=(10, 8))
    
    # Plot observers
    sc = plt.scatter(px, py, c=p_lim, s=p_weight, cmap='RdYlGn_r', 
                     edgecolors='black', alpha=0.7, label='Observers')
    
    # Plot Start/End
    plt.plot(start_pos[0], start_pos[1], 'b*', markersize=15, label='Start')
    plt.plot(end_pos[0], end_pos[1], 'g*', markersize=15, label='End')

    plt.colorbar(sc, label='Noise Limit (dB)')
    plt.title('Acoustic Sensitivity Map (Circle size = Annoyance Weight)')
    plt.xlabel('X Distance (m)')
    plt.ylabel('Y Distance (m)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


class DrydenField:
    def __init__(self, n_waves=20, seed=42):
        np.random.seed(seed)
        # Define a range of spatial frequencies (rad/m)
        # Dryden peaks usually between 0.01 and 0.5 rad/m
        self.k = np.logspace(-2, 0, n_waves) 
        
        # Random phases for each wave in 3D
        self.phi_x = np.random.uniform(0, 2*np.pi, n_waves)
        self.phi_y = np.random.uniform(0, 2*np.pi, n_waves)
        self.phi_z = np.random.uniform(0, 2*np.pi, n_waves)
        
        # Directions for the wave vectors
        self.dirs = np.random.randn(n_waves, 3)
        self.dirs /= np.linalg.norm(self.dirs, axis=1)[:, None]

    def get_gust(self, x, y, z, sigma=1.5):
        """
        Computes a smooth, differentiable gust at (x, y, z).
        This is compatible with Dymos because it uses pure NumPy math.
        """
        u_gust = np.zeros_like(x)
        
        # Apply the Dryden-like summation
        # Local turbulence intensity often scales with sqrt(z)
        intensity = sigma * (1.0 - np.exp(-z/5.0)) 
        
        for i in range(len(self.k)):
            # Spatial projection: k_vec dot position_vec
            proj = self.k[i] * (self.dirs[i,0]*x + self.dirs[i,1]*y + self.dirs[i,2]*z)
            u_gust += np.sin(proj + self.phi_x[i])
            
        return u_gust * (intensity / np.sqrt(len(self.k)))

# Initialize the field globally so it's "Frozen"
dryden_field = DrydenField()

class DroneODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('mass', default=25.0, desc='Total drone mass in kg')
        self.options.declare('lwa_hover', default=98.0, desc='Sound power level at hover in dB')
        self.options.declare('num_nodes', types=int)
        self.options.declare('avoid_points', types=list)
        self.options.declare('penalty_strength', types=float, default=100.0)
        self.options.declare('v_ref', default=5.0)  # Wind speed at ref height
        self.options.declare('z_ref', default=20.0) # Reference height
        self.options.declare('sensitive_points', types=list) 
        self.options.declare('rho', default=50.0) # KS scaling parameter


    def setup(self):
        nn = self.options['num_nodes']
        # Inputs: States & Controls
        for var in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']:
            self.add_input(var, shape=(nn,), units=None)

        # Outputs: Rates & Instantaneous Penalty
        for var in ['x_dot', 'y_dot', 'z_dot', 'vx_dot', 'vy_dot', 'vz_dot', 'inst_penalty', 'acc_mag2', 'energy', 'wind_x', 'wind_y', 'wind_z', 'ks_norm_Lp', 'ks_annoyance', 'ks_rate']:
            self.add_output(var, shape=(nn,), units=None)

        # Use finite difference for partials to keep the script simple
        self.declare_partials(of='*', wrt='*', method='cs')
        # print(f"DEBUG: d(Annoyance)/dx = {partials['ks_norm_Lp', 'x'][0]}")

    def compute(self, inputs, outputs):
        dtype = inputs['x'].dtype

        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        ax, ay, az = inputs['ax'], inputs['ay'], inputs['az']

        # 1. Calculate Total Thrust Magnitude
        # This accounts for tilting to move sideways and climbing

        # REMOVE 
        g = 9.81 

        mass = self.options['mass']
        lwa_hover = self.options['lwa_hover']

        thrust_mag = mass * np.sqrt(ax**2 + ay**2 + (az + g)**2)
        
        # 2. Calculate Instantaneous Sound Power Level (Lwa)
        # We use a small epsilon to avoid log of zero if thrust was 0
        ratio = thrust_mag / (mass * g)
        Lwa = lwa_hover + 25 * np.log10(np.maximum(ratio, 1e-3))

        # addition for acoustics 
        # Lwa = outputs['Lwa']
        points = self.options['sensitive_points']
        rho = self.options['rho']
        
        # Lists to store metrics for all observers at all time-nodes
        all_norm_Lp = []
        all_annoyance = []

        for px, py, pz, limit_Lp, weight in points:
        # # avoid_points = self.options['avoid_points']
        # # for px, py, pz in avoid_points: 
        #     # 1. Geometry
            dist = np.sqrt( (inputs['x'] - px)**2 + (inputs['y'] - py)**2 + 0.*(inputs['z']  - pz)**2)
        #     
        #     # 2. Local Sound Pressure (dB)
        #     # TEMP!!! 
            Lp = Lwa - 20 * np.log10(np.maximum(dist, 1.0)) - 11 + 3 
        #     # Lp = 1. / (dist2 + 1e-2)   
        #     
        #     # 3. Non-Dimensional Noise Ratio (Lp / Limit)
        #     # A value > 1.0 means the limit is exceeded.

         #    # norm_Lp = Lp / 60.0
            norm_Lp = Lp / limit_Lp
            all_norm_Lp.append(norm_Lp)

        #     # 4. Annoyance Metric (Simplified Power Law)
        #     # We scale this so 1.0 is a "baseline" annoyance at the limit
        #     # CHECK!!!  
        #     # annoyance = weight * 10**((Lp - limit_Lp) / 10.0)
        #     annoyance = Lp / 1.0      
        #     # annoyance = Lp / limit_Lp 
        #     all_annoyance.append(annoyance)

        # Convert to arrays for KS aggregation: Shape (num_observers, num_nodes)
        all_norm_Lp = np.array(all_norm_Lp)
        # all_annoyance = np.array(all_annoyance)

        # 5. Aggregate using KS Function (Smooth Max across all observers)
        # This gives the worst-case normalized noise across the ground at each node
        outputs['ks_norm_Lp'] = (1/rho) * np.log(np.sum(np.exp(rho * all_norm_Lp), axis=0))

        outputs['ks_rate'] = np.exp( 1.0 * outputs['ks_norm_Lp'] )

        # outputs['ks_norm_Lp'] = np.sum( all_norm_Lp, axis=0)
       
        #print( (1/rho) * np.log(np.sum(np.exp(rho * all_norm_Lp), axis=0)) ) 
 
        # This gives the "Total Instantaneous Annoyance" felt by the community
        # Note: We use sum here instead of KS if you want the 'total' impact, 
        # or KS if you want the 'worst-off' person's annoyance.
        # outputs['ks_annoyance'] = (1/rho) * np.log(np.sum(np.exp(rho * all_annoyance), axis=0))
        # outputs['ks_annoyance'] = np.sum(all_annoyance, axis=0)
  

        # TEMP!!! 
        # ks_norm_Lp_inst = np.zeros_like(x)
        ks_annoyance_inst = np.zeros_like(x)

        avoid_points = self.options['avoid_points']
        
        #for (px, py, pz) in avoid_points:
        for px, py, pz, limit_Lp, weight in points:
            # dist2 = (inputs['x'] - px)**2 + (inputs['y'] - py)**2 + 0*(inputs['z'] - pz)**2
            dist2 = (inputs['x'] - px)**2 + (inputs['y'] - py)**2 + (inputs['z'] - pz)**2 
            # ks_norm_Lp_inst += 1.0 / (dist2 + 1.e-4) 
            ks_annoyance_inst += 1.0 / (dist2 + 1.e-4) 

        # outputs['ks_norm_Lp'] = 1.0 * ks_norm_Lp_inst
        outputs['ks_annoyance'] = 1.0 * ks_annoyance_inst  

        # print( (1/rho) * np.log(np.sum(np.exp(rho * all_annoyance), axis=0)) )

        # Get the spatially varying Dryden gust
        # (Assuming your implementation provides wx, wy, wz)
        # outputs['wind_x'] = dryden_field.get_gust(x, y, z, sigma=1.0)
        # outputs['wind_y'] = dryden_field.get_gust(y, x, z, sigma=0.5)
        # outputs['wind_z'] = dryden_field.get_gust(y, x, z, sigma=0.25)

        # Let's assume Mean Wind from earlier + this new Gust
        # v_ref, z_ref = 5.0, 20.0
        # mean_wx = v_ref * (np.maximum(z, 0.1) / z_ref)**0.15

        # Kinematics
        # 2. Kinematics: Ground Velocity = Air Velocity + Wind
        # Here, vx, vy, vz are treated as the drone's velocity relative to AIR

        # TEMP NO WIND
        outputs['x_dot'] = inputs['vx'] # + mean_wx + outputs['wind_x']
        outputs['y_dot'] = inputs['vy'] # + outputs['wind_y']
        outputs['z_dot'] = inputs['vz'] # + outputs['wind_z']

        # 3. Dynamics: Accelerations change the Air Velocity
        outputs['vx_dot'] = inputs['ax']
        outputs['vy_dot'] = inputs['ay']
        outputs['vz_dot'] = inputs['az']

        # Penalty calculation: 1 / dist^2
        # avoid_points = self.options['avoid_points']
        # penalty_strength = self.options['penalty_strength']
        # total_penalty = np.zeros(self.options['num_nodes'],dtype=dtype)

        # for (px, py, pz) in avoid_points:
        #     dist2 = (inputs['x'] - px)**2 + (inputs['y'] - py)**2 + (inputs['z'] - pz)**2
        #     # Avoid division by zero with a small epsilon
        #     total_penalty += penalty_strength / (dist2 + 1e-4)
        
        # outputs['inst_penalty'] = total_penalty
        outputs['acc_mag2'] = inputs['ax']**2 + inputs['ay']**2 + inputs['az']**2
        # v_mag3 = inputs['vx']**3 + inputs['vy']**3 + inputs['vz']**3
        eps = 1e-6 
        v_mag2 = inputs['vx']**2 + inputs['vy']**2 + inputs['vz']**2
        outputs['energy'] = 0.01 * np.power(v_mag2 + eps, 1.5) + 0.1 * outputs['acc_mag2']
        # #outputs['energy'] = inputs['vx']**2 + inputs['vy']**2 + inputs['vz']**2

# --- Setup Problem ---

p = om.Problem()
p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
p.driver.opt_settings['print_level'] = 5
# p.driver.opt_settings['delta'] = 1e-1
p.driver.opt_settings['max_iter'] = 400

# --- Part A: Define Specific Landmarks ---
# Format: (x, y, z, limit_dB, annoyance_weight)
landmarks = [
    (10.0, 10.0, 0., 55.0, 3.0),  # Hospital (Very Quiet, High Priority)
    (400.0, 400.0, 0., 55.0, 2.0),  # Primary School (Quiet)
    (250.0, 250.0, 0., 55.0, 0.5),  # Industrial Zone (Loud permitted, Low Priority)
]

# --- Part B: Define Residential Grid ---
residential_points = []
# Create a 5x5 grid covering the "Main Street" residential block
grid_x = np.linspace(250, 450, 5)
grid_y = np.linspace(250, 445, 5)

for px in grid_x:
    for py in grid_y:
        # Standard residential limit (65dB) and baseline weight (1.0)
        residential_points.append((px, py, 0., 65.0, 2.0))

# --- Part C: Combine lists ---
sensitive_points = landmarks + residential_points

print(f"Total observers defined: {len(sensitive_points)}")

# plot_acoustic_map(sensitive_points)

# Generate Obstacles
avoid_points = [(random.uniform(150, 450), random.uniform(150, 450), 0) for _ in range(80)]

traj = dm.Trajectory()
phase = dm.Phase(ode_class=DroneODE, 
                 ode_init_kwargs={'avoid_points': avoid_points, 'sensitive_points': sensitive_points},
                 transcription=dm.GaussLobatto(num_segments=8, order=3))
p.model.add_subsystem('traj', traj)
traj.add_phase('phase0', phase)

# Time, States, and Controls
phase.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=(5, 200))

# States: x, y, z are fixed at start (0,0,0) and end (500,500,10)
# for s in ['x', 'y', 'z']:
for s in ['x', 'y']:
    phase.add_state(s, fix_initial=True, fix_final=True, ref=500.0, rate_source=f'{s}_dot')

for s in ['z']:
    phase.add_state(s, fix_initial=True, fix_final=True, ref=25.0, rate_source=f'{s}_dot')

phase.add_path_constraint('z', lower=0, upper=60., ref=50.0)

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

phase.add_state('total_annoyance', 
                rate_source='ks_annoyance', # The ODE output we want to integrate
                units=None, 
                fix_initial=True, # Starts at 0
                fix_final=False)  # Let the optimizer find the final total

phase.add_state('total_lp_dose', 
                rate_source='ks_norm_Lp', 
                units=None, 
                fix_initial=True, 
                fix_final=False)

# phase.add_state('ks_integral', 
#                 rate_source='ks_rate', 
#                 units=None, 
#                 fix_initial=True, 
#                fix_final=False)

phase.add_state('ks_integral', rate_source='ks_rate', ref=1e1, fix_initial=True, fix_final=False)

# Accelerations as Controls
for a in ['ax', 'ay', 'az']:
    phase.add_control(a, lower=-8.0, upper=8.0, rate_continuity=True, 
                  rate2_continuity=False)

# phase.add_control('az', lower = 0, upper = 0) 

# INTEGRATE PENALTY: This creates the "Accumulated Reward" automatically
phase.add_state('total_penalty', rate_source='inst_penalty', ref=5000.0, fix_initial=True, fix_final=False)



phase.add_timeseries_output('wind_x')
phase.add_timeseries_output('wind_y')
phase.add_timeseries_output('wind_z')

# Objective: Minimize Time + Penalty_Integral
class ObjectiveComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('time', units=None)
        self.add_input('total_lp_dose', units=None) 
        self.add_input('ks_integral', units=None) 
        self.add_input('energy_final', units=None)
        self.add_output('J')
        self.declare_partials('*', '*', method='cs')
    def compute(self, inputs, outputs):
        # outputs['J'] = inputs['time'] + 20.0 * inputs['penalty']
        # 9.04: outputs['J'] = 0.003 * ( inputs['time'] + 6.0 * inputs['penalty'] + 2.0 * inputs['acc_integral'] + 0.001 * inputs['energy_final'] )
        # 10.04 outputs['J'] = 0.002 * ( inputs['time'] + 0.0 * inputs['penalty'] + 2.0 * inputs['acc_integral'] + 0.001 * inputs['energy_final'] ) + 0.035*inputs['total_lp_dose']
        outputs['J'] = 0.005 * inputs['time'] + 0.00*inputs['total_lp_dose'] + 5.e-4 * inputs['energy_final'] + 0.025 * inputs['ks_integral']
        # outputs['J'] = 0.003 * ( inputs['time'] + 6.0 * inputs['penalty'] + 5.0 * inputs['acc_integral'] + 0.001 * inputs['energy_final'] )
        # outputs['J'] = inputs['time'] + 4.0 * inputs['penalty'] + 0.02 * inputs['energy_final']

p.model.add_subsystem('obj_comp', ObjectiveComp())
p.model.connect('traj.phase0.timeseries.time', 'obj_comp.time', src_indices=[-1])
# p.model.connect('traj.phase0.timeseries.total_penalty', 'obj_comp.penalty', src_indices=[-1])
# p.model.connect('traj.phase0.timeseries.acc_integral', 'obj_comp.acc_integral', src_indices=[-1])

# power / energy required 

p.model.connect('traj.phase0.timeseries.energy_spent', 
                'obj_comp.energy_final', src_indices=[-1])

# acoustic annoyance 

# p.model.connect('traj.phase0.states:total_annoyance', 
#              'obj_comp.total_annoyance', src_indices=[-1])

p.model.connect('traj.phase0.states:total_lp_dose', 
              'obj_comp.total_lp_dose', src_indices=[-1])


#p.model.add_subsys('ks_objective', 
#                   om.ExecComp(f'obj = (1.0/{rho}) * log(ks_final)', 
#                               obj=0.0, ks_final=1.0))

# Connect the final value of the state to this component
p.model.connect('traj.phase0.states:ks_integral', 'obj_comp.ks_integral', src_indices=[-1])

p.model.add_objective('obj_comp.J')

p.setup(check=True)

# om.n2(p) 

# --- Initial Guesses ---
p.set_val('traj.phase0.t_duration', 30.0)
p.set_val('traj.phase0.states:x', phase.interp('x', [0, 500.]))
p.set_val('traj.phase0.states:y', phase.interp('y', [0, 450.]))
p.set_val('traj.phase0.states:z', phase.interp('z', [0, 0.01]))
# To set the actual value (e.g., starting at 0 m/s)
p.set_val('traj.phase0.states:vx', phase.interp('vx', [0, 0]))
p.set_val('traj.phase0.states:vy', phase.interp('vy', [0, 0]))
p.set_val('traj.phase0.states:vz', phase.interp('vz', [0, 0]))
p.set_val('traj.phase0.states:total_penalty', 0.0)
p.set_val('traj.phase0.states:energy_spent', phase.interp('energy_spent', [0, 100]))
p.set_val('traj.phase0.states:acc_integral', phase.interp('energy_spent', [0, 100]))



#p.run_model() # Run one iteration to populate data
#data = p.check_partials(compact_print=True)

p.run_model() # Populate the data
p.check_partials(compact_print=True,method='fd', step=1e-4) 

p.run_driver()

# Extracting the values from the objective component
final_time = p.get_val('obj_comp.time')[0]
#final_penalty = p.get_val('obj_comp.penalty')[0]
final_energy = p.get_val('obj_comp.energy_final')[0]
#acc_integral = p.get_val('obj_comp.acc_integral')[0]
total_lp_dose = p.get_val('obj_comp.total_lp_dose')[0]
ks_integral = p.get_val('obj_comp.ks_integral')[0]
final_total_J = p.get_val('obj_comp.J')[0]

print(f"\n{'='*30}")
print(f"OPTIMIZATION RESULTS")
print(f"{'='*30}")
print(f"Final Time:         {final_time:.4f} s")
#print(f"Obstacle Penalty:   {final_penalty:.4f}")
print(f"Energy Expenditure: {final_energy:.4f}")
#print(f"Acceleration integ: {acc_integral:.4f}")
print(f"TOT Lp time integ: {total_lp_dose:.4f}")
print(f"KS_INTEGRAL : {ks_integral:.4f}")
print(f"Total Objective J:  {final_total_J:.4f}")
print(f"{'='*30}")


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
# obs_x, obs_y, obs_z = zip(*avoid_points)
obs_x, obs_y, obs_z, limit, weight = zip(*sensitive_points)
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


