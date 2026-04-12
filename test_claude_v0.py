import openmdao.api as om
import dymos as dm
import numpy as np

# 1. Define the ODE (equations of motion)
class DroneODE(om.ExplicitComponent):
    """
    Simple 2D or 3D drone dynamics
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('avoid_points', types=list)  # List of (x,y) or (x,y,z) points
    
    def setup(self):
        nn = self.options['num_nodes']
        
        # States: position (x, y, z) and velocity (vx, vy, vz)
        self.add_input('vx', shape=(nn,), units='m/s')
        self.add_input('vy', shape=(nn,), units='m/s')
        self.add_input('vz', shape=(nn,), units='m/s')
        
        # Controls: acceleration or velocity direction
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
        
        # Outputs for constraints
        self.add_input('x', shape=(nn,), units='m')
        self.add_input('y', shape=(nn,), units='m')
        self.add_input('z', shape=(nn,), units='m')
        
        # ADD THIS: Output for velocity magnitude squared
        self.add_output('v_squared', shape=(nn,), units='m**2/s**2')
        
        # Distance to avoid points (for path constraints)
        n_avoid = len(self.options['avoid_points'])
        self.add_output('dist_to_avoid', shape=(nn, n_avoid), units='m')
        
    def compute(self, inputs, outputs):
        # Simple dynamics: velocity = integral of acceleration
        outputs['x_dot'] = inputs['vx']
        outputs['y_dot'] = inputs['vy']
        outputs['z_dot'] = inputs['vz']
        outputs['vx_dot'] = inputs['ax']
        outputs['vy_dot'] = inputs['ay']
        outputs['vz_dot'] = inputs['az']

        # ADD THIS: Compute velocity magnitude squared
        outputs['v_squared'] = (inputs['vx']**2 + 
                               inputs['vy']**2 + 
                               inputs['vz']**2)
    
        # Compute distance to each avoid point
        avoid_points = self.options['avoid_points']
        for i, (ax, ay, az) in enumerate(avoid_points):
            dx = inputs['x'] - ax
            dy = inputs['y'] - ay
            dz = inputs['z'] - az
            outputs['dist_to_avoid'][:, i] = np.sqrt(dx**2 + dy**2 + dz**2)

# 2. Setup the optimization problem
p = om.Problem()
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'   # 'IPOPT'  # or 'SNOPT'

# 3. Create trajectory and phase
traj = dm.Trajectory()
p.model.add_subsystem('traj', traj)

avoid_points = [(10, 10, 5), (20, 15, 8), (30, 5, 10)]  # Points to avoid

phase = dm.Phase(
    ode_class=DroneODE,
    ode_init_kwargs={'avoid_points': avoid_points},
    transcription=dm.GaussLobatto(num_segments=20, order=3)
)
traj.add_phase('phase0', phase)

# 4. Set time options (allow optimizer to minimize time)
phase.set_time_options(fix_initial=True, fix_duration=False, 
                       duration_bounds=(1.0, 100.0))

# 5. Add state variables with initial/final constraints
phase.add_state('x', fix_initial=True, fix_final=True, 
                rate_source='x_dot', units='m')
phase.add_state('y', fix_initial=True, fix_final=True,
                rate_source='y_dot', units='m')
phase.add_state('z', fix_initial=True, fix_final=True,
                rate_source='z_dot', units='m')

phase.add_state('vx', fix_initial=True, fix_final=False,
                rate_source='vx_dot', units='m/s')
phase.add_state('vy', fix_initial=True, fix_final=False,
                rate_source='vy_dot', units='m/s')
phase.add_state('vz', fix_initial=True, fix_final=False,
                rate_source='vz_dot', units='m/s')

# 6. Add controls (accelerations)
phase.add_control('ax', lower=-5, upper=5, units='m/s**2')
phase.add_control('ay', lower=-5, upper=5, units='m/s**2')
phase.add_control('az', lower=-5, upper=5, units='m/s**2')

# 7. Add velocity magnitude constraint
phase.add_path_constraint('v_squared', 
                         upper=25.0,  # v_max = 5 m/s, so v^2 <= 25
                         units='m**2/s**2')

# 8. Add obstacle avoidance constraints
phase.add_path_constraint('dist_to_avoid',
                         lower=2.0,  # Applied to all columns
                         units='m')

#for i in range(len(avoid_points)):
#    phase.add_path_constraint(f'dist_to_avoid[:, {i}]',
#                             lower=2.0,  # Minimum safe distance = 2m
#                             units='m')

# 9. Set objective: minimize time
phase.add_objective('time', loc='final', scaler=1.0)

# 10. Setup and set initial values
p.setup()

# Set initial conditions: Point A
p.set_val('traj.phase0.states:x', phase.interp('x', [0, 50]))  # A to B
p.set_val('traj.phase0.states:y', phase.interp('y', [0, 50]))
p.set_val('traj.phase0.states:z', phase.interp('z', [0, 20]))
p.set_val('traj.phase0.states:vx', phase.interp('vx', [0, 0]))
p.set_val('traj.phase0.states:vy', phase.interp('vy', [0, 0]))
p.set_val('traj.phase0.states:vz', phase.interp('vz', [0, 0]))

# Run optimization
dm.run_problem(p)

# Extract and plot results
sol = om.CaseReader('dymos_solution.db').get_case('final')

