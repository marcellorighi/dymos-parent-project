import openmdao.api as om
import dymos as dm
import numpy as np
import matplotlib.pyplot as plt

class DroneODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        # Coordinates of obstacles: [(x, y, z), ...]
        self.options.declare('avoid_points', types=list, default=[])
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
        
        # Controls (Accelerations)
        self.add_input('ax', shape=(nn,), units='m/s**2')
        self.add_input('ay', shape=(nn,), units='m/s**2')
        self.add_input('az', shape=(nn,), units='m/s**2')
        
        # Outputs (State rates)
        self.add_output('x_dot', shape=(nn,), units='m/s')
        self.add_output('y_dot', shape=(nn,), units='m/s')
        self.add_output('z_dot', shape=(nn,), units='m/s')
        self.add_output('vx_dot', shape=(nn,), units='m/s**2')
        self.add_output('vy_dot', shape=(nn,), units='m/s**2')
        self.add_output('vz_dot', shape=(nn,), units='m/s**2')
        
        # Auxiliary outputs for objective
        self.add_output('accel_mag', shape=(nn,), units='m/s**2')
        self.add_output('obstacle_penalty', shape=(nn,), units=None)

        # CRITICAL FIX: Declare partial derivatives. 
        # Using 'cs' (complex step) provides exact derivatives for ExplicitComponents.
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        avoid_points = self.options['avoid_points']
        strength = self.options['penalty_strength']
        radius = self.options['penalty_radius']

        # Kinematics
        outputs['x_dot'] = inputs['vx']
        outputs['y_dot'] = inputs['vy']
        outputs['z_dot'] = inputs['vz']
        
        # Dynamics (States rates of velocity are just accelerations)
        outputs['vx_dot'] = inputs['ax']
        outputs['vy_dot'] = inputs['ay']
        outputs['vz_dot'] = inputs['az']
        
        # Acceleration magnitude (to minimize effort)
        outputs['accel_mag'] = np.sqrt(inputs['ax']**2 + inputs['ay']**2 + inputs['az']**2 + 1e-6)
        
        # Obstacle avoidance penalty (Exponential field)
        penalty = np.zeros(nn, dtype=inputs['x'].dtype)
        for pt in avoid_points:
            dx = inputs['x'] - pt[0]
            dy = inputs['y'] - pt[1]
            dz = inputs['z'] - pt[2]
            dist_sq = dx**2 + dy**2 + dz**2
            # Smooth penalty that increases as distance decreases
            penalty += strength * np.exp(-dist_sq / (2 * radius**2))
        
        outputs['obstacle_penalty'] = penalty

class CombinedObjective(om.ExplicitComponent):
    """
    Component to compute a single scalar value for the optimizer to minimize.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('weight_time', default=1.0)
        self.options.declare('weight_accel', default=0.01)
        self.options.declare('weight_obs', default=10.0)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('t_duration', units='s')
        self.add_input('accel_mag', shape=(nn,), units='m/s**2')
        self.add_input('obstacle_penalty', shape=(nn,), units=None)
        
        self.add_output('obj', units=None)
        
        # Declare partials for the objective
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        w_t = self.options['weight_time']
        w_a = self.options['weight_accel']
        w_o = self.options['weight_obs']
        
        # Integrate acceleration and penalty over time (trapezoidal approximation or mean)
        # Dymos usually prefers we minimize a terminal value or an integral. 
        # Here we use a weighted sum of the duration and the mean values across nodes.
        avg_accel = np.mean(inputs['accel_mag'])
        avg_penalty = np.mean(inputs['obstacle_penalty'])
        
        outputs['obj'] = w_t * inputs['t_duration'] + w_a * avg_accel + w_o * avg_penalty

def optimize_drone_trajectory():
    p = om.Problem(model=om.Group())
    
    # Define Optimizer
    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.options['maxiter'] = 200
    p.driver.options['tol'] = 1e-6
    p.driver.declare_coloring() # Speed up derivative calculations

    # Define Obstacles
    obstacles = [(25.0, 2.0, 0.0), (15.0, -2.0, 5.0)]

    # Define Dymos Trajectory and Phase
    traj = p.model.add_subsystem('traj', dm.Trajectory())
    # Using GaussLobatto transcription
    phase = traj.add_phase('phase0', 
                           dm.Phase(ode_class=DroneODE, 
                                    ode_init_kwargs={'avoid_points': obstacles},
                                    transcription=dm.transcriptions.GaussLobatto(num_segments=15, order=3)))

    # Set Time Options
    phase.set_time_options(fix_initial=True, duration_bounds=(1.0, 100.0), units='s')

    # Set State Options
    phase.add_state('x', fix_initial=True, fix_final=True, units='m', rate_source='x_dot')
    phase.add_state('y', fix_initial=True, fix_final=True, units='m', rate_source='y_dot')
    phase.add_state('z', fix_initial=True, fix_final=True, units='m', rate_source='z_dot')
    phase.add_state('vx', fix_initial=True, fix_final=False, units='m/s', rate_source='vx_dot')
    phase.add_state('vy', fix_initial=True, fix_final=False, units='m/s', rate_source='vy_dot')
    phase.add_state('vz', fix_initial=True, fix_final=False, units='m/s', rate_source='vz_dot')

    # Set Control Options (Accelerations)
    phase.add_control('ax', continuity=True, rate_continuity=True, units='m/s**2', lower=-10.0, upper=10.0)
    phase.add_control('ay', continuity=True, rate_continuity=True, units='m/s**2', lower=-10.0, upper=10.0)
    phase.add_control('az', continuity=True, rate_continuity=True, units='m/s**2', lower=-10.0, upper=10.0)

    # Path constraints
    phase.add_boundary_constraint('vx', loc='final', equals=0.0, units='m/s')
    phase.add_boundary_constraint('vy', loc='final', equals=0.0, units='m/s')
    phase.add_boundary_constraint('vz', loc='final', equals=0.0, units='m/s')

    # Add Objective Component
    # We connect ODE outputs to the objective
    p.model.add_subsystem('obj_comp', CombinedObjective(num_nodes=phase.options['transcription'].grid_data.num_nodes))
   
    p.model.connect('traj.phase0.t_duration', 'obj_comp.t_duration')
    p.model.connect('traj.phase0.ode.accel_mag', 'obj_comp.accel_mag')
    p.model.connect('traj.phase0.ode.obstacle_penalty', 'obj_comp.obstacle_penalty')

    p.model.add_objective('obj_comp.obj')

    p.setup()

    # Initial Guesses
    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 20.0)

    p.set_val('traj.phase0.states:x', phase.interp('x', [0, 50]))
    p.set_val('traj.phase0.states:y', phase.interp('y', [0, 0]))
    p.set_val('traj.phase0.states:z', phase.interp('z', [0, 10]))
    
    p.set_val('traj.phase0.states:vx', phase.interp('vx', [0, 0]))
    p.set_val('traj.phase0.states:vy', phase.interp('vy', [0, 0]))
    p.set_val('traj.phase0.states:vz', phase.interp('vz', [0, 0]))

    # Seed controls with non-zero values to avoid starting at a local flat point
    p.set_val('traj.phase0.controls:ax', 0.1)
    p.set_val('traj.phase0.controls:ay', 0.1)
    p.set_val('traj.phase0.controls:az', 0.1)

    # Run Optimization
    dm.run_problem(p)

    return p

if __name__ == "__main__":
    prob = optimize_drone_trajectory()
    
    # Extract results
    t = prob.get_val('traj.phase0.timeseries.time')
    x = prob.get_val('traj.phase0.timeseries.x')
    y = prob.get_val('traj.phase0.timeseries.y')
    z = prob.get_val('traj.phase0.timeseries.z')
    ax = prob.get_val('traj.phase0.timeseries.ax')
    
    print(f"Final Duration: {prob.get_val('traj.phase0.t_duration')[0]:.2f}s")
    
    # Plotting
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x, y, z, label='Trajectory')
    ax1.set_title('Drone Path')
    
    ax2 = fig.add_subplot(122)
    ax2.plot(t, ax, label='ax')
    ax2.set_title('Acceleration X vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('m/s^2')
    plt.show()
