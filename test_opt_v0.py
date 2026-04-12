import openmdao.api as om
import dymos as dm
import numpy as np

# Simple ODE for testing
class SimpleODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('v', shape=(nn,), units='m/s')
        self.add_output('x_dot', shape=(nn,), units='m/s')
        self.declare_partials('x_dot', 'v', val=1.0, rows=np.arange(nn), cols=np.arange(nn))
    
    def compute(self, inputs, outputs):
        outputs['x_dot'] = inputs['v']

# Create problem
print("Creating problem...")
p = om.Problem()

# Set optimizer FIRST
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'IPOPT'
print(f"Optimizer set to: {p.driver.options['optimizer']}")

# Create simple trajectory
traj = dm.Trajectory()
p.model.add_subsystem('traj', traj)

phase = dm.Phase(
    ode_class=SimpleODE,
    transcription=dm.GaussLobatto(num_segments=3, order=3)
)
traj.add_phase('phase0', phase)

phase.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=(1, 10))
phase.add_state('x', fix_initial=True, fix_final=True, rate_source='x_dot')
phase.add_control('v', lower=0, upper=10, units='m/s')
phase.add_objective('time', loc='final')

# Setup
print("Setting up...")
p.setup()

# Initial values
print("Setting initial values...")
p.set_val('traj.phase0.t_initial', 0)
p.set_val('traj.phase0.t_duration', 5)
p.set_val('traj.phase0.states:x', phase.interp('x', [0, 10]))
p.set_val('traj.phase0.controls:v', 2.0)

# Run
print("Running optimization with IPOPT...")
try:
    p.run_driver()
    print("✓ SUCCESS!")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
