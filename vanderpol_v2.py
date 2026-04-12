import numpy as np
import openmdao.api as om
from openmdao.utils.array_utils import evenly_distrib_idxs


class VanderpolODE(om.ExplicitComponent):
    """intentionally slow version of vanderpol_ode for effects of demonstrating distributed component calculations

    MPI can run this component in multiple processes, distributing the calculation of derivatives.
    This code has a delay in it to simulate a longer computation. It should run faster with more processes.
    """

    def __init__(self, *args, **kwargs):
        self.progress_prints = False
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('distrib', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, nn)  # (#cpus, #inputs) -> (size array, offset array)
        self.start_idx = offsets[rank]
        self.io_size = sizes[rank]  # number of inputs and outputs managed by this distributed process
        self.end_idx = self.start_idx + self.io_size

        # inputs: 2 states and a control
        self.add_input('x0', val=np.ones(nn), desc='derivative of Output', units='V/s')
        self.add_input('x1', val=np.ones(nn), desc='Output', units='V')
        self.add_input('x2', val=np.ones(nn), desc='derivative of Output', units='V/s')
        self.add_input('x3', val=np.ones(nn), desc='Output', units='V')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)
        self.add_input('omega', val=np.ones(nn), desc='control', units=None)

        # outputs: derivative of states
        # the objective function will be treated as a state for computation, so its derivative is an output
        self.add_output('x0dot', val=np.ones(self.io_size), desc='second derivative of Output',
                        units='V/s**2', distributed=self.options['distrib'])
        self.add_output('x1dot', val=np.ones(self.io_size), desc='derivative of Output',
                        units='V/s', distributed=self.options['distrib'])
        self.add_output('x2dot', val=np.ones(self.io_size), desc='second derivative of Output',
                        units='V/s**2', distributed=self.options['distrib'])
        self.add_output('x3dot', val=np.ones(self.io_size), desc='derivative of Output',
                        units='V/s', distributed=self.options['distrib'])
        self.add_output('Jdot', val=np.ones(self.io_size), desc='derivative of objective',
                        units='1.0/s', distributed=self.options['distrib'])

        # self.declare_coloring(method='cs')
        # # partials
        r = np.arange(self.io_size, dtype=int)
        c = r + self.start_idx

        self.declare_partials(of='x0dot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='x2dot', wrt='x2',  rows=r, cols=c)
        self.declare_partials(of='x2dot', wrt='x3',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='u',   rows=r, cols=c, val=1.0)
        self.declare_partials(of='x0dot', wrt='omega',   rows=r, cols=c, val=1.0)

        self.declare_partials(of='x1dot', wrt='x0',  rows=r, cols=c, val=1.0)

        self.declare_partials(of='Jdot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='x2',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='x3',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='u',   rows=r, cols=c)

    def compute(self, inputs, outputs):

        # The inputs contain the entire vector, be each rank will only operate on a portion of it.
        x0 = inputs['x0'][self.start_idx:self.end_idx]
        x1 = inputs['x1'][self.start_idx:self.end_idx]
        x2 = inputs['x2'][self.start_idx:self.end_idx]
        x3 = inputs['x3'][self.start_idx:self.end_idx]
        u = inputs['u'][self.start_idx:self.end_idx]
        omega = inputs['omega'][self.start_idx:self.end_idx]

        outputs['x0dot'] = (1.0 - x1**2) * u *x0 - omega*x1 
        outputs['x1dot'] = x0
        outputs['x2dot'] = -x3 
        outputs['x3dot'] = x2
        # outputs['Jdot'] = x0**2 + x1**2 + u**2
        outputs['Jdot'] = (x1 - x3)**2 + (x0 - x2)**2 # + u**2

    def compute_partials(self, inputs, jacobian):
        x0 = inputs['x0'][self.start_idx:self.end_idx]
        x1 = inputs['x1'][self.start_idx:self.end_idx]
        x2 = inputs['x0'][self.start_idx:self.end_idx]
        x3 = inputs['x1'][self.start_idx:self.end_idx]
        u = inputs['u'][self.start_idx:self.end_idx]
        omega = inputs['omega'][self.start_idx:self.end_idx]

        jacobian['x0dot', 'x0'] = (1.0 - x1 * x1) * u 
        jacobian['x0dot', 'u'] = (1.0 - x1 * x1) * x0 
        jacobian['x0dot', 'omega'] = -x1 
        jacobian['x2dot', 'x2'] = 0. 
        jacobian['x0dot', 'x1'] = -2.0 * x1 * u * x0  - 1 
        jacobian['x2dot', 'x3'] = -1.0 
        # jacobian['Jdot', 'x0'] = 2.0 * x0
        jacobian['Jdot', 'x0'] = (x0 - x2)*2. 
        jacobian['Jdot', 'x2'] = -(x0 - x2)*2. 
        # jacobian['Jdot', 'x1'] = 2.0 * x1
        jacobian['Jdot', 'x1'] = (x1 - x3)*2. 
        jacobian['Jdot', 'x3'] = -(x1 - x3)*2
        # jacobian['Jdot', 'u'] = 2.0 * u
import openmdao.api as om
import dymos as dm


def vanderpol(transcription='gauss-lobatto', num_segments=15, transcription_order=3,
              compressed=True, optimizer='IPOPT', use_pyoptsparse=False):
    """Dymos problem definition for optimal control of a Van der Pol oscillator"""

    # define the OpenMDAO problem
    p = om.Problem(model=om.Group())

    if not use_pyoptsparse:
        p.driver = om.ScipyOptimizeDriver()
    else:
        p.driver = om.pyOptSparseDriver(print_results=False)
    p.driver.options['optimizer'] = optimizer
    if use_pyoptsparse:
        if optimizer == 'SNOPT':
            p.driver.opt_settings['iSumm'] = 6  # show detailed SNOPT output
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['print_level'] = 4
    p.driver.declare_coloring()

    # define a Trajectory object and add to model
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', subsys=traj)

    # define a Transcription
    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)

    # define a Phase as specified above and add to Phase
    phase = dm.Phase(ode_class=VanderpolODE, transcription=t)
    traj.add_phase(name='phase0', phase=phase)

    t_final = 24
    phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=t_final, units='s')

    # set the State time options
    phase.add_state('x0', fix_initial=True, fix_final=True,
                    rate_source='x0dot',
                    units='V/s', ref=0.1, defect_ref=0.1)  # target required because x0 is an input
    phase.add_state('x1', fix_initial=True, fix_final=True,
                    rate_source='x1dot',
                    units='V', ref=0.1, defect_ref=0.1)
    phase.add_state('x2', fix_initial=True, fix_final=False,
                    rate_source='x2dot',
                    units='V/s', ref=0.1, defect_ref=0.1)  # target required because x0 is an input
    phase.add_state('x3', fix_initial=True, fix_final=False,
                    rate_source='x3dot',
                    units='V', ref=0.1, defect_ref=0.1)
    phase.add_state('J', fix_initial=True, fix_final=False,
                    rate_source='Jdot',
                    units=None)

    # define the control
    phase.add_control(name='u', units=None, lower=-1.25, upper=1.5, continuity=True,
                      rate_continuity=True)
    phase.add_control(name='omega', units=None, lower=0.75, upper=1.25, continuity=True,
                      rate_continuity=True)

    # define objective to minimize
    phase.add_objective('J', loc='final')

    # setup the problem
    p.setup(check=True)

    phase.set_time_val(0.0, t_final)

    # add a linearly interpolated initial guess for the state and control curves
    # phase.set_state_val('x0', [1, 0])
    # phase.set_state_val('x0', [1, 0])
    phase.set_state_val('x0', [0, 1])
    # phase.set_control_val('u', -0.05)
    phase.set_control_val('u', +0.75)

    return p

# Create the Dymos problem instance
# p = vanderpol(transcription='gauss-lobatto', num_segments=15,
#               transcription_order=3, compressed=True, optimizer='IPOPT')

# # Enable grid refinement and find optimal control solution to stop oscillation
# p.model.traj.phases.phase0.set_refine_options(refine=True, tol=1.0E-6)

# dm.run_problem(p, simulate=True, refine_iteration_limit=2, refine_method='hp')

# Create the Dymos problem instance
p = vanderpol(transcription='gauss-lobatto', use_pyoptsparse=True, optimizer='IPOPT', num_segments=16)

# Run the problem (simulate only)
dm.run_problem(p, run_driver=True, simulate=True)

from dymos.examples.plotting import plot_results

# Display the results

sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim_prob = p.model.traj.sim_prob
sim = om.CaseReader(sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')

# Extract time
t_sim = sim.get_val('traj.phase0.timeseries.time')
t_sol = sol.get_val('traj.phase0.timeseries.time')

# Extract Control (u)
u_sim = sim.get_val('traj.phase0.timeseries.u')

# Extract Objective/Cost (J)
cost_sim = sim.get_val('traj.phase0.timeseries.J')


plot_results([('traj.phase0.timeseries.time',
                'traj.phase0.timeseries.x1',
                'time (s)',
                '$x_1$ (V)'),
              ('traj.phase0.timeseries.time',
              'traj.phase0.timeseries.x0',
              'time (s)',
              '$x_0$ (V/s)'),
              ('traj.phase0.timeseries.time',
                'traj.phase0.timeseries.J',
                'time (s)',
                'J'),
              ('traj.phase0.timeseries.x0',
                'traj.phase0.timeseries.x1',
                '$x_0$ (V/s)',
                '$x_1$ (V)'),
              ('traj.phase0.timeseries.time',
              'traj.phase0.timeseries.u',
              'time (s)',
              'control u'),
              ],
              title='Van Der Pol Simulation',
              p_sol=sol, p_sim=sim)

import matplotlib.pyplot as plt

# 1. Extract the data into simple arrays
t_sim = sim.get_val('traj.phase0.timeseries.time')
x1_sim = sim.get_val('traj.phase0.timeseries.x1')
x0_sim = sim.get_val('traj.phase0.timeseries.x0')
x3_sim = sim.get_val('traj.phase0.timeseries.x3')
x2_sim = sim.get_val('traj.phase0.timeseries.x2')

t_sol = sol.get_val('traj.phase0.timeseries.time')
x1_sol = sol.get_val('traj.phase0.timeseries.x1')

# Extract Control (u)
u_sim = sim.get_val('traj.phase0.timeseries.u')
omega_sim = sim.get_val('traj.phase0.timeseries.omega')

# Extract Objective/Cost (J)
cost_sim = sim.get_val('traj.phase0.timeseries.J')

# 2. Create the plot

# plt.figure(figsize=(10, 5))
fig, ax = plt.subplots(5, 1, figsize=(12, 9))

plt.subplot(5, 1, 1)
# Plot the smooth simulation line
plt.plot(t_sim, x1_sim, label='Simulation (Continuous)', color='blue', linewidth=1)
plt.plot(t_sim, x3_sim, label='Simulation (Continuous)', color='green', linewidth=1)

plt.xlabel('Time (s)')
plt.ylabel('$x_1$ (V)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(t_sim, x0_sim, label='Simulation (Continuous)', color='blue', linewidth=1)
plt.plot(t_sim, x2_sim, label='x_2 Simulation (Continuous)', color='green', linewidth=1)

plt.xlabel('Time (s)')
plt.ylabel('$x_1$ (V)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(x0_sim, x1_sim, label='Simulation (Continuous)', color='blue', linewidth=1)
plt.plot(x2_sim, x3_sim, label='Simulation (Continuous)', color='green', linewidth=1)

plt.xlabel('$x_1$ (V)')
plt.ylabel('$x_0$ (V)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot the solution nodes as points (to see the LGL discretization)
# plt.scatter(t_sol, x1_sol, label='Solution Nodes (Discrete)', color='red', zorder=5)

plt.subplot(5, 1, 4)
plt.plot(t_sim, u_sim, label='Simulation (Continuous)', color='blue', linewidth=1)
plt.plot(t_sim, omega_sim, label='Simulation (Continuous)', color='green', linewidth=1)

plt.subplot(5, 1, 5)
plt.plot(t_sim, cost_sim, label='Simulation (Continuous)', color='blue', linewidth=1)

# plt.title('Van Der Pol Oscillator: State $x_1$')
plt.xlabel('Time (s)')
plt.ylabel('$x_1$ (V)')
plt.grid(True, alpha=0.3)
plt.legend()

# 3. CRITICAL: This keeps the window from closing
plt.tight_layout() 
plt.show()

