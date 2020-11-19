import os
import numpy as np
import gurobipy as gp

FOLDER = os.getcwd().split('\\')[-1]
if FOLDER == 'overtake':
    from overtake_settings import ERROR
elif FOLDER == 'simple':
    from simple_settings import ERROR

### Helpers ###
def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def is_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

###############################################################################
################################# Controller ##################################
###############################################################################

class ClfCbfController():

    @property
    def xd(self):
        if FOLDER == 'overtake':

            goal = xd(self.x,self.setpoint)
            self.td = goal[2]

        elif FOLDER == 'simple':
            goal = np.array([0,0])

        return goal

    def __init__(self,
                 u_max: float or np.ndarray):
        """
        """
        self.t             = 0
        self.x             = None
        self.u             = None
        self.dt            = None
        self.theta_hat     = None
        self.theta_max     = None
        self.theta_min     = None
        self.psi_hat       = None
        self.psi_max       = None
        self.sol           = None
        self.performance   = None
        self.safety        = None
        self.violations    = 0
        self.n_suboptimal  = 0
        self.setpoint      = 0
        self.u_max         = u_max
        self.name          = None

        self.clf = None

        # Assign value to nControls
        assert type(u_max) is float or type(u_max) is np.ndarray or u_max is None
        if type(u_max) is float:
            self.nControls = 1
        elif type(u_max) is np.ndarray:
            self.nControls = u_max.shape[0]
        else:
            # Irrelevant for now, but may need to fill in for future applications
            pass

        # Create the model
        self.m          = gp.Model("qp")
        self.m.setParam('OutputFlag',0)

    def _set_initial_conditions(self,**settings):
        """ """
        # Initialize some variables
        self.V = None
        self.B = None
        self.dV = None
        self.dBx = None
        self.dBt = None
        self.LfV = None
        self.LgV = None
        self.LdV = None#phi_max_re
        self.LfB = None
        self.LgB = None
        self.LdB = None

        # Set Default Initial Conditions
        self.dt        = 1e-3
        self.x         = np.zeros((2,))
        self.theta_hat = np.zeros((2,))
        self.theta_max = 1
        self.theta_min = -self.theta_max

        if 'x0' in settings.keys():
            assert type(settings['x0']) == np.ndarray
            self.x = settings['x0']

        if 'theta_hat0' in settings.keys():
            assert type(settings['theta_hat0']) == np.ndarray
            self.theta_hat = settings['theta_hat0']
            if 'psi_hat0' not in settings.keys() and 'psi_est0' not in settings.keys():
                self.psi_hat = settings['theta_hat0']

        if 'theta_est0' in settings.keys():
            assert type(settings['theta_hat0']) == np.ndarray
            self.theta_hat = settings['theta_est0']
            if 'psi_hat0' not in settings.keys() and 'psi_est0' not in settings.keys():
                self.psi_hat = settings['theta_est0']

        if 'psi_hat0' in settings.keys():
            assert type(settings['psi_hat0']) == np.ndarray
            self.psi_hat = settings['psi_hat0']

        if 'psi_est0' in settings.keys():
            assert type(settings['psi_est0']) == np.ndarray
            self.psi_hat = settings['psi_est0']

        if 'theta_max' in settings.keys():
            assert type(settings['theta_max']) == np.ndarray
            self.theta_max = settings['theta_max']
            if 'psi_max' not in settings.keys():
                self.psi_max = settings['theta_max']
            if 'theta_min' not in settings.keys():
                self.theta_min = -1*self.theta_max

        if 'theta_min' in settings.keys():
            assert type(settings['theta_min']) == np.ndarray
            self.theta_min = settings['theta_min']
            if 'psi_min' not in settings.keys():
                self.psi_min = settings['theta_min']

        if 'psi_max' in settings.keys():
            assert type(settings['psi_max']) == np.ndarray
            self.psi_max = settings['psi_max']
            if 'theta_max' not in settings.keys():
                self.theta_max = settings['psi_max']

        if 'psi_min' in settings.keys():
            assert type(settings['psi_min']) == np.ndarray
            self.psi_min = settings['psi_min']
            if 'theta_min' not in settings.keys():
                self.theta_min = settings['psi_min']

        if 'dt' in settings.keys():
            assert type(settings['dt']) == float
            self.dt = settings['dt']

    def update_tx(self,t,x):
        self.t = t
        self.x = x

    def _compute(self,not_skip=True):
        """ Computes the solution to the Quadratic Program.

        INPUTS
        ------
        None

        RETURNS
        ------
        sol: (np.ndarray) - decision variables which solve opt. prob.

        """
        # # Remove past constraints
        # self.m.remove(self.performance)
        # for s in self.safety:
        #     self.m.remove(s)

        # Update State and CBF/CLF expressions
        self.update_qp(self.x,self.t)

        # Solve QP
        # self.m.optimize()
        self.solve_qp()

        # Check QP for Errors
        self.sol, success, code = self.check_qp_sol(level=0)
        if not success:
            return code

        self.u = self.sol[:self.nControls]

        return self.sol

    def solve_qp(self):
        """ Reverts the Model settings to the defaults and then calls the 
        gurobipy.Model.optimize method to solve the optimization problem. 

        INPUTS
        ------
        None

        RETURNS
        ------
        None

        """
        # Revert to default settings
        self.m.setParam('BarHomogeneous',-1)
        self.m.setParam('NumericFocus',0)
        # self.m.setParam('OutputFlag',0)

        # Solve
        self.m.optimize()

    def check_qp_sol(self,
                     level: int = 0,
                     multiplier: int = 10):
        """
        Processes the status flag associated with the Model in order to perform
        error handling. If necessary, this will make adjustments to the solver
        settings and attempt to re-solve the optimization problem to obtain an
        accurate, feasible solution.

        INPUTS
        ------
        level: (int, optional) - current level of recursion in solution attempt

        RETURNS
        ------
        sol  : (np.ndarray)    - decision variables which solve the opt. prob.
        T/F  : (bool)          - pure boolean value denoting success or failure
        ERROR: (np.ndarray)    - error code for loop-breaking at higher level 
        """
        # Define Error Checking Parameters
        status  = self.m.status
        epsilon = 0.1
        success = 2

        # Obtain solution
        try:
            sol = np.array([v.x for v in self.m.getVars()])
        except AttributeError:
            sol = None

        # Check status
        if status == success:
            # Saturate u at u_max in case of solver error
            for uu in range(self.nControls):
                try:
                    # if abs(sol[uu]) > self.u_max[uu]:
                    #     print("{}: U{} = {}".format(self.name,uu,sol[uu])) 
                    sol[uu] = np.min([np.max([sol[uu],-self.u_max[uu]]),self.u_max[uu]])
                except TypeError:
                    sol[uu] = np.min([np.max([sol[uu],-self.u_max]),self.u_max])
            return sol,True,0

        else:
            self.m.write('diagnostic.lp')
            # self.m.setParam('OutputFlag',1)

            if status == 3:
                msg = "INFEASIBLE"
            elif status == 4:
                msg = "INFEASIBLE_OR_UNBOUNDED"
                self.m.setParam('BarHomogeneous',1)
                self.m.setParam('NumericFocus',np.min([level+1,3]))
                self.update_qp(self.x,self.t,multiplier)

                print("V:   {}".format(self.V))
                print("B:   {}".format(self.B))
                print("dV:  {}".format(self.dV))
                print("dBx: {}".format(self.dBx))
                print("dBt: {}".format(self.dBt))
                print("LfV: {}".format(self.LfV))
                print("LgV: {}".format(self.LgV))
                print("LdV: {}".format(self.LdV))#phi_max_re
                print("LfB: {}".format(self.LfB))
                print("LgB: {}".format(self.LgB))
                print("LdB: {}".format(self.LdB))

            elif status == 5:
                msg = "UNBOUNDED"
            elif status == 6:
                msg = "CUTOFF"
            elif status == 7:
                msg = "ITERATION_LIMIT"
            elif status == 8:
                msg = "NODE_LIMIT"
            elif status == 9:
                msg = "TIME_LIMIT"
            elif status == 10:
                msg = "SOLUTION_LIMIT"
            elif status == 11:
                msg = "INTERRUPTED"
            elif status == 12:
                msg = "NUMERIC"
                self.m.setParam('BarHomogeneous',1)
                self.m.setParam('NumericFocus',1)
            elif status == 13:
                msg = "SUBOPTIMAL"
            elif status == 14:
                msg = "INPROGRESS"
            elif status == 15:
                msg = "USER_OBJ_LIMIT"

            if status == 13:
                # print("SUBOPTIMAL SOLUTION")
                self.n_suboptimal = self.n_suboptimal + 1
                # Saturate u at u_max in case of solver error
                for uu in range(self.nControls):
                    try:
                        sol[uu] = np.min([np.max([sol[uu],-self.u_max[uu]]),self.u_max[uu]])
                    except TypeError:
                        sol[uu] = np.min([np.max([sol[uu],-self.u_max]),self.u_max])
                return sol,True,0

            if level < 3:
                self.m.optimize()
                return self.check_qp_sol(level+1,10**(level+1)**2)

            print("Solver Returned Code: {}".format(msg))
            
            return sol,False,ERROR