import os
import copy
import time
import traceback
import collections
import numpy as np
import gurobipy as gp

from gurobipy import GRB
from scipy import sparse
from clfcbfcontroller import ClfCbfController

FOLDER = os.getcwd().split('\\')[-1]
if FOLDER == 'overtake':
    from overtake_settings import *
elif FOLDER == 'simple':
    from simple_settings import *

###############################################################################
##################### FxT Parameter Estimation Parameters #####################
############################################################################### 

kG       = 1.2
k_e      = 0.001

def rcbf(x,t,err,Gamma):
    return cbf(x,t) - 1/2 * err.T @ np.linalg.inv(Gamma) @ err

def rcbf_partialx(x,t):
    return cbf_partialx(x,t)

def rcbf_partialt(x,t):
    return cbf_partialt(x,t)


###############################################################################
#################################### Class ####################################
###############################################################################

class LopezController(ClfCbfController):

    def __init__(self,u_max):
        super().__init__(u_max) 

        # Create the decision variables
        self.decision_variables = np.zeros((len(DECISION_VARS),),dtype='object')
        decision_vars_copy      = copy.deepcopy(DECISION_VARS)
        for i,v in enumerate(decision_vars_copy):
            lb   = v['lb']
            ub   = v['ub']
            name = v['name']

            self.decision_variables[i] = self.m.addVar(lb=lb,ub=ub,name=name)

        # Set the objective function
        self.m.setObjective(OBJ(self.decision_variables),GRB.MINIMIZE)

    def set_initial_conditions(self,**kwargs):
        """ """
        self._set_initial_conditions(**kwargs)

        self.T          = 0.05
        self.safety_    = 0
        self.safety_obj = None
        self.xf         = np.zeros(self.x.shape)
        self.xf_dot     = np.zeros(self.x.shape)
        self.xdot       = np.zeros(self.x.shape)
        self.last_x     = self.x
        self.err_max    = self.theta_max - self.theta_min
        c               = np.linalg.norm(self.err_max)
        self.Gamma      = kG * c**2 / (2 * cbf(self.x,self.t)) * np.eye(self.theta_hat.shape[0])
        self.V0         = 1/2 * self.err_max.T @ np.linalg.inv(self.Gamma) @ self.err_max 
        self.Vmax       = self.V0

        self.rcbf = None
        self.smid_queue = collections.deque(maxlen=100)

        self.name       = "LOP"

    def update_filter_2ndOrder(self):
        """
        Updates 2nd-order state filtering scheme according to 1st-order Euler
        derivative approximations.

        INPUTS:
        None

        OUTPUTS:
        None

        """
        # Second-Order Filter
        self.xf_dot = self.xf_dot   + (self.dt * xf_2dot(self.x,self.xf,self.xf_dot,k_e))
        self.xf     = self.xf       + (self.dt * self.xf_dot)

    def update_unknown_parameter_estimates(self):
        """
        """

        # Update theta_hat
        idx                = np.argmin(rcbf(self.x,self.t,self.err_max,self.Gamma))
        rcbf_partialx_term = rcbf_partialx(self.x,self.t)[idx]
        theta_hat_dot      = self.Gamma @ regressor(self.x,self.t) @ rcbf_partialx_term.T

        self.theta_hat     = self.theta_hat + (self.dt * theta_hat_dot)
        self.theta_hat     = np.clip(self.theta_hat,self.theta_min,self.theta_max)

    def update_qp(self,x,t,multiplier=1):
        """
        """
        # Remove old constraints
        if self.performance is not None:
            self.m.remove(self.performance)
        if self.safety is not None:
            self.m.remove(self.safety)

        # Update CLF and CBF
        V    = clf(x,self.xd)
        B    = cbf(x,t)
        rB   = rcbf(x,t,self.err_max,self.Gamma)

        # Update Partial Derivatives of CLF and CBF
        dV   = clf_partial(x,self.xd)
        dBx  = cbf_partialx(x,t)
        dBt  = cbf_partialt(x,t)
        drBx = rcbf_partialx(x,t)
        drBt = rcbf_partialt(x,t)

        # Evaluate Lie Derivatives
        self.LfV  = LfV  = np.dot(dV,f(x))
        self.LgV  = LgV  = np.dot(dV,g(x))
        self.LdV  = LdV  = np.dot(dV,regressor(x,t))
        self.LfB  = LfB  = np.dot(dBx,f(x))
        self.LgB  = LgB  = np.dot(dBx,g(x))
        self.LdB  = LdB  = np.dot(dBx,regressor(x,t))
        self.LfrB = LfrB = np.dot(drBx,f(x))
        self.LgrB = LgrB = np.dot(drBx,g(x))
        self.LdrB = LdrB = np.dot(drBx,regressor(x,t))

        PsiV      = self.get_PsiV()
        PsiB      = self.get_PsiB()

        # CLF (FxTS) Conditions:
        # LfV + LgV*u + LdV*theta <= -c1c*Vc^gamma1c - c2c*Vc^gamma2c + delta0
        try:
            if np.sum(LgV.shape) > 1:
                p = self.m.addLConstr(LfV + np.sum(np.array([gg*self.decision_variables[i] for i,gg in enumerate(LgV)])) + PsiV 
                                      <= 
                                      - c1_c*V**(1 - 1/mu_c) - c2_c*V**(1 + 1/mu_c) + self.decision_variables[2]
                )
                self.performance = p
            else:
                p = self.m.addConstr(LfV + LgV*self.decision_variables[0]
                                     <= 
                                     - c1_c*V**(1 - 1/mu_c) - c2_c*V**(1 + 1/mu_c) + self.decision_variables[1]
                )
                self.performance = p

            # CBF (Safety) Conditions
            # LfrB + LgrB*u + LdrB*theta + drBt - c1e*Ve**gamma1e - c2e*Ve**gamma2e >= -delta1(rB)
            self.safety = []
            for ii,(ff,gg,pp,bb) in enumerate(zip(LfrB,LgrB,PsiB,rB)):
                if np.sum(LgV.shape) > 1:
                    s = self.m.addLConstr(ff + np.sum(np.array([g*self.decision_variables[i] for i,g in enumerate(gg)])) + pp + drBt 
                                          >= 
                                          -self.decision_variables[3+ii]*bb**POWER
                    )
                    self.safety.append(s)
                else:
                    s = self.m.addLConstr(ff + gg*self.decision_variables[0] + pp + drBt 
                                          >= 
                                          -self.decision_variables[2+ii]*bb**POWER
                    )
                    self.safety.append(s)

        except Exception as e:
            traceback.print_exc()
            raise e

        self.clf  = V
        self.cbf  = B
        self.rcbf = rB

    def compute(self,
                not_skip: bool = True) -> np.ndarray:
        """ Computes the solution to the Quadratic Program by first updating
        the unknown parameter estimates and then by calling the parent
        Controller._compute method.

        INPUTS
        ------
        not_skip: (bool) - True -> update parameters, False -> do not update

        RETURNS
        ------
        return parameter for Controller._compute()

        """
        if not_skip:
            # Update Unknown Parameter Estimates
            self.update_unknown_parameter_estimates()

        # Compute Solution
        self._compute()

        return self.sol

    def smid(self):
        """
        """
        D         = 0.1
        D         = D / np.sqrt(2) * np.ones((2,))
        set_a     = self.xf_dot - f(self.last_x) - np.dot(g(self.last_x),self.u)
        self.xdot = f(self.last_x) + np.dot(g(self.last_x),self.u) + regressor(self.last_x,self.t - self.dt) @ np.array([-1,1])

        A  = regressor(self.last_x,self.t - self.dt)
        b1 = set_a - D
        b2 = set_a + D

        thetas_a = np.linalg.solve(A,b1)
        thetas_b = np.linalg.solve(A,b2)

        self.smid_queue.append(np.array([thetas_a,thetas_b]))

        thetas_a = np.array(self.smid_queue)[:,0]
        thetas_b = np.array(self.smid_queue)[:,1]

        new_theta1_min = np.min([thetas_a[:,0],thetas_b[:,0]])
        new_theta1_max = np.max([thetas_a[:,0],thetas_b[:,0]])
        new_theta2_min = np.min([thetas_a[:,1],thetas_b[:,1]])
        new_theta2_max = np.max([thetas_a[:,1],thetas_b[:,1]])

        cond1a    = abs(new_theta1_min - self.theta_max[0]) >= D[0]
        cond1b    = abs(new_theta1_max - self.theta_min[0]) >= D[0]
        cond1c    = new_theta1_min >= self.theta_min[0]
        cond1d    = new_theta1_min <= self.theta_max[0]
        cond1e    = new_theta1_max >= self.theta_min[0]
        cond1f    = new_theta1_max <= self.theta_max[0]
        cond1g    = abs(new_theta1_max - new_theta1_min) >= D[0]
        cond1h    = new_theta1_min < new_theta1_max

        cond2a    = abs(new_theta2_min - self.theta_max[1]) >= D[1]
        cond2b    = abs(new_theta2_max - self.theta_min[1]) >= D[1]
        cond2c    = new_theta2_min >= self.theta_min[1]
        cond2d    = new_theta2_min <= self.theta_max[1]
        cond2e    = new_theta2_max >= self.theta_min[1]
        cond2f    = new_theta2_max <= self.theta_max[1]
        cond2g    = abs(new_theta2_max - new_theta2_min) >= D[1]
        cond2h    = new_theta2_min < new_theta2_max

        if self.t > self.T:
            if cond1a and cond1b and cond1c and cond1d and cond1e and cond1f and cond1g and cond1h:
                self.theta_min[0] = new_theta1_min
                self.theta_max[0] = new_theta1_max
            if cond2a and cond2b and cond2c and cond2d and cond2e and cond2f and cond2g and cond2h:
                self.theta_min[1] = new_theta2_min
                self.theta_max[1] = new_theta2_max

        self.err_max = self.theta_max - self.theta_min

    def compute_w_smid(self,
                       not_skip: bool = True) -> np.ndarray:
        """ Computes the solution to the Quadratic Program by first updating
        the unknown parameter estimates and then by calling the parent
        Controller._compute method.

        INPUTS
        ------
        not_skip: (bool) - True -> update parameters, False -> do not update

        RETURNS
        ------
        return parameter for Controller._compute()

        """
        if not_skip:
            # Update 2nd Order Filter
            self.update_filter_2ndOrder()

            # Execute Set Membership Identification
            self.smid()

            # Update Unknown Parameter Estimates
            self.update_unknown_parameter_estimates()

        self.last_x = self.x

        # Compute Solution
        self._compute()

        return self.sol

    def get_PsiV(self):
        # Impose bounds on Theta set
        theta_V  = np.zeros(self.theta_hat.shape)
        upper    = self.theta_max
        lower    = self.theta_min

        # Determine maximum of LdV*theta_err
        for i,v in enumerate(self.LdV):
            theta_V[i]  = lower[i]*(v*lower[i] > v*upper[i]) + upper[i]*(v*lower[i] <= v*upper[i])

        PsiV = np.dot(self.LdV,theta_V)
        return PsiV

    def get_PsiB(self):
        theta_rB = np.zeros((2,self.theta_hat.shape[0]))
        upper    = self.theta_max
        lower    = self.theta_min
        PsiB     = np.zeros((self.LdrB.shape[0],))

        # Determine minimum of LdrB*theta_err
        for i,bb in enumerate(self.LdrB):
            for j,b in enumerate(bb):
                theta_rB[i,j] = upper[j]*(b*lower[j] > b*upper[j]) + lower[j]*(b*lower[j] <= b*upper[j])

            PsiB[i] = np.dot(self.LdrB[i],theta_rB[i]) 

        return PsiB

