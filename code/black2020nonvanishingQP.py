import os
import copy
import time
import pickle
import winsound
import numpy as np
import gurobipy as gp

from gurobipy import GRB
from scipy import sparse
from clfcbfcontroller import ClfCbfController

FOLDER = os.getcwd().split('\\')[-1]
if FOLDER == 'overtake':
    from overtake_settings import *
elif FOLDER == 'simple':
    TIME_HEADWAY       = 0
    ONCOMING_FREQUENCY = 0
    from simple_settings import *

###############################################################################
#################################### Class ####################################
###############################################################################

class BlackController(ClfCbfController):

    @property
    def safe_to_pass(self):

        # time_needed = 21.85 # Theta_Max = 1
        # time_needed = 22.67 # Theta_Max = 2
        # time_needed = 24.60 # Theta_Max = 4
        # time_needed = 27.30 # Theta_Max = 6
        # time_needed = 31.40 # Theta_Max = 8
        time_needed = 39.20 # Theta_Max = 10

        if time_needed < TIME_HEADWAY or self.safe:
            return True
        elif time_needed > ONCOMING_FREQUENCY:
            return False
        elif self.t > TIME_HEADWAY:
            self.setpoint = self.setpoint + 1
            self.safe = True
            print("Safe to Pass! New Goal: ",self.setpoint)
            return True
    
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

        self.setpoint   = 0
        self.dV         = None
        self.clf        = None
        self.td         = None

        self.name       = "BLA"
        self.safe       = False

    def update_qp(self,x,t):
        """
        """
        # Remove old constraints
        if self.performance is not None:
            self.m.remove(self.performance)
        if self.safety is not None:
            for s in self.safety:
                self.m.remove(s)

        # Update Goal
        reached = False
        if self.safe_to_pass or FOLDER == 'simple':
            if clf(x,self.xd) < 0.0:
                print("Goal {} Reached! Time = {:.3f}".format(self.setpoint,self.t))
                self.setpoint = self.setpoint + 1
                frequency = 1500  # Set Frequency To 2500 Hertz
                duration  = 100   # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                time.sleep(0.1)
                winsound.Beep(frequency, duration)

        # Assign c1, c2 for FxTS
        try:
            c1        = c1_c[self.setpoint]
            c2        = c2_c[self.setpoint]
        except TypeError:
            c1        = c1_c
            c2        = c1_c

        # Update CLF and CBF
        self.V   = V   = np.max([clf(x,self.xd),0]);
        self.B   = B   = cbf(x,t);

        # Update Partial Derivatives of CLF and CBF
        self.dV  = dV  = clf_partial(x,self.xd)
        self.dVd = dVd = clf_partiald(x,self.xd)
        self.dBx = dBx = cbf_partialx(x,t)
        self.dBt = dBt = cbf_partialt(x,t)

        # Evaluate Lie Derivatives
        phi_max              = self.get_phi_max()
        LfV1                 = np.dot(dV,f(x))
        self.LgV    = LgV    = np.dot(dV,g(x))
        self.LdVmax = LdVmax = self.max_LdV(phi_max,dV)
        self.LfB    = LfB    = np.dot(dBx,f(x))
        self.LgB    = LgB    = np.dot(dBx,g(x))
        self.LdB    = LdB    = self.min_LdB(phi_max,dBx)

        # Lie Derivative wrt Time-Varying Goal
        LfVd           = np.dot(dVd,fd(x,self.xd))
        self.LfV = LfV = LfV1 + LfVd

        # Configure CLF and CBF constraints
        try:
            # CLF (FxTS) Conditions: LfV + LgV*u + LdV*theta <= -c1c*Vc^gamma1c - c2c*Vc^gamma2c + delta0
            if np.sum(LgV.shape) > 1:
                p = self.m.addLConstr(LfV + np.sum(np.array([gg*self.decision_variables[i] for i,gg in enumerate(LgV)]))
                                      <= 
                                      - c1*V**(1 - 1/mu_c) - c2*V**(1 + 1/mu_c) - LdVmax + self.decision_variables[2]
                )
                self.performance = p
            else:
                p = self.m.addLConstr(LfV + LgV*self.decision_variables[0]
                                      <= 
                                      - c1*V**(1 - 1/mu_c) - c2*V**(1 + 1/mu_c) - LdVmax + self.decision_variables[1]
                )
                self.performance = p

            # CBF (Safety) Conditions: LfrB + LgrB*u + LdrB*theta + >= -delta1(rB)
            self.safety  = []
            for ii,(ff,gg,dd,bb) in enumerate(zip(LfB,LgB,LdB,B)):
                if np.sum(LgV.shape) > 1:
                    s = self.m.addLConstr(ff + np.sum(np.array([g*self.decision_variables[i] for i,g in enumerate(gg)])) + dd
                                          >= 
                                          -self.decision_variables[3+ii]*bb**POWER
                    )
                    self.safety.append(s)
                else:
                    s = self.m.addLConstr(ff + gg*self.decision_variables[0] + dd
                                          >= 
                                          -self.decision_variables[2+ii]*bb**POWER
                    )
                    self.safety.append(s)
        
        except:
            self.report_msg()
            raise ValueError("GurobiError")

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
        # Compute Solution
        self._compute()

        return self.sol

    def max_LdV(self,pmax,dV):
        """ Computes maximum of (dV/dx)(Delta)(theta)

        INPUTS:
        pmax: array (n,p) - zeros, will be populated within this function
        dV:   array (1,n) - partial derivative of V wrt x

        OUTPUTS:
        m: float - maximum of LdV
        """

        # Determine maximum of LdV*(theta_hat + theta_err)
        for i,v in enumerate(dV):
            pmax[i]  = -pmax[i]*(v*-pmax[i] > v*pmax[i]) + pmax[i]*(v*-pmax[i] <= v*pmax[i])
        
        m = np.dot(dV,pmax)
        return m


    def min_LdB(self,pmax,dBx):
        """ Computes supremum of (dV/dx)(Delta)(theta)

        INPUTS:
        pmax: array (n,p) - max/min allowable values of theta
        dV:   array (m,n) - partial derivative of m CBFs wrt x

        OUTPUTS:
        m: float - minimum of LdB
        """

        # Initialize phi array
        if np.sum(dBx.shape) > dBx.shape[0]:
            phi_rB = np.zeros((dBx.shape[0],pmax.shape[0]))
        else:
            phi_rB = np.zeros(pmax.shape)

        # Determine minimum of LdB*(theta_hat + theta_err) for each CBF
        for i,b in enumerate(dBx):
            if np.sum(b.shape) > 0:
                for j,bb in enumerate(b):
                    phi_rB[i,j] = -pmax[j]*(bb*pmax[j] > bb*-pmax[j]) + pmax[j]*(bb*pmax[j] <= bb*-pmax[j])
            else:
                phi_rB[i] = -pmax[i]*(b*pmax[i] > b*pmax[i]) + pmax[i]*(b*pmax[i] <= b*-pmax[i])

        m = np.einsum('ij,ij->i',dBx,phi_rB)
        return m

    def report_msg(self):
        """ Prints out important state and controller information.

        INPUTS:
        None

        OUTPUTS:
        None

        """
        print("X:   {}".format(self.x))
        print("LfV: {}".format(self.LfV))
        print("LgV: {}".format(self.LgV))
        print("LdV: {}".format(self.LdVmax))
        print("V:   {}".format(V))

    def get_phi_max(self):
        """ Obtains the worst-case effect of uncertainty.

        INPUTS:
        None

        OUTPUTS:
        phi_max: array (1,p): worst-case disturbance action

        """

        theta_max  = self.theta_max[0]

        if FOLDER == 'overtake':
            scale1     = regressor(self.x,0)[0,0]
            scale2     = regressor(self.x,0)[1,1]
            phi_max    = np.array([0,0,0,0,scale1*theta_max, scale2*theta_max,0,0,0,0,0,0])
        elif FOLDER == 'simple':
            phi_max = regressor(self.x,0) @ (theta_max * np.ones((self.x.shape[0],)))

        return phi_max
