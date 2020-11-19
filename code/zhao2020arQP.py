import os
import copy
import numpy as np

from gurobipy import GRB
from scipy import sparse
from clfcbfcontroller import ClfCbfController

FOLDER = os.getcwd().split('\\')[-1]
if FOLDER == 'overtake':
    from overtake_settings import *
elif FOLDER == 'simple':
    from simple_settings import *

FXTS = True
###############################################################################
#################################### Class ####################################
###############################################################################

class ZhaoController(ClfCbfController):

    def __init__(self,u_max):
        super().__init__(u_max) 

        # Create the decision variables
        self.decision_variables = np.zeros((len(DECISION_VARS),),dtype='object')
        decision_vars_copy      = copy.deepcopy(DECISION_VARS)
        for i,v in enumerate(decision_vars_copy):
            name = v['name']
            lb   = v['lb']
            ub   = v['ub']
            
            self.decision_variables[i] = self.m.addVar(lb=lb,ub=ub,name=name)

        # Set the objective function
        self.m.setObjective(OBJ(self.decision_variables),GRB.MINIMIZE)

    def set_initial_conditions(self,**kwargs):
        """ """
        self._set_initial_conditions(**kwargs)

        self.a    = 1
        self.T    = 0.001
        self.xhat = self.x
        self.xerr = np.zeros((self.x.shape))
        self.dhat = np.zeros((self.x.shape))

        self.name       = "ZHA"

    def estimate_state(self):
        """
        """
        # Update State Estimation
        xhat_dot = f(self.x) + np.dot(g(self.x),self.u) + self.dhat - self.a * self.xerr 
        self.xhat = self.xhat + (self.dt * xhat_dot)

        # Update State Estimation Error
        self.xerr = self.xhat - self.x

    def update_unknown_disturbance(self):
        """
        """
        # Update Disturbance Estimate
        if self.t % self.T:
            self.xerr_iT = self.xerr
            self.dhat = -self.a / (np.exp(self.a*self.T) - 1) * self.xerr_iT

    def update_qp(self,x,t,multiplier=1):
        """
        """
        # Remove old constraints
        if self.performance is not None:
            self.m.remove(self.performance)
        if self.safety is not None:
            self.m.remove(self.safety)

        # Uncertainty Bound
        theta_max = self.theta_max[0]
        
        # Gamma Function Parameters
        xi         = 2.0
        XminusY    = 10.0
        TminusTau  = 5.0
        scale1     = regressor(x,0.125)[0,0]#1.25*(1+np.sin(np.pi/4)**2)
        scale2     = regressor(x,0.125)[1,1]#1.25*(1+np.cos(np.pi/4)**2)
        maxDist    = np.sqrt((scale1*theta_max)**2 + (scale2*theta_max)**2)
        ld         = maxDist / XminusY * xi
        lt         = maxDist / TminusTau * xi
        bd         = maxDist * xi
        theta_zhao = ld * XminusY + bd
        phi        = np.linalg.norm(U_MAX) + theta_zhao
        eta        = lt + ld*phi

        if self.t < self.T:
            gam = theta_zhao
        else:
            gam = 2 * np.sqrt(eta) * eta * self.T + np.sqrt(self.x.shape[0]) * (1 - np.exp(-self.a * self.T)) * theta_zhao

        # Update CLF and CBF
        V    = clf(x,self.xd)
        B    = cbf(x,t)

        # Update Partial Derivatives of CLF and CBF
        dV   = clf_partial(x,self.xd)
        dBx  = cbf_partialx(x,t)

        # Evaluate Lie Derivatives for f, g, Delta
        LfV   = np.dot(dV,f(x))
        LgV   = np.dot(dV,g(x))
        LdV   = np.dot(dV,self.dhat)
        LgamV = np.linalg.norm(dV)*gam
        LfB   = np.dot(dBx,f(x))
        LgB   = np.dot(dBx,g(x))
        LdB   = np.dot(dBx,self.dhat)
        LgamB = np.linalg.norm(dBx)*gam

        # CLF (FxTS) Conditions: LfV + LgV*u <= perf_max  + delta0
        if FXTS:
            perf_max = - c1_c*V**(1 - 1/mu_c) - c2_c*V**(1 + 1/mu_c)
        else:
            perf_max = -V

        # CLF (FxTS) Conditions:
        # LfV + LgV*u <= -V  + delta0
        if np.sum(LgV.shape) > 1:
            p = self.m.addConstr(LfV + LgV[0]*self.decision_variables[0] + LgV[1]*self.decision_variables[1] + LdV - LgamV 
                                 <= 
                                 perf_max + self.decision_variables[2]
            )
            self.performance = p
        else:
            p = self.m.addConstr(LfV + LgV*self.decision_variables[0] + LdV - LgamV 
                                 <= 
                                 perf_max + self.decision_variables[2]
            )
            self.performance = p
        
        # CBF (Safety) Conditions
        # LfB + LgB*u >= 0
        self.safety = []
        for ii,(ff,gg,dd,bb) in enumerate(zip(LfB,LgB,LdB,B)):
            if np.sum(LgV.shape) > 1:
                s = self.m.addConstr(ff + gg[0]*self.decision_variables[0] + gg[1]*self.decision_variables[1] + dd - LgamB 
                                     >=
                                     - self.decision_variables[3+ii]*bb**POWER
                )
                self.safety.append(s)
            else:
                s = self.m.addLConstr(ff + gg*self.decision_variables[0] + dd - LgamB
                                     >=
                                     - self.decision_variables[3+ii]*bb**POWER
                )
                self.safety.append(s)

        # Store CBF/CLF values
        self.clf  = V
        self.cbf  = B
        self.dV   = dV 

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
            self.estimate_state()
            self.update_unknown_disturbance()

        return self._compute()
