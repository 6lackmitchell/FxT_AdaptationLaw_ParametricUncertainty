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

FXTS  = True
ALPHA = 1.3
kG    = 1

###############################################################################
################################## Functions ##################################
###############################################################################

def acbf(x,t,alpha=ALPHA):
    if cbf(x,t)[0] >= alpha:
        acbf1 = alpha ** 2
    else:
        acbf1 = (alpha ** 2) - (cbf(x,t)[0] - alpha) ** 2

    if cbf(x,t)[1] >= alpha:
        acbf2 = alpha ** 2
    else:
        acbf2 = (alpha ** 2) - (cbf(x,t)[1] - alpha) ** 2

    return np.array([acbf1,acbf2])

def acbf_partialx(x,t,alpha=ALPHA):
    if cbf(x,t)[0] >= alpha:
        acbf1_partial = np.array([0,0])
    else:
        acbf1_partial = 2 * cbf_partialx(x,t)[0] * (alpha - cbf(x,t)[0])

    if cbf(x,t)[1] >= alpha:
        acbf2_partial = np.array([0,0])
    else:
        acbf2_partial = 2 * cbf_partialx(x,t)[1] * (alpha - cbf(x,t)[1])

    return np.array([acbf1_partial,acbf2_partial])

def acbf_partialpsi(x,t,alpha=ALPHA):
    return np.zeros(x.shape)

def acbf_partialt(x,t,alpha=ALPHA):
    return cbf_partialt(x,t)


def aclf(x,xd):
    return clf(x,xd)

def aclf_partialx(x,xd):
    return clf_partial(x,xd)

def aclf_partialtheta(x,theta):
    return np.zeros(x.shape)


def f_clf(x,t,theta,Gamma):
    return f(x) + regressor(x,t) @ (theta + Gamma @ aclf_partialtheta(x,theta).T)

def f_cbf(x,t,psi,Gamma):
    return np.array([f(x) + regressor(x,t) @ (p - Gamma @ acbf_partialpsi(x,t,p).T) for p in psi])
    return f(x) + regressor(x,t) @ (psi - Gamma @ acbf_partialpsi(x,t,psi).T)

###############################################################################
#################################### Class ####################################
###############################################################################

class TaylorController(ClfCbfController):

    def __init__(self,u_max):
        super().__init__(u_max) 

        # Create the decision variables
        self.decision_variables = np.zeros((len(DECISION_VARS),),dtype='object')
        decision_vars_copy      = copy.deepcopy(DECISION_VARS)
        for i,v in enumerate(decision_vars_copy):
            name = v['name']

            # Specify the input constraints as an explict model constraint
            if name == 'ux' or name == 'uy':
                v['lb'] = -1e+100
                v['ub'] =  1e+100
            
            lb   = v['lb']
            ub   = v['ub']
            
            self.decision_variables[i] = self.m.addVar(lb=lb,ub=ub,name=name)

        # Set the objective function
        self.m.setObjective(OBJ(self.decision_variables),GRB.MINIMIZE)

        # Relaxation on input constraints
        self.m.addConstr(self.decision_variables[0] <=  U_MAX[0] + self.decision_variables[3])
        self.m.addConstr(self.decision_variables[0] >= -U_MAX[0] - self.decision_variables[3])
        self.m.addConstr(self.decision_variables[1] <=  U_MAX[1] + self.decision_variables[4])
        self.m.addConstr(self.decision_variables[1] >= -U_MAX[1] - self.decision_variables[4])

    def set_initial_conditions(self,**kwargs):
        """ """
        self._set_initial_conditions(**kwargs)

        self.err_max   = self.theta_max - self.theta_min
        c              = np.linalg.norm(self.err_max)
        self.Gamma     = kG * c**2 / (2 * cbf(self.x,self.t)) * np.eye(self.theta_hat.shape[0])

        self.psi_hat   = np.array([self.psi_hat,self.psi_hat]).T

        self.name      = "TAY"

    def update_unknown_parameter_estimates(self):
        """
        """

        # Update theta_hat
        theta_hat_dot  = self.Gamma @ (aclf_partialx(self.x,self.xd) @ regressor(self.x,self.t)).T
        self.theta_hat = self.theta_hat + (self.dt * theta_hat_dot)

        # Update psi_hat
        psi_hat_dot    = self.Gamma @ (-acbf_partialx(self.x,self.t) @ regressor(self.x,self.t)).T
        self.psi_hat   = self.psi_hat + (self.dt * psi_hat_dot)

    def update_qp(self,x,t,multiplier=1):
        """
        """
        # Remove old constraints
        if self.performance is not None:
            self.m.remove(self.performance)
        if self.safety is not None:
            self.m.remove(self.safety)

        # Update CLF and CBF
        self.V   = V    = aclf(x,self.xd)
        self.B   = B    = acbf(x,t)

        # Update Partial Derivatives of CLF and CBF
        self.dV  = dV   = aclf_partialx(x,self.xd)
        self.dBx = dBx  = acbf_partialx(x,t)

        # Impose bounds on Theta set
        thetaV = np.zeros(self.theta_hat.shape)
        psiB   = np.zeros((2,self.theta_hat.shape[0]))
        upper  = self.theta_max
        lower  = self.theta_min

        # Evaluate Lie Derivatives for g, Delta
        self.LgV = LgV  = np.dot(dV,g(x))
        self.LdV = LdV  = np.dot(dV,regressor(x,t))
        self.LgB = LgB  = np.dot(dBx,g(x))
        self.LdB = LdB  = np.dot(dBx,regressor(x,t))

        # Determine supremum of LdV*theta_err
        for i,v in enumerate(LdV):
            thetaV[i]  = lower[i]*(v*lower[i] > v*upper[i]) + upper[i]*(v*lower[i] <= v*upper[i])

        # Determine infimum of LdrB*theta_err
        for i,bb in enumerate(LdB):
            for j,b in enumerate(bb):
                psiB[i,j] = upper[j]*(b*lower[j] > b*upper[j]) + lower[j]*(b*lower[j] <= b*upper[j])

        # Update Lie Derivatives for f based on worst-case theta estimate
        LfV  = np.dot(dV,f_clf(x,t,thetaV,self.Gamma))
        LfB  = np.einsum('ij,ij->i',dBx,f_cbf(x,t,psiB,self.Gamma))

        # CLF (FxTS) Conditions: LfV + LgV*u <= perf_max  + delta0
        if FXTS:
            perf_max = - c1_c*V**(1 - 1/mu_c) - c2_c*V**(1 + 1/mu_c)
        else:
            perf_max = -V

        if np.sum(LgV.shape) > 1:
            p = self.m.addConstr(LfV + np.sum(np.array([gg*self.decision_variables[i] for i,gg in enumerate(LgV)]))
                                 <= 
                                 perf_max + self.decision_variables[2]
            )
            self.performance = p
        else:
            p = self.m.addConstr(LfV + LgV*self.decision_variables[0]
                                 <= 
                                 perf_max + self.decision_variables[1]
            )
            self.performance = p

        # CBF (Safety) Conditions
        # LfB + LgB*u >= 0
        self.safety = []
        for ii,(ff,gg) in enumerate(zip(LfB,LgB)):
            if np.sum(LgV.shape) > 1:
                s = self.m.addLConstr(ff + np.sum(np.array([g*self.decision_variables[i] for i,g in enumerate(gg)])) 
                                      >= 
                                      0
                )
                self.safety.append(s)
            else:
                s = self.m.addLConstr(ff + gg*self.decision_variables[0]
                                      >= 
                                      0
                )
                self.safety.append(s)

        # Store CBF/CLF values
        self.clf  = clf(x,self.xd)
        self.cbf  = cbf(x,t)

        self.aclf  = V
        self.acbf  = B

        self.LfV = LfV
        self.LgV = LgV
        self.LfB = LfB
        self.LgB = LgB

    def compute(self,
                not_skip: bool = True):
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

        self._compute()

        self.safety_check()

        return self.sol

    def safety_check(self):
        if self.sol is not None:
            perf = -self.LfV - self.LgV[0]*self.sol[0] - self.LgV[1]*self.sol[1] - self.aclf + self.sol[2]
            safe = []
            for ii,(ff,gg) in enumerate(zip(self.LfB,self.LgB)):
                safe.append(ff + gg[0]*self.sol[0] + gg[1]*self.sol[1])
            safe = np.array(safe)

            if (safe < 0).any():
                pass
                # print("Time:        {:.3f}".format(self.t))
                # print("Performance: {} = -{} - {}ux - {}uy - {} + {} ".format(perf,self.LfV,self.LgV[0],self.LgV[1],self.aclf,self.sol[2]))
                # print("Safety1:     {} = {} + {}ux + {}uy".format(safe[0],self.LfB[0],self.LgB[0,0],self.LgB[0,1]))
                # print("Safety2:     {} = {} + {}ux + {}uy".format(safe[1],self.LfB[1],self.LgB[1,0],self.LgB[1,1]))
                # print("(ux,uy): ({},{})".format(self.sol[0],self.sol[1]))



