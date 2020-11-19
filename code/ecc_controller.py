import os
import copy
import time
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
    from simple_settings import *

###############################################################################
##################### FxT Parameter Estimation Parameters #####################
############################################################################### 

mu_e     = 5
c1_e     = 50
c2_e     = 50
k_e      = 0.001
l_e      = 100
gamma1_e = 1 - 1/mu_e
gamma2_e = 1 + 1/mu_e
T_e      = 1 / (c1_e * (1 - gamma1_e)) + 1 / (c2_e * (gamma2_e - 1))
kG       = 1.2 # Must be greater than 1

def rcbf(x,t,err,Gamma):
    return cbf(x,t) - 1/2 * err.T @ np.linalg.inv(Gamma) @ err

def rcbf_partialx(x,t):
    return cbf_partialx(x,t)

def rcbf_partialt(x,t):
    return cbf_partialt(x,t)

###############################################################################
#################################### Class ####################################
###############################################################################

class EccController(ClfCbfController):

    @property
    def T_e(self):
        arg1 = np.sqrt(c2_e) * self.V0_T**(1/mu_e)
        arg2 = np.sqrt(c1_e)
        return mu_e / np.sqrt(c1_e*c2_e) * np.arctan2(arg1,arg2)

    @property
    def xd(self):
        if FOLDER == 'overtake':

            goal = xd(self.x,self.setpoint)
            self.td = goal[2]

        elif FOLDER == 'simple':
            goal = np.array([0,0])

        return goal
    
    
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

        # This is more conservative than necessary -- could be fcn of initial estimate
        self.err_max    = self.theta_max - self.theta_min
        c               = np.linalg.norm(self.err_max)
        self.Gamma      = kG * c**2 / (2 * np.min(cbf(self.x,self.t))) * np.eye(self.theta_hat.shape[0])
        self.V0_T       = np.inf
        self.V0         = 1/2 * self.err_max.T @ np.linalg.inv(self.Gamma) @ self.err_max 
        self.Vmax       = self.V0
        self.eta        = self.Vmax
        self.safety_    = 0
        self.safety_obj = None
        self.setpoint   = 0
        self.clf        = None
        self.td         = 0

        # print("NormErrMax: {}".format(np.linalg.norm(self.err_max)))
        # print("ErrMax: {}".format(self.err_max))
        # print("Gamma:  {}".format(self.Gamma))
        # print("CBF:    {}".format(cbf(self.x,self.t)))
        # print("Vmax:   {}".format(self.Vmax))
        # print("Alternate Vmax: {}".format(1/2 * self.err_max.T @ np.linalg.inv(100/1.2*self.Gamma) @ self.err_max))

        self.xf        = self.x
        self.xf_dot    = np.zeros(self.x.shape)
        self.phif      = np.zeros(f(self.x).shape)
        self.phif_dot  = np.zeros(f(self.x).shape)
        self.Phif      = np.zeros(regressor(self.x,self.t).shape)
        self.Phif_dot  = np.zeros(regressor(self.x,self.t).shape)
        self.P         = np.zeros(np.dot(regressor(self.x,self.t).T,regressor(self.x,self.t)).shape)
        self.Q         = np.zeros(self.theta_hat.shape)

        self.name       = "ECC"
        self.update_error_bounds(0)

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
        self.xf_dot   = self.xf_dot   + (self.dt * xf_2dot(self.x,self.xf,self.xf_dot,k_e))
        self.phif_dot = self.phif_dot + (self.dt * phif_2dot(self.x,self.u,self.phif,self.phif_dot,k_e))
        self.Phif_dot = self.Phif_dot + (self.dt * Phif_2dot(regressor(self.x,self.t),self.Phif,self.Phif_dot,k_e))

        self.xf   = self.xf   + (self.dt * self.xf_dot)
        self.phif = self.phif + (self.dt * self.phif_dot)
        self.Phif = self.Phif + (self.dt * self.Phif_dot)

    def update_filter_1stOrder(self):
        """
        Updates 2nd-order state filtering scheme according to 1st-order Euler
        derivative approximations.

        INPUTS:
        None

        OUTPUTS:
        None

        """

        # First-Order Filter
        self.xf_dot   = (self.x - self.xf) / k_e
        self.phif_dot = (f(self.x) + np.dot(g(self.x),self.u) - self.phif) / k_e
        self.Phif_dot = (regressor(self.x,self.t) - self.Phif) / k_e

        self.xf   = self.xf   + (self.dt * self.xf_dot)
        self.phif = self.phif + (self.dt * self.phif_dot)
        self.Phif = self.Phif + (self.dt * self.Phif_dot)

    def update_auxiliaries(self):
        """ Updates the auxiliary matrix and vector for the filtering scheme.

        INPUTS:
        None

        OUTPUTS:
        float -- minimum eigenvalue of P matrix

        """
        Pdot = -l_e * self.P + np.dot(self.Phif.T,self.Phif)
        Qdot = -l_e * self.Q + np.dot(self.Phif.T,(self.xf_dot - self.phif))

        self.P = self.P + (self.dt * Pdot)
        self.Q = self.Q + (self.dt * Qdot)

        return np.min(np.linalg.eig(self.P)[0])

    def update_unknown_parameter_estimates(self):
        """
        """
        tol = 0.0#1e-10
        self.update_filter_2ndOrder()
        self.update_auxiliaries()

        # Compute quantities for theta_hat_dot
        W    = self.P @ self.theta_hat - self.Q
        Pinv = np.linalg.inv(self.P)
        pre  = self.Gamma @ W / (W.T @ Pinv.T @ W)
        V    = (1/2 * W.T @ Pinv.T @ np.linalg.inv(self.Gamma) @ Pinv @ W)

        # Update theta_hat
        theta_hat_dot  = pre * (-c1_e * V**gamma1_e - c2_e * V**gamma2_e)
        if np.linalg.norm(theta_hat_dot) >= tol:
            self.theta_hat = self.theta_hat + (self.dt * theta_hat_dot)
        else:
            print("No Theta updated: Time = {}sec".format(self.t))

        self.theta     = Pinv @ self.Q

    def update_error_bounds(self,t):
        """
        """
        # Update Max Error Quantities
        arc_tan     = np.arctan2(np.sqrt(c2_e) * self.V0**(1/mu_e),np.sqrt(c1_e))
        tan_arg     = -np.min([t,self.T_e]) * np.sqrt(c1_e * c2_e) / mu_e + arc_tan
        Vmax        = (np.sqrt(c1_e / c2_e) * np.tan(np.max([tan_arg,0]))) ** mu_e
        self.Vmax   = np.clip(Vmax,0,np.inf)

        # Update eta
        self.eta    = self.Vmax

        # Update etadot
        edot_coeff  = -np.sqrt(2 * np.max(self.Gamma) * c1_e **(mu_e/2 + 1) / c2_e **(mu_e/2 - 1))
        self.etadot = edot_coeff * np.tan(np.max([tan_arg,0]))**(mu_e/2 - 1) / np.cos(np.max([tan_arg,0]))**2

        # if self.t % 0.1 <= 0.0001:
        #     print(self.Vmax)
        #     print(self.err_max)
        #     print(self.theta_hat)

        # print("arctan: {}".format(arc_tan))
        # print("tan(arctan): {}".format(np.tan(arc_tan)))
        # print("tanarg: {}".format(tan_arg))
        # print("Vmax:   {}".format(Vmax))
        # Update max theta_tilde
        self.err_max = np.clip(np.sqrt(2*np.diagonal(self.Gamma)*self.Vmax),0,self.theta_max-self.theta_min)
        # print("Emax: {}".format(self.err_max))

    def update_qp(self,x,t,multiplier=1):
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
        if clf(x,self.xd) < 0.0:#self.epsilon:
            # if FOLDER == 'overtake':
            #     if self.setpoint != 2:
            #         reached = True
            #     else:
            #         xd = self.xd
            #         xd[0] = x[4] + (xd[3]*tau)
            #         if clf(x,xd) < 0.0:
            #             reached = True

            # if reached:
            print("Goal Reached! Time = {:.3f}".format(self.t))
            self.setpoint = self.setpoint + 1
            frequency = 1500  # Set Frequency To 2500 Hertz
            duration  = 100  # Set Duration To 1000 ms == 1 second
            winsound.Beep(frequency, duration)
            time.sleep(0.1)
            winsound.Beep(frequency, duration)

        # Assign c1, c2
        try:
            c1        = c1_c[self.setpoint]
            c2        = c2_c[self.setpoint]
        except IndexError:
            c1        = np.pi * mu_c / (2*1000)
            c2        = np.pi * mu_c / (2*1000)
        except TypeError:
            c1        = c1_c
            c2        = c1_c

        # Update CLF and CBF
        V         = clf(x,self.xd);
        B         = cbf(x,t);                          
        rB        = rcbf(x,t,self.err_max,self.Gamma); 

        # Update Partial Derivatives of CLF and CBF
        dV        = clf_partial(x,self.xd)
        dVd       = clf_partiald(x,self.xd)
        dBx       = cbf_partialx(x,t)
        dBt       = cbf_partialt(x,t)
        drBx      = rcbf_partialx(x,t)
        drBt      = rcbf_partialt(x,t);     

        # Evaluate Lie Derivatives
        LfV       = np.dot(dV,f(x))
        LgV       = np.dot(dV,g(x))
        LdV       = np.dot(dV,regressor(x,t))
        LfB       = np.dot(dBx,f(x))
        LgB       = np.dot(dBx,g(x))
        LdB       = np.dot(dBx,regressor(x,t))
        LfrB      = np.dot(drBx,f(x))
        LgrB      = np.dot(drBx,g(x))
        LdrB      = np.dot(drBx,regressor(x,t))

        # Lie Derivative wrt Time-Varying Goal
        LfVd      = np.dot(dVd,fd(x,self.xd))
        LfV       = LfV + LfVd

        # Impose bounds on Theta set
        theta_V      = np.zeros(self.theta_hat.shape)
        if np.sum(B.shape) > 0:
            theta_rB = np.zeros((B.shape[0],self.theta_hat.shape[0]))
        else:
            theta_rB = np.zeros(self.theta_hat.shape)
        upper    = np.clip(self.theta_hat+self.err_max,self.theta_min,self.theta_max)
        lower    = np.clip(self.theta_hat-self.err_max,self.theta_min,self.theta_max)

        # print("Lower: {}".format(lower))
        # print("Upper: {}".format(upper))

        # Determine supremum of LdV*theta_err
        for i,v in enumerate(LdV):
            theta_V[i]  = lower[i]*(v*lower[i] > v*upper[i]) + upper[i]*(v*lower[i] <= v*upper[i])

        # Determine infimum of LdrB*theta_err
        for i,b in enumerate(LdrB):
            if np.sum(b.shape) > 0:
                for j,bb in enumerate(b):
                    theta_rB[i,j] = upper[j]*(bb*lower[j] > bb*upper[j]) + lower[j]*(bb*lower[j] <= bb*upper[j])
            else:
                theta_rB[i] = upper[i]*(b*lower[i] > b*upper[i]) + lower[i]*(b*lower[i] <= b*upper[i])

        # CLF (FxTS) Conditions:
        # LfV + LgV*u - LdV*theta <= -c1c*Vc^gamma1c - c2c*Vc^gamma2c + delta0
        try:
            if np.sum(LgV.shape) == 2:
                self.performance = self.m.addConstr(LfV + LgV[0]*self.decision_variables[0] + LgV[1]*self.decision_variables[1] + np.dot(LdV,theta_V) <= - c1*V**(1 - 1/mu_c) - c2*V**(1 + 1/mu_c) + self.decision_variables[2])
            else:
                self.performance = self.m.addConstr(LfV + LgV*self.decision_variables[0] + np.dot(LdV,theta_V) <= - c1*V**(1 - 1/mu_c) - c2*V**(1 + 1/mu_c) + self.decision_variables[1])
        except:
            print("X: {}".format(self.x))
            print("LfV: {}".format(LfV))
            print("LgV: {}".format(LgV))
            print("LdV: {}".format(np.dot(LdV,theta_V)))
            print("V:   {}".format(- c1*V**(1 - 1/mu_c) - c2*V**(1 + 1/mu_c)))
            print("c1,c2: {},{}".format(c1,c2))
            print("Vv:  {}".format(V))
            raise ValueError("GurobiError")

        self.safety  = []
        for ii,(ff,gg,dd,tt,bb) in enumerate(zip(LfrB,LgrB,LdrB,theta_rB,rB)):
            if np.sum(LgV.shape) == 2:
                # if ii < 2:
                #     self.safety.append(self.m.addLConstr(ff + gg[0]*self.decision_variables[0] + gg[1]*self.decision_variables[1] + np.dot(dd,tt) + drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e)) >= -multiplier*bb**POWER))
                # else:
                    # self.safety.append(self.m.addConstr(ff + gg[0]*self.decision_variables[0] + gg[1]*self.decision_variables[1] + np.dot(dd,tt) + drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e)) >= -self.decision_variables[3+ii]*bb**POWER))
                    self.safety.append(self.m.addConstr(ff + gg[0]*self.decision_variables[0] + gg[1]*self.decision_variables[1] + np.dot(dd,tt) + drBt - (np.trace(np.linalg.inv(self.Gamma)) * self.eta * self.etadot) >= -self.decision_variables[3+ii]*bb**POWER))
            else:
                # print("Safety{}: {} >= {}u".format(ii,ff + np.dot(dd,tt) + drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e)) + multiplier*bb,-gg))
                # print("LfrB:     {}".format(ff))
                # print("LgrB:     {}".format(gg))
                # print("LdrB:     {}".format(np.dot(dd,tt)))
                # print("drBt:     {}".format(drBt))
                # print("AdaV:     {}".format(- (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e))))
                # print("rB:       {}".format(bb**POWER))
                self.safety.append(self.m.addLConstr(ff + gg*self.decision_variables[0] + np.dot(dd,tt) + drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e)) >= -self.decision_variables[2]*bb**POWER))
        # self.safety = self.m.addConstr(LfrB + LgrB[:,0]*self.decision_variables[0] + LgrB[:,1]*self.decision_variables[1] + np.einsum('ij,ij->i',LdrB,theta_rB) + drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e)) >= -self.decision_variables[3]*rB**POWER)
        # if self.sol is not None:
        #     for ii,(ff,gg,dd,tt,bb) in enumerate(zip(LfrB,LgrB,LdrB,theta_rB,rB)):
        #         self.safety_.append(ff + gg[0]*self.decision_variables[0] + gg[1]*self.decision_variables[1] + np.dot(dd,tt) + drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e)) >= -self.decision_variables[3]*bb**POWER))
        
        #     self.safety_ = LfrB + LgrB[:,0]*self.sol[0] + LgrB[:,1]*self.sol[1] + np.einsum('ij,ij->i',LdrB,theta_rB) + drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e)) + self.sol[3]*rB**POWER
        #     self.safety_obj = {'LgrB':LgrB,'ArB':rB**POWER,'remainder':LfrB + np.einsum('ij,ij->i',LdrB,theta_rB) + drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e))}
        #     if self.t % 0.1 <= 0.00001 and self.sol is not None:
        # # print("Multiplier: {}".format(multiplier))
        #         print("Safety{}: {} >= {}u".format(ii,ff + np.dot(dd,tt) + drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e)) + multiplier*bb,-gg))
        #         print("LfrB:     {}".format(ff))
        #         print("LgrB:     {}".format(gg))
        #         print("LdrB:     {}".format(np.dot(dd,tt)))
        #         print("drBt:     {}".format(drBt))
        #         print("AdaV:     {}".format(- (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e))))
        #         print("rB:       {}".format(bb**POWER))
        #         if bb < 0:
        #             print("SAFETY VIOLATED SEE ABOVE")

        self.clf  = V
        self.cbf  = B
        self.rcbf = rB

        self.V        = V
        self.B        = B
        self.rB       = rB
        self.dV       = dV
        self.dBx      = dBx
        self.dBt      = dBt
        self.drBx     = drBx
        self.drBt     = drBt
        self.LfV      = LfV
        self.LgV      = LgV
        self.LdV      = LdV
        self.LfB      = LfB
        self.LgB      = LgB
        self.LdB      = LdB
        self.LfrB     = LfrB
        self.LgrB     = LgrB
        self.LdrB     = LdrB
        self.theta_V  = theta_V
        self.theta_rB = theta_rB

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

            # Update theta_tilde upper/lower bounds
            self.update_error_bounds(self.t)

        # Compute Solution
        self._compute()

        # Have observed some solutions in violation of safety constraint
        self.safety_check()

        return self.sol

    def safety_check(self):
        """
        """
        return
        tolerance = 1e-7

        if self.sol is not None:
            if np.isnan(self.sol).any() or (self.sol > 1e10).any():
                nan_idx = np.where(np.isnan(self.sol))
                print(nan_idx)
                self.sol[nan_idx] = 0

            safe_check = np.zeros(self.rB.shape)
            for ii,(ff,gg,dd,tt,bb) in enumerate(zip(self.LfrB,self.LgrB,self.LdrB,self.theta_rB,self.rB)):
                safe_check[ii] = ff + gg[0]*self.sol[0] + gg[1]*self.sol[1] + np.dot(dd,tt) + self.drBt - (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e)) + self.sol[3+ii]*bb**POWER

            idx = np.where(safe_check <= -tolerance)[0]
            # if np.sum(idx.shape) > 0:
            #     print("Solution Violates Safety Condition(s) ",idx)

            if 2 in idx:
                before = self.sol[1]
                term = (c1_e*self.Vmax**(1 - 1/mu_e) + c2_e*self.Vmax**(1 + 1/mu_e))
                u = (-ff - gg[0]*self.sol[0] - np.dot(dd,tt) - self.drBt + term - self.sol[3]*bb**POWER)/gg[1]
                self.sol[1] = np.sign(u) * np.min([abs(u),U_MAX[1]])
                # print("Delta: ",self.sol[1] - before)

        # if self.safety_obj is not None:
        #     if self.safety_ < 0:
        #         u = -(self.sol[3]*self.safety_obj['ArB'] + self.safety_obj['remainder']) / self.safety_obj['LgrB']
        #         self.sol[0] = np.sign(u) * np.min([abs(u),U_MAX])