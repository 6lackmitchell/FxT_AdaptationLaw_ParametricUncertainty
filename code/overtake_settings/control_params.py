import numpy as np
from .physical_params import *

###############################################################################
############################# Control Constraints #############################
###############################################################################
U_MAX    = np.array([w_max,u_max])

###############################################################################
############################# FxT-CLF Parameters ##############################
###############################################################################
T_c      = np.array([3,4,10,4]) # sec
mu_c     = 5
c1_c     = np.pi * mu_c / (2*T_c)
c2_c     = np.pi * mu_c / (2*T_c)
gamma1_c = 1 - 1/mu_c
gamma2_c = 1 + 1/mu_c

kx       = 1 / (4**2)
ky       = 100
kt       = 400
kv       = 1 / (1**2)

kV       = 0.00001


###############################################################################
############################ CBF-CLF-QP Parameters ############################
###############################################################################
p0       = 1.0 / U_MAX[0]**2
p1       = 1.0 / U_MAX[1]**2
p2       = 5e8
p3       = 1
p4       = 1
p5       = 1
POWER    = 1
K_cbf    = 1

DECISION_VARS = [{'name':'w',  'lb':-U_MAX[0],'ub':U_MAX[0]},
                 {'name':'u',  'lb':-U_MAX[1],'ub':U_MAX[1]},
                 {'name':'dp', 'lb':0,        'ub':1e+100},
                 {'name':'ds1','lb':1,        'ub':1e+100},
                 {'name':'ds2','lb':1,        'ub':1e+100},
                 {'name':'ds3','lb':1,        'ub':1e+100}]

def OBJ(v):
    return p0*v[0]*v[0] + p1*v[1]*v[1] + p2*v[2]*v[2] + p3*v[3]*v[3]\
              + p3*v[3]*v[3] + p4*v[4]*v[4] + p5*v[5]*v[5]

###############################################################################
############################# General Parameters ##############################
###############################################################################
ERROR    = -999*np.ones((len(DECISION_VARS),))

###############################################################################
################################## Functions ##################################
###############################################################################

### Goal Sets ###
def xd(x,setpoint=-1):
    # Approach Lead Vehicle
    if setpoint == 0:
        vd = 25.0
        # xd = x[4] - (vd*tau*1.33)
        xd = x[4] - (vd*tau*1.25)
        yd = lane_width/2

    # Move into Overtaking Lane
    elif setpoint == 1:
        vd = 28.0
        xd = x[4] - (vd*tau)*0.25
        yd = 3/2*lane_width

    # Overtake Vehicle
    elif setpoint == 2:
        vd = 30.0
        # xd = x[4] + (vd*tau)
        xd = x[4] + (vd*tau*0.75)
        yd = 3/2*lane_width

    # Return to original Lane
    elif setpoint == 3:
        vd = 25.0
        xd = x[4] + (vd*tau*1.25)
        yd = lane_width/2

    # Keep moving forward
    else:
        xd = x[4] + 1000
        yd = lane_width/2
        vd = 25.0

    # td   = np.arctan2(yd - x[1],xd - x[0])
    td   = np.arcsin((yd - x[1]) / (TIME_CONSTANT*x[3]))
    goal = np.array([xd,yd,td,vd])

    return np.concatenate([goal,np.zeros((8,))])

### Overtake Problem ###
def cbf(x,t):
    # # 1st Order
    # cbf1_0 = K_cbf*(x[1] - ER)*(EL - x[1])

    # # 2nd-Order CBF
    # cbf1 = (-2*x[1] + ER + EL)*x[3]*np.sin(x[2])  + cbf1_0

    cbf1a = K_cbf*(x[1] - ER(x))
    cbf1b = K_cbf*(EL(x) - x[1])

    cbf1 = K_cbf*(x[1] - ER(x))*(EL(x) - x[1])

    # # 1st-Order CBF
    # cbf2 = log(1 - (x[3] - speed_limit))
    # cbf3 = log(( (x[0] - x[4]) / (x[3] * np.cos(x[2]) * tau + car_length) )**2 + \
    #            ( (x[1] - x[5]) / safe_lateral_distance)**2 )

    # Speed Limit CBF
    cbf2 = K_cbf * (speed_limit - x[3])

    # Safe Collision Distance
    if x[0] < x[4]:
        cbf3 = K_cbf * (( (x[0] - x[4]) / (x[3] * np.cos(x[2]) * tau + car_length) )**2 + ( (x[1] - x[5]) / safe_lateral_distance)**2 - 1)
    else:
        cbf3 = K_cbf * (( (x[0] - x[4]) / (x[7] * np.cos(x[6]) * tau + car_length) )**2 + ( (x[1] - x[5]) / safe_lateral_distance)**2 - 1)
    
    return np.array([cbf1,cbf2,cbf3])
    return np.array([cbf1a,cbf1b,cbf2,cbf3])

def cbf_partialx(x,t):
    # cbf_partialx1 = \
    # np.array([0,
    #           -2*x[3]*np.sin(x[2]) + K_cbf*(ER + EL - 2*x[1]),
    #           -2*x[1]*x[3]*np.cos(x[2]) + (ER + EL)*x[3]*np.cos(x[2]),
    #           -2*x[1]*np.sin(x[2]) + (ER + EL)*np.sin(x[2])])
    cbf_partialx1a = K_cbf*\
    np.array([0,
              1,
              -dERt(x),
              -dERv(x)])

    cbf_partialx1b = K_cbf*\
    np.array([0,
              1,
              dELt(x),
              dELv(x)])

    cbf_partialx1 = K_cbf*\
    np.array([0,
              ER(x) + EL(x) - 2*x[1],
              x[1]*(dERt(x) + dELt(x)) - ER(x)*dELt(x) - EL(x)*dERt(x),
              x[1]*(dERv(x) + dELv(x)) - ER(x)*dELv(x) - EL(x)*dERv(x)])

    cbf_partialx2 = K_cbf*\
    np.array([0,
              0,
              0,
             -1])

    if x[0] < x[4]:
        cbf_partialx3 = K_cbf*\
        np.array([2 * (x[0] - x[4]) / (x[3]*np.cos(x[2])*tau + car_length)**2,
                  2 * (x[1] - x[5]) / safe_lateral_distance**2,
                  2 * (x[0] - x[4])**2 * (x[3]*tau*np.sin(x[2])) / (x[3]*tau*np.cos(x[2]) + car_length)**(3),
                 -2 * (x[0] - x[4])**2 * (tau*np.cos(x[2])) / (x[3]*tau*np.cos(x[2]) + car_length)**(3)])
    else:
        cbf_partialx3 = K_cbf*\
        np.array([2 * (x[0] - x[4]) / (x[3]*np.cos(x[2])*tau + car_length)**2,
                  2 * (x[1] - x[5]) / safe_lateral_distance**2,
                  0,
                  0])
        
        
    cbf_partialx1a = np.concatenate([cbf_partialx1,np.zeros((8,))])
    cbf_partialx1b = np.concatenate([cbf_partialx1,np.zeros((8,))])
    cbf_partialx1  = np.concatenate([cbf_partialx1,np.zeros((8,))])
    cbf_partialx2  = np.concatenate([cbf_partialx2,np.zeros((8,))])
    cbf_partialx3  = np.concatenate([cbf_partialx3,np.zeros((8,))])

    return np.array([cbf_partialx1,cbf_partialx2,cbf_partialx3])
    return np.array([cbf_partialx1a,cbf_partialx1b,cbf_partialx2,cbf_partialx3])

def cbf_partialt(x,t):
    return 0

def clf(x,xd):
    xbar = x - xd
    V = -1 + kx*xbar[0]**2 + ky*xbar[1]**2 + kt*xbar[2]**2 + kv*xbar[3]**2 \
           + kxv*xbar[0]*xbar[3] + kyt*xbar[1]*xbar[2]
    return kV * V

def clf_partial(x,xd):
    xbar = x - xd
    dV = np.array([2*kx*xbar[0] + kxv*xbar[3],
                   2*ky*xbar[1] + kyt*xbar[2],
                   2*kt*xbar[2] + kyt*xbar[1],
                   2*kv*xbar[3] + kxv*xbar[0]])

    dV = np.concatenate([dV,np.zeros((8,))])
    return kV * dV

def clf(x,xd):
    xbar = x - xd
    V = -1 + kx*xbar[0]**2 + ky*xbar[1]**2 + kt*xbar[2]**2 + kv*xbar[3]**2
    return kV * V

def clf_partial(x,xd):
    xbar = x - xd
    dV = np.array([2*kx*xbar[0],
                   2*ky*xbar[1],
                   2*kt*xbar[2],
                   2*kv*xbar[3]])

    dV = np.concatenate([dV,np.zeros((8,))])

    dV = np.array([2*kx*xbar[0],
                   2*ky*xbar[1],
                   2*kt*xbar[2],
                   2*kv*xbar[3],
                  -2*kx*xbar[0]])

    dV = np.concatenate([dV,np.zeros((7,))])
    return kV * dV

def clf_partiald(x,xd):
    return -clf_partial(x,xd)