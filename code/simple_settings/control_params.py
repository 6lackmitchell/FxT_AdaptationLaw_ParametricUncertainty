import numpy as np

###############################################################################
############################# Control Constraints #############################
###############################################################################
U_MAX    = np.array([2.5,2.5])

###############################################################################
############################# FxT-CLF Parameters ##############################
###############################################################################
T_c      = 4 # sec
mu_c     = 5
c1_c     = np.pi * mu_c / (2*T_c)
c2_c     = np.pi * mu_c / (2*T_c)
gamma1_c = 1 - 1/mu_c
gamma2_c = 1 + 1/mu_c
kV       = 1

###############################################################################
############################ CBF-CLF-QP Parameters ############################
###############################################################################
p0       = 1 # 0.5
p1       = 1
p2       = 100
p3       = 10
p4       = 10
q1       = 0
POWER    = 1
kB       = 1

DECISION_VARS = [{'name':'ux', 'lb':-U_MAX[0],'ub':U_MAX[0]},
                 {'name':'uy', 'lb':-U_MAX[1],'ub':U_MAX[1]},
                 {'name':'d0', 'lb':0,     'ub':1e+100},
                 {'name':'d1', 'lb':1,     'ub':1e+100},
                 {'name':'d2', 'lb':1,     'ub':1e+100}]
                 
def OBJ(v):
    return 1/2*(p0*v[0]*v[0] + p1*v[1]*v[1] + p2*v[2]*v[2] + p3*v[3]*v[3] + p4*v[4]*v[4])

###############################################################################
############################# General Parameters ##############################
###############################################################################
ERROR    = -999*np.ones((len(DECISION_VARS),))

###############################################################################
################################## Functions ##################################
###############################################################################

# ### 1D Time-Varying Problem ###
# def cbf(x,t):
#     return x[0] + 5 - t

# def cbf_partialx(x,t):
#     return np.array([1,0])

# def cbf_partialt(x,t):
#     return -1

# def clf(x):
#     return kV * x[0]**2

# def clf_partial(x):
#     return np.array([2*kV*x[0],0])

# ### 1D Problem ###
# def cbf(x,t):
#     return kB*x[0]

# def cbf_partialx(x,t):
#     return kB*np.array([1,0])

# def cbf_partialt(x,t):
#     return 0

# def clf(x):
#     return kV * x[0]**2

# def clf_partial(x):
#     return np.array([2*kV*x[0],0])

### 1D Problem w/ Barriers ###
a = 1.0
b = 4.99
x1 = x2 = 1.0
y1 = -6.0; y2 = 4.0

def cbf(x,t):
    cbf1 = ((x[0] - x1)/a)**2 + ((x[1] - y1)/b)**2 - 1
    cbf2 = ((x[0] - x2)/a)**2 + ((x[1] - y2)/b)**2 - 1
    return kB*np.array([cbf1,cbf2])

def cbf_partialx(x,t):
    cbf1_partialx = np.array([2*(x[0] - x1)/a**2,
                              2*(x[1] - y1)/b**2])
    cbf2_partialx = np.array([2*(x[0] - x2)/a**2,
                              2*(x[1] - y2)/b**2])
    return kB*np.array([cbf1_partialx,cbf2_partialx])

def cbf_partialt(x,t):
    return 0

def clf(x,xd):
    return kV * (x[0]**2 + x[1]**2)

def clf_partial(x,xd):
    return np.array([2*kV*x[0],2*kV*x[1]])

def clf_partiald(x,xd):
    return np.array([0,0])

# # # New Test
# def clf_0(x,xd):
#     return kV * (x[0]**2 + x[1]**2)

# def clf_partial_0(x,xd):
#     return np.array([2*kV*x[0],2*kV*x[1]])

# def clf_partiald(x,xd):
#     return np.array([0,0])

# def clf(x,xd):
#     return clf_0(x,xd) + 1 / cbf(x,0)[1]

# def clf_partial(x,xd):
#     return clf_partial_0(x,xd) + np.array([-2 * cbf(x,0)[1]**(-2) * (x[0] - x1)/a**2, -2 * cbf(x,0)[1]**(-2) * (x[1] - y1)/b**2])
