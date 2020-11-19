import numpy as np
from .physical_params import M, TIME_CONSTANT, u_max

def f(x):
    if np.sum(x.shape) == x.shape[0]:
        f = [x[3] *np.cos(x[2]),
             x[3] *np.sin(x[2]),
             0,
             0,
             x[7] *np.cos(x[6]),
             x[7] *np.sin(x[6]),
             0,
             0,
             x[11]*np.cos(x[10]),
             x[11]*np.sin(x[10]),
             0,
             0]
        f = np.array(f)
    else:
        
        f = [np.array([xx[3] *np.cos(xx[2]),
                       xx[3] *np.sin(xx[2]),
                       0,
                       0,
                       xx[7] *np.cos(xx[6]),
                       xx[7] *np.sin(xx[6]),
                       0,
                       0,
                       xx[11]*np.cos(xx[10]),
                       xx[11]*np.sin(xx[10]),
                       0,
                       0]) \
        for xx in x]
        f = np.array(f)

    return f

def g(x):
    a = np.array([[0,  0],
                  [0,  0],
                  [1,  0],
                  [0,1/M],
                  [0,  0],
                  [0,  0],
                  [0,  0],
                  [0,  0],
                  [0,  0],
                  [0,  0],
                  [0,  0],
                  [0,  0]])
    if np.sum(x.shape) == a.shape[0]:
        return a
    else:
        return np.array(x.shape[0]*[a])

def regressor(x,t):
    freq  = 0.25 # 1/4 Hz -- 4 second cycle
    scale = 0.5 / (2*np.pi*freq)
    a = scale*np.array([[0,                                                     0],
                        [0,                                                     0],
                        [0,                                                     0],
                        [0,                                                     0],
                        [0,                                                     0],
                        [0,                                                     0],
                        [1 + np.sin(2*np.pi*t)**2,                              0],
                        [0,                        0.1*(1 + np.cos(2*np.pi*t)**2)],
                        [0,                                                     0],
                        [0,                                                     0],
                        [0,                                                     0],
                        [0,                                                     0]])
    
    a = scale*np.array([[0,                                                        0],
                        [0,                                                        0],
                        [0,                                                        0],
                        [0,                                                        0],
                        [2 + np.cos(2*np.pi*freq*t),                               0],
                        [0,                         0.1*(2 + np.sin(2*np.pi*freq*t))],
                        [0,                                                        0],
                        [0,                                                        0],
                        [0,                                                        0],
                        [0,                                                        0],
                        [0,                                                        0],
                        [0,                                                        0]])

    freq_x  = 0.01 # 1/100 m^(-1) -- Full cycle every 100m
    freq_y  = 0.02 # 1/50 m^(-1)  -- Full cycle every 50m
    if np.sum(x.shape) == x.shape[0]:
        DX = 1 + 0.5 * (1 - np.cos(2*np.pi*freq_x*x[4]))
        DY = 0.1*(1 + 0.5 * (1 - np.sin(2*np.pi*freq_y*x[4])))
        a = np.array([[0,   0],
                      [0,   0],
                      [0,   0],
                      [0,   0],
                      [DX,  0],
                      [0,  DY],
                      [0,   0],
                      [0,   0],
                      [0,   0],
                      [0,   0],
                      [0,   0],
                      [0,   0]])
    else:
        DX = 1 + 0.5 * (1 - np.cos(2*np.pi*freq_x*x[:,4]))
        DY = 0.1*(1 + 0.5 * (1 - np.sin(2*np.pi*freq_y*x[:,4])))
        a = np.array([[[0,   0],
                      [0,   0],
                      [0,   0],
                      [0,   0],
                      [dx,  0],
                      [0,  dy],
                      [0,   0],
                      [0,   0],
                      [0,   0],
                      [0,   0],
                      [0,   0],
                      [0,   0]] for dx,dy in zip(DX,DY)])
    return a

    # Time-varying
    a = np.array([[0,                                                                                                                           0],
                  [0,                                                                                                                           0],
                  [0,                                                                                                                           0],
                  [0,                                                                                                                           0],
                  [1 + u_max / (10*M*2*np.pi*freq) * (1 - np.cos(2*np.pi*freq*t)),                                                                 0],
                  [0,                                                                 0.1*(1 + u_max / (10*M*2*np.pi*freq) * np.sin(2*np.pi*freq*t))],
                  [0,                                                                                                                           0],
                  [0,                                                                                                                           0],
                  [0,                                                                                                                           0],
                  [0,                                                                                                                           0],
                  [0,                                                                                                                           0],
                  [0,                                                                                                                           0]])

    if np.sum(x.shape) == a.shape[0]:
        return a
    else:
        return np.array(x.shape[0]*[a])

def system_dynamics(t,x,u,theta):
    black_reg = regressor(x,t); #black_reg[-1,:,:] = 0
    xdot = f(x) + np.einsum('ijk,ik->ij',g(x),u) + np.dot(black_reg,theta)

    return xdot

def xf_2dot(x,xf,xf_dot,k):
    return (x - xf - 2*k*xf_dot) / (k**2)

def phif_2dot(x,u,phif,phif_dot,k):
    return (f(x) + np.dot(g(x),u) - phif - 2*k*phif_dot) / (k**2)

def Phif_2dot(reg,Phif,Phif_dot,k):
    return (reg - Phif - 2*k*Phif_dot) / (k**2)

def fd(x,xd):
    # if np.sum(x.shape) == x.shape[0]:
    xdot_d = x[7]
    ydot_d = 0
    tdot_d = (xdot_d - x[3]*np.cos(x[2]))*(x[1]-xd[1]) - (ydot_d - x[3]*np.sin(x[2]))*(x[0]-xd[0])
    tdot_d = tdot_d / ((x[0]-xd[0])**2 + (x[1]-xd[1])**2)
    tdot_d = -x[3]*np.sin(x[2]) / np.sqrt(1 - ((xd[1]-x[1])/x[3]*TIME_CONSTANT)**2)
    vdot_d = 0

    f = [xdot_d,
         ydot_d,
         tdot_d,
         vdot_d,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0]
    f = np.array(f)

    return f


