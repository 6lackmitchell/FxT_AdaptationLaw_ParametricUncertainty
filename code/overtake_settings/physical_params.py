import numpy as np

G                     = 9.81 # meters / sec^2
tau                   = 1.8  # sec

# Taken from 2020 Ford Explorer Base Model (https://www.ford.com/suvs/explorer/models/explorer)
car_length            = 5.05 # meters
car_width             = 2.27 # meters
M                     = 1970 # kg

# Taken from 2020 Ford Mustang Shelby GT (https://www.ford.com/cars/mustang/models/shelby-gt350r)
car_length            = 4.81 # meters
car_width             = 1.92 # meters
M                     = 1994 # kg

# Car position set as center of vehicle, must keep all wheels on road
lane_width            = 3 # meters
edge_right            = car_width / 2
edge_left             = 2 * lane_width - edge_right
ER                    = edge_right
EL                    = edge_left



# Ensures that there is always at 1 meter side-to-side between vehicles (https://mylicence.sa.gov.au/the-hazard-perception-test/safe-distance-to-the-side)
safe_lateral_distance = (lane_width - car_width)/2 + car_width
safe_lateral_distance = 0.75 + car_width

speed_limit           = 30 # meters / sec

# Control Constraints
w_max                 = np.pi / 18  # angular control limit
u_max                 = 0.25 * M * G # longitudinal control limit
TIME_CONSTANT         = 0.5#1.0#0.4

# Oncoming Vehicle Parameters
TIME_HEADWAY = 24.0
ONCOMING_FREQUENCY = 30.0

# CBF Edge Definitions
def ER(x):
    return edge_right + x[2]*x[3]*np.sin(x[2])/w_max - 1/2*(x[2]/w_max)**2 * (u_max/M*np.sin(x[2]) + x[3]*w_max*np.cos(x[2]))
def EL(x):
    return edge_left  - x[2]*x[3]*np.sin(x[2])/w_max - 1/2*(x[2]/w_max)**2 * (u_max/M*np.sin(x[2]) - x[3]*w_max*np.cos(x[2]))
def dERt(x):
    return  (x[3]*np.sin(x[2]) + x[3]*x[2]*np.cos(x[2])) / w_max - x[2]/w_max**2 *\
           (u_max/M*np.sin(x[2]) + x[3]*w_max*np.cos(x[2])) - x[2]**2/(2*w_max**2)*\
           (u_max/M*np.cos(x[2]) - x[3]*w_max*np.sin(x[2]))
def dELt(x):
    return -(x[3]*np.sin(x[2]) + x[3]*x[2]*np.cos(x[2])) / w_max - x[2]/w_max**2 *\
           (u_max/M*np.sin(x[2]) - x[3]*w_max*np.cos(x[2])) - x[2]**2/(2*w_max**2)*\
           (u_max/M*np.cos(x[2]) + x[3]*w_max*np.sin(x[2]))
def dERv(x):
    return  x[2]*np.sin(x[2]) / w_max - 1/2*(x[2]/w_max)**2*w_max*np.cos(x[2])
def dELv(x):
    return -x[2]*np.sin(x[2]) / w_max + 1/2*(x[2]/w_max)**2*w_max*np.cos(x[2])