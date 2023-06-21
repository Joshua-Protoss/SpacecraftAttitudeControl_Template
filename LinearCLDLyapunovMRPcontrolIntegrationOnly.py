import numpy as np
import matplotlib.pyplot as plt

# Functions
def G(y,t):
    sigma_d, sigma = y[0], y[1]
    sigma_dd = -np.dot(P, sigma_d) - K * sigma

    return np.array([sigma_dd, sigma_d])

def RK4(y, t, dt):
    k1 = G(y, t)
    k2 = G(y+0.5*k1*dt, t+0.5*dt)
    k3 = G(y+0.5*k2*dt, t+0.5*dt)
    k4 = G(y+k3*dt, t+dt)
    
    return dt * (k1 + 2*k2 + 2*k3 + k4)/6

def mrpShadow(mrp):
    norm = np.linalg.norm(mrp) ** 2
    return np.array([-i / norm for i in mrp])

# Parameters and initial conditions

sigma_0 = np.array([0.1, 0.2, -0.1])
sigma_d0 = np.array([np.deg2rad(i) for i in [3, 1, -2]])
y = np.array([sigma_d0, sigma_0])
K = 0.11                                                                #s^-2
P = 3 * np.eye(3)                                                       # Nms
I1, I2, I3 = 100, 75, 80                                                # kg m^2
I = np.array([[I1, 0, 0], [0, I2, 0], [0, 0, I3]])
umax = 1                                                                # Nm
h = 0.01
time = 121
tvec = np.linspace(0, time, int(time/h + 1))
prev_t = 0
x1 = [0.1]
x2 = [0.2]
x3 = [-0.1]

for ti in tvec[1:]:
    dt = ti - prev_t
    prev_t = ti
    y = y + RK4(y, ti, dt)
    if np.dot(y[1], y[1]) > 1:
        y[1] = mrpShadow(y[1])
    x1.append(y[1][0])
    x2.append(y[1][1])
    x3.append(y[1][2])
 
    if ti % 25 == 0:
        print("Simulated {} seconds".format(ti))
        
plt.plot(tvec, x1)
plt.plot(tvec, x2)
plt.plot(tvec, x3)
plt.xlabel('time (s)')
plt.ylabel('MRPs')
plt.title('Response Curves switching')
plt.legend(['X1', 'X2', 'X3'], loc='lower right')
plt.show()

