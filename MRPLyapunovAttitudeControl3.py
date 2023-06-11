import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Parameters
mrp = np.array([0.1, 0.2, -0.1]).T                  # Initial MRP of the Spacecraft relative to the inertial frame
mrp_history = [mrp]
w = np.array([np.deg2rad(i) for i in [30, 10, -20]]).T  # Initial spinrate of the Spacecraft relative to the inertial frame
w_history = [w]

K = 5 # Nm
P = 10 * np.eye(3) #NMs
delta_L = np.array([0.5, -0.3, 0.2]).T
I1 = 100
I2 = 75
I3 = 80 # kg m^2

I = np.array([[I1,0,0], [0, I2, 0], [0,0,I3]])
targetIsStationary = True

#L = 0;
L = np.array([0.5, -0.3, 0.2]).T

# Functions
def tilde(x):
    x = np.squeeze(x)
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]
                    ])

def mrpToDCM(sigma):
    sigma_squared = np.inner(sigma, sigma)
    q0 = (1 - sigma_squared) / (1 + sigma_squared)
    q = [2 * sigma_i / (1 + sigma_squared) for sigma_i in sigma]
    q.extend([q0])
    return Rotation.from_quat(q).as_matrix().T

def mrpToDCMScipy(sigma):
    return Rotation.from_mrp(sigma).as_matrix().T

def mrpToQuaternion(sigma):
    sigma_squared = np.inner(sigma, sigma)
    q0 = (1 - sigma_squared) / (1 + sigma_squared)
    q = [2 * sigma_i / (1 + sigma_squared) for sigma_i in sigma]
    q.insert(0,q0)
    return q

def quaternionToDCM(quat):
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    r11 = q0*q0 + q1*q1 - q2*q2 - q3*q3
    r12 = 2*(q1*q2 + q0*q3)
    r13 = 2*(q1*q3 - q0*q2)
    r21 = 2*(q1*q2 - q0*q3)
    r22 = q0*q0 - q1*q1 + q2*q2 - q3*q3
    r23 = 2*(q2*q3 + q0*q1)
    r31 = 2*(q1*q3 + q0*q2)
    r32 = 2*(q2*q3 - q0*q1)
    r33 = q0*q0 - q1*q1 - q2*q2 + q3*q3
    return np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])

def mrpToDCMNoLib(sigma):
    q = mrpToQuaternion(sigma)
    dcm = quaternionToDCM(q)
    return dcm

def dcmToMRP(matrix):
    zeta = np.sqrt(np.trace(matrix)+1)
    constant = 1 / (zeta**2 + 2 * zeta)
    s1 = constant * (matrix[1, 2] - matrix[2, 1])
    s2 = constant * (matrix[2, 0] - matrix[0, 2])
    s3 = constant * (matrix[0, 1] - matrix[1, 0])
    return np.array([s1, s2, s3])

def mrpShadow(mrp):
    norm = np.linalg.norm(mrp) ** 2
    return np.array([-i / norm for i in mrp])

def targetMRP(tval):
    if targetIsStationary:
        return np.array([0, 0, 0])
    f = 0.05
    s1 = 0.2 * np.sin(f * tval)
    s2 = 0.3 * np.cos(f * tval)
    s3 = -0.3 * np.sin(f * tval)
    return np.array([s1, s2, s3])

def targetSpinRate(tval):  # Target spin rate relative to the inertial frame
    if targetIsStationary:
        return np.array([0,0,0])
    f = 0.05
    s1_dot = 0.2 * f * np.cos(f * tval)
    s2_dot = -0.3 * f * np.sin(f * tval)
    s3_dot = -0.3 * f * np.cos(f * tval)
    sigma_dot = np.array([s1_dot, s2_dot, s3_dot])
    sigma = targetMRP(tval)
    A = mrpDotMatrix(sigma)
    w = 4 * np.dot(np.linalg.inv(A), sigma_dot)
    return w

def targetSpinRate_dot(tval, dt): # The rate of change of the target spin rate relative to the inertial frame
    if targetIsStationary:
        return np.array([0, 0, 0])
    w1 = targetSpinRate(tval)
    w2 = targetSpinRate(tval - dt)
    return (w1 - w2)/dt

def mrpDot(mrp, w):
    return 0.25 * np.dot(((1-np.dot(mrp,mrp)) * np.eye(3) + 2 * tilde(mrp) + 2 * np.outer(mrp, mrp)), w)

def mrpDotMatrix(mrp):
    ss = np.dot(mrp, mrp)
    A = np.zeros((3,3))
    A[0, 0] = 1 - ss + 2 * mrp[0] **2
    A[1, 0] = 2*(mrp[1] * mrp[0] + mrp[2])
    A[2, 0] = 2*(mrp[2] * mrp[0] - mrp[1])
    A[0, 1] = 2*(mrp[0] * mrp[1] - mrp[2])
    A[1, 1] = 1 - ss + 2 * mrp[1] ** 2
    A[2, 1] = 2*(mrp[2] * mrp[1] + mrp[0])
    A[0, 2] = 2*(mrp[0] * mrp[2] + mrp[1])
    A[1, 2] = 2*(mrp[1] * mrp[2] - mrp[0])
    A[2, 2] = 1 - ss + 2 * mrp[2] ** 2
    return A

def control(t, dt, mrp, w):
    sigmaRN = targetMRP(t)                                              # MRP of target relative to inertial frame
    sigmaBN = mrp
    sigmaBR = dcmToMRP(np.dot(mrpToDCM(sigmaBN), mrpToDCM(sigmaRN).T))  # MRP of Spacecraft relative to target
    wRN = targetSpinRate(t)
    wRN_dot = targetSpinRate_dot(t, dt)
    wRN_bodyFrame = np.dot(mrpToDCM(sigmaBR), wRN)                      # The DCM of sigmaBR is a transformation matrix that convert the target frame (R) into the body frame (B)
    wRN_dot_bodyFrame = np.dot(mrpToDCM(sigmaBR), wRN_dot)
    wBR = w - wRN_bodyFrame
    # Control Function
    #u = -K * sigmaBR - np.dot(P, wBR) + np.dot(I, wRN_dot_bodyFrame - np.cross(w, wRN_bodyFrame)) + np.cross(w, np.dot(I, w)) - L
    u = -K * sigmaBR - np.dot(P, wBR) + np.dot(I, wRN_dot_bodyFrame - np.cross(w, wRN_bodyFrame)) + np.cross(w, np.dot(I, w))
    #u  =-K * sigmaBR - np.dot(P,wBR)
    return u

def wDot(t, dt, mrp, w):
    u = control(t, dt, mrp, w)
    # eq : [I] w_dot = - w X [I] w + u
    w_dot = np.dot(np.linalg.inv(I), (-np.cross(w, np.dot(I, w)) + u + L))
    return w_dot

# Runge-Kutta Functions

def wDot_RK4(t, dt, mrp, w):
    k1 = wDot(t, dt, mrp, w)
    k2 = wDot(t+0.5*dt, dt, mrp, w+0.5*k1*dt)
    k3 = wDot(t+0.5*dt, dt, mrp, w+0.5*k2*dt)
    k4 = wDot(t+dt, dt, mrp, w+k3*dt)
    return dt * (k1 + 2*k2 + 2*k3 + k4)/6
    
def mrpDot_RK4(y, t, dt):
    k1 = mrpDot(y, t)
    k2 = mrpDot(y+0.5*k1*dt, t+0.5*dt)
    k3 = mrpDot(y+0.5*k2*dt, t+0.5*dt)
    k4 = mrpDot(y+k3*dt, t+dt)
    return dt * (k1 + 2*k2 + 2*k3 + k4)/6

h = 0.01
time = 121
tvec = np.linspace(0, time, int(time/h +1))
prev_t = 0
targetMRP_histories = [targetMRP(0)]
target_spinrate_history = [targetSpinRate(0)]
error_history = [dcmToMRP(np.dot(mrpToDCM(mrp), mrpToDCM(targetMRP(0)).T))]
for ti in tvec[1:]:
    dt = ti - prev_t
    prev_t = ti
    sigmaRN = targetMRP(ti)
    targetMRP_histories.append(sigmaRN)
    sigmaBR = dcmToMRP(np.dot(mrpToDCM(mrp), mrpToDCM(sigmaRN).T))     # MRP of Spacecraft relative to target
    error_history.append(sigmaBR)
    wRN = targetSpinRate(ti)
    target_spinrate_history.append(wRN)
    wRN_dot = targetSpinRate_dot(ti, dt)
    wBR = w - wRN

    # integrator
    mrp = mrp + mrpDot(mrp, w) * dt
    w = w + wDot_RK4(ti, dt, mrp, w)
##    mrp = mrp + mrpDot(mrp, w) * dt
##    w = w + wDot(ti, dt, mrp, w) * dt

    # MRP switching
    if np.dot(mrp, mrp) > 1:
        mrp = mrpShadow(mrp)

    mrp_history.append(mrp)
    w_history.append(w)
    
    if ti % 25 ==0:
        print("Simulated {} seconds".format(ti))

mrp_history = np.array(mrp_history)
mrp_norm = [np.dot(i, i) for i in mrp_history]
print("Norm at ", tvec[3000], "s: ", np.sqrt(mrp_norm[3000]))
w_history = np.array(w_history)
targetMRP_histories = np.array(targetMRP_histories)
target_spinrate_history = np.array(target_spinrate_history)
error_history = np.array(error_history)
error_norm = [np.sqrt(i**2 + j**2 + k**2) for i, j, k in error_history]
print("Norm sigmaBR at ", tvec[3500], "s: ", error_norm[3500])
print(sigmaBR)

plt.figure(0)
plt.plot(tvec, mrp_history[:, 0], 'g')
plt.plot(tvec, targetMRP_histories[:, 0], 'g--')
plt.plot(tvec, mrp_history[:, 1], 'b')
plt.plot(tvec, targetMRP_histories[:, 1], 'b--')
plt.plot(tvec, mrp_history[:, 2], 'r')
plt.plot(tvec, targetMRP_histories[:, 2], 'r--')
plt.plot(tvec, mrp_norm, 'k')
plt.title('Attitude (sigma) history')
plt.legend(["sigma_1", "sigma_1_target", "sigma_2", "sigma_2_target", "sigma_3", "sigma_3_target", "norm^2"])
plt.grid()
plt.figure(1)
plt.plot(tvec, w_history[:, 0], 'b')
plt.plot(tvec, target_spinrate_history[:, 0], 'b--')
plt.plot(tvec, w_history[:, 1], 'r')
plt.plot(tvec, target_spinrate_history[:, 1], 'r--')
plt.plot(tvec, w_history[:, 2], 'g')
plt.plot(tvec, target_spinrate_history[:, 2], 'g--')
plt.title("Spin Rate (w) history")
plt.legend(["w_1", "w_1_target", "w_2", "w_2_target", "w_3", "w_3_target"])
plt.grid()
plt.show()
