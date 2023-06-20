import numpy as np
import matplotlib.pyplot as plt

# Computation Functions
def tilde(x):
    x = np.squeeze(x)
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]
                    ])

def mrp2Quaternion(sigma):
    sigma_squared = np.inner(sigma, sigma)
    q0 = (1 - sigma_squared) / (1 + sigma_squared)
    q = [2 * sigma_i / (1 + sigma_squared) for sigma_i in sigma]
    q.insert(0,q0)
    return q

def quaternion2DCM(quat):
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    r11 = q0*q0 + q1*q1 - q2*q2 - q3*q3
    r12 = 2 * (q1*q2 + q0*q3)
    r13 = 2 * (q1*q3 - q0*q2)
    r21 = 2 * (q1*q2 - q0*q3)
    r22 = q0*q0 - q1*q1 + q2*q2 - q3*q3
    r23 = 2 * (q2*q3 + q0*q1)
    r31 = 2 * (q1*q3 + q0*q2)
    r32 = 2 * (q2*q3 - q0*q1)
    r33 = q0*q0 - q1*q1 - q2*q2 + q3*q3
    return np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])

def mrp2DCM(sigma):
    q = mrp2Quaternion(sigma)
    dcm = quaternion2DCM(q)
    return dcm

def dcm2MRP(dcm):
    zeta = np.sqrt(np.trace(dcm)+1)
    constant = 1 / (zeta**2 + 2 * zeta)
    s1 = constant * (dcm[1, 2] - dcm[2, 1])
    s2 = constant * (dcm[2, 0] - dcm[0, 2])
    s3 = constant * (dcm[0, 1] - dcm[1, 0])
    return np.array([s1, s2, s3])

# Dynamics Functions

def mrpShadow(mrp):
    norm = np.linalg.norm(mrp) ** 2
    return np.array([-i / norm for i in mrp])

def mrpDot(mrp, w):
    return 0.25 * np.dot(((1-np.dot(mrp,mrp)) * np.eye(3) + 2 * tilde(mrp) + 2 * np.outer(mrp,mrp)),w)

def mrpDotBMatrix(mrp):
    ss = np.dot(mrp,mrp)
    B = np.zeros((3,3))
    B[0, 0] = 1 - ss + 2 * mrp[0]**2
    B[1, 0] = 2 * (mrp[1] * mrp[0] + mrp[2])
    B[2, 0] = 2 * (mrp[2] * mrp[0] - mrp[1])
    B[0, 1] = 2 * (mrp[0] * mrp[1] - mrp[2])
    B[1, 1] =  1 - ss + 2 * mrp[1]**2
    B[2, 1] = 2 * (mrp[2] * mrp[1] + mrp[0])
    B[0, 2] = 2 * (mrp[0] * mrp[2] + mrp[1])
    B[1, 2] = 2 * (mrp[1] * mrp[2] - mrp[0])
    B[2, 2] =  1 - ss + 2 * mrp[2]**2
    return B
    
def targetMRP(tval):
    if targetIsStationary:
        return np.array([0, 0, 0])
    f = 0.05
    s1 = 0.2 * np.sin(f * tval)
    s2 = 0.3 * np.cos(f * tval)
    s3 = -0.3 * np.sin(f * tval)
    return np.array([s1, s2, s3])

def targetSpinRate(tval):                                               # Target spin rate relative to the inertial frame
    if targetIsStationary:
        return np.array([0,0,0])
    f = 0.05
    s1_dot = 0.2 * f * np.cos(f * tval)
    s2_dot = -0.3 * f * np.sin(f * tval)
    s3_dot = -0.3 * f * np.cos(f * tval)
    sigma_dot = np.array([s1_dot, s2_dot, s3_dot])
    sigma = targetMRP(tval)
    B = mrpDotBMatrix(sigma)
    target_w = 4 * np.dot(np.linalg.inv(B), sigma_dot)
    return target_w

def targetSpinRate_dot(tval, dt):                                       # The rate of change of the target spin rate relative to the inertial frame
    if targetIsStationary:
        return np.array([0, 0, 0])
    target_w_current = targetSpinRate(tval)
    target_w_previous = targetSpinRate(tval - dt)
    return (target_w_current - target_w_previous)/dt

def control(t, dt, mrp, w, z):
    sigmaRN = targetMRP(t)                                              # MRP of target relative to the inertial frame
    sigmaBN = mrp                                                       # MRP of spacecraft relative to the inertial frame
    sigmaBR = dcm2MRP(np.dot(mrp2DCM(sigmaBN), mrp2DCM(sigmaRN).T))     # MRP of Spacecraft relative to target
    wRN = targetSpinRate(t)
    wRN_dot = targetSpinRate_dot(t, dt)
    wRN_bodyFrame = np.dot(mrp2DCM(sigmaBR), wRN)                       # The DCM of sigmaBR is a transformation matrix that convert the target frame (R) into the body frame (B)
    wRN_dot_bodyFrame = np.dot(mrp2DCM(sigmaBR), wRN_dot)
    wBR = w - wRN_bodyFrame                                             # Spacecraft's spinrate - target's spinrate, both are expressed in the body frame

    # Control Algorithm
    #u = -K * sigmaBR - np.dot(P, wBR) + np.dot(I, wRN_dot_bodyFrame - np.cross(w, wRN_bodyFrame)) + np.cross(w, np.dot(I, w)) #- L
    u = -K * sigmaBR - np.dot(P, wBR) + np.dot(I, wRN_dot_bodyFrame - np.cross(w, wRN_bodyFrame)) + np.cross(w, np.dot(I, w)) - Ki * np.dot(P, z)

    return u

def wDot(t, dt, mrp, w, z):
    u = control(t, dt, mrp, w, z)
    # eq of motion : [I] w_dot = - w X [I] w + u + delta_L
    w_dot = np.dot(np.linalg.inv(I), (-np.cross(w, np.dot(I, w)) + u + delta_L))
    return w_dot

def wDot_RK4(t, dt, mrp, w, z):                                            # Integrate wDot with runge-kutta4
    k1 = wDot(t, dt, mrp, w, z)
    k2 = wDot(t+0.5*dt, dt, mrp, w+0.5*k1*dt, z)
    k3 = wDot(t+0.5*dt, dt, mrp, w+0.5*k2*dt, z)
    k4 = wDot(t+dt, dt, mrp, w+k3*dt, z)
    return dt * (k1 + 2*k2 + 2*k3 + k4)/6

# Parameters and Setup

spacecraftMRP = np.array([0.1, 0.2, -0.1]).T                                                # Initial MRP of the Spacecraft relative to the inertial frame
spacecraftMRP_history = [spacecraftMRP]
spacecraftSpinRate = np.array([np.deg2rad(i) for i in [3, 1, -2]]).T                     # Initial spinrate of the Spacecraft relative to the inertial frame
spacecraftSpinRate_history = [spacecraftSpinRate]

targetIsStationary = False
targetMRP_history = [targetMRP(0)]
targetSpinRate_history = [targetSpinRate(0)]
MRP_error_history = [dcm2MRP(np.dot(mrp2DCM(spacecraftMRP), mrp2DCM(targetMRP(0)).T))]      # dot product of the spacecraft MRP (BN) with the target MRP (RN.transpose) = BR

prev_sigmaBR = np.array([0, 0, 0])
prev_wBR = np.array([0, 0, 0])
z = np.array([0, 0, 0])                                                 # Integral feedback term
sum_sigmaBR = np.array([0, 0, 0])                                       # cumulative integral of sigmaBR

K = 5                                                                   # Nm
Ki = 0.005                                                              # s^-1
P = 10 * np.eye(3)                                                      # Nms
delta_L = np.array([0.5, -0.3, 0.2]).T                                        # Known external torque
I1, I2, I3 = 100, 75, 80                                                # kg m^2
I = np.array([[I1, 0, 0], [0, I2, 0], [0, 0, I3]])

h = 0.01
time = 241
tvec = np.linspace(0, time, int(time/h + 1))
prev_t = 0

for ti in tvec[1:]:
    dt = ti - prev_t
    prev_t = ti
    sigmaRN = targetMRP(ti)
    targetMRP_history.append(sigmaRN)
    sigmaBN = spacecraftMRP
    sigmaBR = dcm2MRP(np.dot(mrp2DCM(sigmaBN), mrp2DCM(sigmaRN).T))     # MRP of Spacecraft relative to target
    MRP_error_history.append(sigmaBR)
    wRN = targetSpinRate(ti)
    targetSpinRate_history.append(wRN)
    wRN_bodyFrame = np.dot(mrp2DCM(sigmaBR), wRN)

    # Integrating MRP, spinrate(w), and Integral feedback term
    
    sum_sigmaBR = sum_sigmaBR + (K * dt * 0.5 * (sigmaBR + prev_sigmaBR))
    prev_sigmaBR = sigmaBR
    wBR = spacecraftSpinRate - wRN_bodyFrame
    z = sum_sigmaBR + np.dot(I, (wBR - prev_wBR))
    prev_wBR = wBR
    
    spacecraftSpinRate = spacecraftSpinRate + wDot(ti, dt, spacecraftMRP, spacecraftSpinRate, z) * dt
    spacecraftMRP = spacecraftMRP + mrpDot(spacecraftMRP, spacecraftSpinRate) * dt
    
    #wDot_RK4(ti, dt, spacecraftMRP, spacecraftSpinRate, z)
    #wDot(ti, dt, spacecraftMRP, spacecraftSpinRate, z) * dt
    
    # MRP Switching
    if np.dot(spacecraftMRP, spacecraftMRP) > 1:
        spacecraftMRP = mrpShadow(spacecraftMRP)

    spacecraftMRP_history.append(spacecraftMRP)
    spacecraftSpinRate_history.append(spacecraftSpinRate)

    if ti % 25 ==0:
        print("Simulated {} seconds".format(ti))

spacecraftMRP_history = np.array(spacecraftMRP_history)
spacecraftMRP_norm = [np.dot(i, i) for i in spacecraftMRP_history]
print("Norm at ", tvec[4500], "s: ", np.sqrt(spacecraftMRP_norm[4500]))
spacecraftSpinRate_history = np.array(spacecraftSpinRate_history)
targetMRP_history = np.array(targetMRP_history)
targetSpinRate_history = np.array(targetSpinRate_history)
MRP_error_history = np.array(MRP_error_history)
MRP_error_norm  = [np.sqrt(i**2 + j**2 + k**2) for i, j, k in MRP_error_history]
print("Norm sigmaBR at ", tvec[4500], "s: ", MRP_error_norm[4500])
print("Ki :", Ki)
print("z :", z)

plt.figure(0)
plt.plot(tvec, spacecraftMRP_history[:, 0], 'g')
plt.plot(tvec, targetMRP_history[:, 0], 'g--')
plt.plot(tvec, spacecraftMRP_history[:, 1], 'b')
plt.plot(tvec, targetMRP_history[:, 1], 'b--')
plt.plot(tvec, spacecraftMRP_history[:, 2], 'r')
plt.plot(tvec, targetMRP_history[:, 2], 'r--')
plt.plot(tvec, spacecraftMRP_norm, 'k')
plt.title('Attitude (sigma) history')
plt.legend(["sigma_1", "sigma_1_target", "sigma_2", "sigma_2_target", "sigma_3", "sigma_3_target", "norm^2"])
plt.grid()
plt.figure(1)
plt.plot(tvec, spacecraftSpinRate_history[:, 0], 'b')
plt.plot(tvec, targetSpinRate_history[:, 0], 'b--')
plt.plot(tvec, spacecraftSpinRate_history[:, 1], 'r')
plt.plot(tvec, targetSpinRate_history[:, 1], 'r--')
plt.plot(tvec, spacecraftSpinRate_history[:, 2], 'g')
plt.plot(tvec, targetSpinRate_history[:, 2], 'g--')
plt.title("Spin Rate (w) history")
plt.legend(["w_1", "w_1_target", "w_2", "w_2_target", "w_3", "w_3_target"])
plt.grid()
plt.show()
