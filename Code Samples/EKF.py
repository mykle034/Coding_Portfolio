import numpy as np
import math

def rotMat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def EKFPropagate(x_hat_t, # robot position and orientation
                 Sigma_x_t, # estimation uncertainty
                 u, # control signals, v and phi # u[1] = omega
                 Sigma_n, # uncertainty in control signals
                 dt # timestep
    ):

    ct = np.cos(x_hat_t[2])
    st = np.sin(x_hat_t[2])
    v = u[0]

    phi = np.array([
            [1, 0, -v * dt * st],
            [0, 1, v * dt * ct],
            [0, 0, 1]
            ])

    G = np.array([
            [dt * ct, 0],
            [dt * st, 0],
            [0, dt]
        ])

    x_hat_t = np.array([x_hat_t]).T

    x_hat_t = x_hat_t + np.array([[u[0]*ct], [u[0]*st], [u[1]]]) * dt

    x_hat_t = x_hat_t.squeeze()

    Sigma_x_t = phi @ Sigma_x_t @ phi.T + G @ Sigma_n @ G.T

    return x_hat_t, Sigma_x_t

def EKFRelPosUpdate(x_hat_t, # robot position and orientation
                    Sigma_x_t, # estimation uncertainty
                    z, # measurements
                    Sigma_m, # measurements' uncertainty
                    G_p_L, # landmarks' global positions
                    dt # timestep
                   ):

    x_hat_t = np.array([x_hat_t]).T
    z = np.array([z]).T
    G_p_L = np.array([G_p_L]).T

    H = np.append(-rotMat(x_hat_t[2]).T, -rotMat(x_hat_t[2]).T @ [[0, -1], [1, 0]] @ (G_p_L[0:2] - x_hat_t[0:2]), axis=1)

    K_t = Sigma_x_t @ H.T @ np.linalg.inv(H @ Sigma_x_t @ H.T + Sigma_m)

    z_hat = rotMat(x_hat_t[2]).T @ (G_p_L[0:2] - x_hat_t[0:2])

    z_tilde = z - z_hat # little r

    x_hat_t = x_hat_t + K_t @ z_tilde

    Sigma_x_t = Sigma_x_t - K_t @ H @ Sigma_x_t

    x_hat_t = x_hat_t.squeeze()

    return x_hat_t, Sigma_x_t

def EKFSLAMRelPosUpdate(x_hat_t, # robot position and orientation
                    Sigma_x_t, # estimation uncertainty
                    z, # measurements
                    Sigma_m, # measurements' uncertainty
                    G_p_L, # landmarks' global positions
                    dt # timestep
                   ):

    x_hat_t = np.array([x_hat_t]).T
    z = np.array([z]).T
    G_p_L = np.array([G_p_L]).T

    H = np.append(-rotMat(x_hat_t[2]).T, -rotMat(x_hat_t[2]).T @ [[0, -1], [1, 0]] @ (G_p_L[0:2] - x_hat_t[0:2]), axis=1)

    K_t = Sigma_x_t @ H.T @ np.linalg.inv(H @ Sigma_x_t @ H.T + Sigma_m)

    z_hat = rotMat(x_hat_t[2]).T @ (G_p_L[0:2] - x_hat_t[0:2])

    z_tilde = z - z_hat # little r

    x_hat_t = x_hat_t + K_t @ z_tilde

    Sigma_x_t = Sigma_x_t - K_t @ H @ Sigma_x_t

    x_hat_t = x_hat_t.squeeze()

    return x_hat_t, Sigma_x_t