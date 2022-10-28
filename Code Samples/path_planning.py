import numpy as np
import math
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float


def bug2(x_hat_t, CurrentState, m, b, goal, dists, EntrancePoint):

    theta_gain = .1
    desired_wall_dist = 0.1
    return_to_line_gain = .1

    if CurrentState == 'FOLLOW GOAL LINE':
        # Find out far from the ideal line the robot is:
        rel_dist_to_line = (m*x_hat_t[0] - x_hat_t[1] + b) / np.sqrt(m * m + 1)

        # A factor to correct the desired theta to return to the line
        return_to_line_angle_factor = return_to_line_gain * rel_dist_to_line

        # Find distance to goal
        goal_dist = math.sqrt(math.pow(goal[0] - x_hat_t[0], 2) + math.pow(goal[1] - x_hat_t[1], 2))
        if goal_dist < 0.1:
            print('at goal')
            goal_dist = 0


        # Find the line angle, correct for offset from line, and set the command signal
        theta_desired = return_to_line_angle_factor + np.arctan((goal[1] - x_hat_t[1]) / (goal[0] - x_hat_t[0]))
        # Move the robot
        # leftSpeed, rightSpeed = MotorController(x_hat_t, goal_dist, theta_desired)
        dist_to_line = np.abs(m*x_hat_t[0] - x_hat_t[1] + b) / np.sqrt(m * m + 1)

        if min(dists) < 0.08:
            CurrentState = 'FOLLOW WALL'
            EntrancePoint = [x_hat_t[0], x_hat_t[1]]
            print('Changing to Turn to follow wall')


    if CurrentState == 'FOLLOW WALL':

        # dists = np.random.rand(512)
        # dists[0:400] = 10000
        point_thetas = np.linspace(0, 2*np.pi, num=len(dists))
        theta_min_dist = point_thetas[np.argmin(dists)]
        # print('minimum Lidar distance:', min(dists))
        # print('minimim Lidar Angle [deg]', np.rad2deg(theta_min_dist))

        theta_desired = x_hat_t[2] - (theta_min_dist - np.pi) + np.pi/2
        # theta_desired = theta_min_dist - 3/2*np.pi

        # print('theta_desired before correction', np.rad2deg(theta_desired))
        # print('theta correction factor', (desired_wall_dist/min(dists))**theta_gain)
        # if min(dists) > desired_wall_dist:
        # theta_desired = (desired_wall_dist/min(dists)**theta_gain) * theta_desired
        theta_desired = (desired_wall_dist/min(dists))**theta_gain * (theta_desired + 2*np.pi) - 2*np.pi

        # print('theta_desired', np.rad2deg(theta_desired), 'Robot Bearing', np.rad2deg(x_hat_t[2]))
        # leftSpeed, rightSpeed = MotorController(x_hat_t, 10, theta_desired)

        # Finding exit point
        dist_to_line = np.abs(m*x_hat_t[0] - x_hat_t[1] + b) / np.sqrt(m * m + 1)
        # print('distance to line: ', dist_to_line)
        dist_to_entrance_point = math.sqrt(math.pow(EntrancePoint[0] - x_hat_t[0], 2) + math.pow(EntrancePoint[1] - x_hat_t[1], 2))
        if dist_to_line <= 0.01 and dist_to_entrance_point > 1:
            CurrentState = 'FOLLOW GOAL LINE'
            print('Changing to Turn to Follow goal line')


    return CurrentState, theta_desired, EntrancePoint