"""project_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
from controller import CameraRecognitionObject
from scipy.stats.distributions import chi2
import numpy as np
import math
import sys
from EKF import *
from Corners import *
import path_planning


# initial pose
x, y, theta = -2.89, 0.0, 0.0
# goal position
goal_x, goal_y = 2.89, 0.0
# threshold for reaching goal
goal_threshold = 0.01
# threshold for mahalanobis distance
md_threshold = 1.00
# threshold distance for same corner 
d_threshold = 0.25
# iterations to scan LIDAR
scan_f = 1
# min measurements
min_m = 1
# list of landmarks
landmarks = []
corners = []

# initial control
v = 0.0
omega = 0.0
fov = 0.84 # the field of view of the camera (rad)
w = 640 # the width of camera (pixels)
wheelRadius = 0.0205
axleLength = 0.0568 # Data from Webots website seems wrong. The real axle length should be about 56-57mm

# Initializations for path planning
# Added goal line function
def GoalLine(pos1, goal):
     slope = (goal[1]-pos1[1]) / (goal[0]-pos1[0])
     intercept = goal[1] - slope * goal[0]
     return slope, intercept
from controller import Motor # Added

robot = Supervisor()

x_hat_t = [x, y, theta]
goal = [goal_x, goal_y]
m, b = GoalLine(x_hat_t, goal)
CurrentState = 'FOLLOW GOAL LINE'
EntrancePoint = [0,0]
counter = 0
theta_desired = 0
def MotorController(x_hat_t, d, theta_desired):
    # A simple p controller

    ka = 15 # Angle gain
    kd = 30 # Distance gain

    omega_command = ka * (theta_desired - x_hat_t[2])

    # set the velocity command signal
    v_command = kd * d

    # Set the motor command signals
    leftSpeed  = v_command - omega_command
    rightSpeed = v_command + omega_command

    #limit speeds
    leftSpeed = min(leftSpeed, 6.28)
    leftSpeed = max(leftSpeed, -6.28)
    rightSpeed = min(rightSpeed, 6.28)
    rightSpeed = max(rightSpeed, -6.28)

    # write commands to motor contorllers
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)


# create the Robot instance.
camera = robot.getDevice('camera')
camera.enable(1)

if camera.hasRecognition():
    camera.recognitionEnable(1)
    camera.enableRecognitionSegmentation()
else:
    print("Your camera does not have recognition")

gyro = robot.getDevice('gyro')
gyro.enable(1)

timestep = int(robot.getBasicTimeStep())
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)
lidar = robot.getDevice('lidar')
lidar.enable(1)
lidar.enablePointCloud()

# init var for EKF
dt = timestep / 1000.0
# dt = 0.001
x_hat_t = np.array([x, y, theta])
Sigma_x_t = np.zeros((3,3))
Sigma_x_t[0,0], Sigma_x_t[1,1], Sigma_x_t[2,2] = 0.01, 0.01, np.pi/90
Sigma_n = np.zeros((2,2))
std_n_v = 1.0
std_n_omega = 10.0
Sigma_n[0,0] = std_n_v * std_n_v
Sigma_n[1,1] = std_n_omega * std_n_omega
u = np.array([v, omega])
std_m = 0.01 # gaussian noise for camera
Sigma_m = [[std_m*std_m, 0], [0,std_m*std_m]] 
std_d = 0.01 # gaussian noise for lidar
Sigma_d = [[std_d*std_d, 0], [0,std_d*std_d]]

# helper functions
def omegaToWheelSpeeds(omega, v):
    wd = omega * axleLength * 0.5
    return v - wd, v + wd

def reachGoal(x_hat_t):
    d = ((x_hat_t[0] - goal_x)**2 + (x_hat_t[1] - goal_y)**2)**0.5
    if d <= goal_threshold:
        return True
    return False

class Corner():
    def __init__(self, G_p_C, Sigma_x):
        self.G_p = G_p_C
        self.positions = [G_p_C]
        self.cov = Sigma_m + Sigma_x[0:2,0:2]

    def meanPos(self):
        return np.mean(self.positions, axis = 0)

    def mahalanobisDistance(self, G_p):
        first = (G_p.T - self.meanPos().T).T
        # issues when cov matrix is singular
        second = np.linalg.inv(self.cov)
        third = (G_p.T - self.meanPos().T)
        return (first @ second @ third) ** 0.5

    def sameFeature(self, G_p):
        if np.linalg.norm(self.G_p - G_p) < d_threshold:
        #if self.mahalanobisDistance(G_p) < md_threshold:
            return True
        return False

    def updatePositions(self, G_p):
        self.positions.append(G_p)
        self.cov = np.cov(np.array(self.positions))
        self.G_p = self.meanPos()
        return

def UpdateFeatures(features, G_p, Sigma_x):
    for i in range(len(features)):
    # determine if known corner
        if features[i].sameFeature(G_p):
            features[i].updatePositions(G_p)
            return features, i

    # unknown corner
    feature = Corner(G_p, Sigma_x)
    features.append(feature)
    return features, len(features) - 1

class Landmark(Corner):
    def __init__(self, ID, G_p_L, Sigma_x):
        self.ID = ID
        self.G_p = G_p_L
        self.positions = [G_p_L]
        self.cov = Sigma_m + Sigma_x[0:2,0:2]

def UpdateLandmarks(landmarks, recObj, G_p_L, Sigma_x):
    # Update known landmarks given landmark in view and return its pose
    ID = recObj.get_id()
    if ID not in [l.ID for l in landmarks]:
        # new landmark
        landmark = Landmark(recObj.get_id(), G_p_L, Sigma_x)
        landmarks.append(landmark)
    else:
        # determine landmark in list
        for x in range(len(landmarks)):
            if landmarks[x].ID == ID:
                landmarks[x].updatePositions(G_p_L)
                return landmarks, x

    return landmarks, len(landmarks) - 1

count = 100
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    # Get landmarks in view of camera
    recObjs = camera.getRecognitionObjects()
    recObjsNum = camera.getRecognitionNumberOfObjects()
    # Get lidar sensor info
    dists = np.array(lidar.getRangeImage())

    # Update u
    leftSpeed = leftMotor.getVelocity()
    rightSpeed = rightMotor.getVelocity()
    v = ((leftSpeed + rightSpeed) / 2) * wheelRadius
    #omega = (rightSpeed - leftSpeed) * wheelRadius /axleLength
    omega = gyro.getValues()[2]/7500 + np.random.normal(0, 0.1)
    u = [v, omega]

    x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)

    # Find recognizable features
    curLandmarks = [] # indices of landmarks currently visible
    for i in range(recObjsNum):
        # iterate over all landmarks in view
        # get z relative position
        z_pos = recObjs[i].get_position()[0:2]
        z_pos = [z_pos[0] + np.random.normal(0,std_m), z_pos[1] + np.random.normal(0,std_m)] # add noise

        # Update known landmarks
        G_p_L = rotMat(x_hat_t[2]) @ z_pos[0:2] + x_hat_t[0:2]
        landmarks, idx = UpdateLandmarks(landmarks, recObjs[i], G_p_L, Sigma_x_t)
        #landmarks, idx = UpdateFeatures(landmarks, G_p_L, Sigma_x_t) # without using IDs
        curLandmarks.append(idx)
        if len(landmarks[idx].positions) > min_m:
            x_hat_t, Sigma_x_t = EKFRelPosUpdate(x_hat_t, Sigma_x_t, z_pos, Sigma_m, landmarks[idx].G_p, dt)

    # identify visible corners
    intersections = []
    
    if count%scan_f == 0:
        count = 0
        lines = FindLines(dists)
        intersections = FindIntersections(lines) # in robot coordinate frame
    count = count + 1

    curCorners = [] # indices of corners currently visible
    for i in range(len(intersections)):
        # update corners
        G_p_C = rotMat(x_hat_t[2]) @ intersections[i] + x_hat_t[0:2]
        corners, idx = UpdateFeatures(corners, G_p_C, Sigma_x_t)
        curCorners.append(idx)
        if len(corners[idx].positions) > min_m :
            x_hat_t, Sigma_x_t = EKFRelPosUpdate(x_hat_t, Sigma_x_t, intersections[i], Sigma_d, corners[idx].G_p, dt)

    # Perform EKF SLAM with full robot state
    # EKF Propagate
    #X_hat_t = np.concatenate(x_hat_t, [landmarks[curLandmarks].G_p for x in curLandmarks], [corners[x].G_p for x in curCorners]).ravel()
    #Sigma_X_t = np.zeros([len(X_hat_t), len(X_hat_t)])
    #Sigma_X_t[0:3,0:3] = Sigma_x_t
    #i=3
    #for x in range(len(curLandmarks)):
    #    Sigma_X_t[i:i+2,i:i+2] = landmarks[curLandmarks[x]].cov
    #    i = i+2
    #for x in range(len(curCorners)):
    #    Sigma_X_t[i:i+2,i:i+2] = corners[curCorners[x]].cov
    #    i = i+2
    #X_hat_t, Sigma_X_t = EKFSLAMPropagate(X_hat_t, Sigma_X_t, u, Sigma_n, dt)

    # EKF update
    #R_p_Ls =
    #G_p_Ls =
    #Sigma_M =
    #X_hat_t, Sigma_X_t = EKFRelPosUpdate(X_hat_t, Sigma_X_t, R_p_Ls, Sigma_M, G_p_Ls)

    #x_hat_t = X_hat_t[0:2]
    #Sigma_x_t = Sigma_X_t[0:2, 0:2]
    #i=3
    #for x in range(len(curLandmarks)):
    #    landmarks[curLandmarks[x]].cov = Sigma_X_t[i:i+2,i:i+2]
    #    landmarks[curLandmarks[x]].G_p = X_hat_t[i:i+2]
    #    i = i+2
    #for x in range(len(curCorners)):
    #    corners[curCorners[x]].cov = Sigma_X_t[i:i+2,i:i+2]
    #    corners[curCorners[x]].G_p = X_hat_t[i:i+2]
    #    i = i+2

    # Perform path planning algorithm to reach goal
    CurrentState, theta_desired, EntrancePoint = path_planning.bug2(x_hat_t, CurrentState, m, b, goal, dists, EntrancePoint)
    MotorController(x_hat_t, min(dists), theta_desired)

    print("x: ", x_hat_t)

    # Check if reach goal
    if reachGoal(x_hat_t):
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
        break

    pass

# Enter here exit cleanup code.
