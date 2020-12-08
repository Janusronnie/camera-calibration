#! usr/bin/env python3

# takes in the calibrated translation and rotation matrix with the recorded transform from the arm and gives the
# location of the end effector in the world frame.

import numpy as np
import csv
import sys
import os
import pandas as pd

directory = os.path.dirname(os.path.realpath(__file__))

# sets up matrices to read the stored matrices into
tran_mat = np.zeros((4, 4))
translation_mat = np.zeros((4, 4))
rotation_mat = np.zeros((4, 4))

########################################################################################################################
# Read in the translation rotation and transformation matrices
########################################################################################################################

with open(directory + '/TranslationMatrix.csv', newline='') as f:
    reader = csv.reader(f)
    for j, row in enumerate(reader):
        for i, col in enumerate(row):
            translation_mat[j][i] = float(col)


with open(directory + '/RotationMatrix.csv', newline='') as f:
    reader = csv.reader(f)
    for j, row in enumerate(reader):
        for i, col in enumerate(row):
            rotation_mat[j][i] = float(col)


with open(directory + '/test_data/Matrices/TransformMatrix_9.0.csv', newline='') as f:
    reader = csv.reader(f)
    for j, row in enumerate(reader):
        for i, col in enumerate(row):
            tran_mat[j][i] = float(col)


########################################################################################################################
# Calculate the end effector location in the world frame
########################################################################################################################

loc = rotation_mat @ translation_mat @ tran_mat @ np.transpose([0, 0, 0, 1])
print('The location of the end effector')
print(loc)
print('\n')

# load the ArUco Markers information
ArUco_data = pd.read_csv('data_file_9.csv')
palm_data = ArUco_data[ArUco_data.location == 'palm']
palm_data = np.array(palm_data)
print('Palm ArUco Marker Location')
print(palm_data[0, [1, 2, 3]])
print('\n')

# create the rotation matrix of the palm in x-axis
R_x = np.eye(4)
R_x[1, 1] = np.cos(palm_data[0, 4])
R_x[1, 2] = -np.sin(palm_data[0, 4])
R_x[2, 1] = np.sin(palm_data[0, 4])
R_x[2, 2] = np.cos(palm_data[0, 4])

# create the rotation matrix of the palm in y-axis
R_y = np.eye(4)
R_y[0, 0] = np.cos(-palm_data[0, 5])
R_y[0, 2] = np.sin(-palm_data[0, 5])
R_y[2, 0] = -np.sin(-palm_data[0, 5])
R_y[2, 2] = np.cos(-palm_data[0, 5])

# create the rotation matrix of the palm in z-axis
R_z = np.eye(4)
R_z[0, 0] = np.cos(-palm_data[0, 6])
R_z[0, 1] = -np.sin(-palm_data[0, 6])
R_z[1, 0] = np.sin(-palm_data[0, 6])
R_z[1, 1] = np.cos(-palm_data[0, 6])

# calculate the rotation matrix of palm
R_P = R_x @ R_y @ R_z

# load the translation and rotation matrix from the end effector to palm
T = pd.read_csv('EE_to_Palm_Translation_Matrix.csv', header=None)
R = pd.read_csv('EE_to_Palm_Rotation_Matrix.csv', header=None)
T = np.array(T)
R = np.array(R)

# calculate the physical location of palm
phy_palm = R_P @ T @ R @ loc
print('Palm Physical Location')
print(phy_palm)