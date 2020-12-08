import cv2
from cv2 import aruco
import numpy as np
import math
import glob

# calculate the orientation of ArUco Markers
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# Read the location and orientation of the ArUco Markers
def read_inf(ids, i):
    # make a dictionary of the ArUco Markers
    id = [1, 14]
    location = ['palm', 'end_effector']
    dict={}
    for n in range(len(id)):
        dict[id[n]] = location[n]

    name = dict[ids[i, 0]]

    # --- 180 deg rotation matrix around the x axis
    R_flip = np.zeros((3, 3), dtype=np.float32)
    R_flip[0, 0] = 1.0
    R_flip[1, 1] = -1.0
    R_flip[2, 2] = -1.0

    # transformation matrix (from camera frame to world frame)
    #change unit from cm to m
    tranf = np.eye(4)
    tranf[1, 1] = -1
    tranf[0, 3] = 0.015
    tranf[2, 2] = -1
    tranf[1, 3] = 0.01
    tranf[2, 3] = 0.8

    # make the location matrix of ArUco Marker
    pos = np.insert(tvecs[i],[3], 100, axis=None) / 100
    # get the world frame location
    trans = np.dot(tranf, pos)
    # get the output of the location
    str_position = name + " Position x=%4.3f  y=%4.3f  z=%4.3f" % (
        trans[0], trans[1], trans[2])


    # -- Obtain the rotation matrix tag->camera
    R_ct = np.matrix(cv2.Rodrigues(rvecs[i])[0])
    R_tc = R_ct.T

    # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
    roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

    str_attitude = name + " Euler Angle x=%4.1f  y=%4.1f  z=%4.1f" % (
        math.degrees(roll_marker), math.degrees(pitch_marker),
        math.degrees(yaw_marker))

    # draw axis
    aruco.drawAxis(img, camera_matrix, camera_distortion, rvecs[i], tvecs[i], 1.5)

    rotates = [math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker)]

    return str_position, str_attitude, trans, rotates

def rigid_transform_3D(loc_e, loc_p, rot_e, rot_p):

    # obtain the location and orientation differences of the end effector and palm
    diff_loc = []
    diff_rot = []
    for i in range(loc_e.shape[0]):
        diff_loc.append(loc_e[i, :] - loc_p[i, :])
        diff_rot.append(rot_e[i, :] - rot_p[i, :])
    diff_loc = np.array(diff_loc)
    diff_rot = np.array(diff_rot)

    # obtain the average matrix
    transf = sum(diff_loc) / loc_e.shape[0]
    rotate = sum(diff_rot) / rot_e.shape[0]

    # create the rotation matrix of x-axis
    R_x = np.eye(4)
    R_x[1, 1] = np.cos(rotate[0])
    R_x[1, 2] = -np.sin(rotate[0])
    R_x[2, 1] = np.sin(rotate[0])
    R_x[2, 2] = np.cos(rotate[0])

    # create the rotation matrix of y-axis
    R_y = np.eye(4)
    R_y[0, 0] = np.cos(rotate[1])
    R_y[0, 2] = np.sin(rotate[1])
    R_y[2, 0] = -np.sin(rotate[1])
    R_y[2, 2] = np.cos(rotate[1])

    # create the rotation matrix of z-axis
    R_z = np.eye(4)
    R_z[0, 0] = np.cos(rotate[2])
    R_z[0, 1] = -np.sin(rotate[2])
    R_z[1, 0] = np.sin(rotate[2])
    R_z[1, 1] = np.cos(rotate[2])

    # calculate the rotation matrix
    R = R_x @ R_y @ R_z

    # create the translation matrix
    T = np.eye(4)
    T[0, 3] = transf[0]
    T[1, 3] = transf[1]
    T[2, 3] = transf[2]

    return T, R

# the size of ArUco Markers
marker_size = 3.6  # cm

# Marker IDs
end_effector_id = 14
palm_id = 1

# Load the dictionary of the ArUco Markers
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
parameters = aruco.DetectorParameters_create()

# --- Get the camera calibration path
calib_path = ""
camera_matrix = np.load(calib_path + 'camera_mtx.npy')
camera_distortion = np.load(calib_path + 'dist_mtx.npy')

train_images = glob.glob('EE_Palm/*.jpg')

loc_e = []
loc_p = []
rot_e = []
rot_p = []

num_file = len(train_images)

for n in range(num_file):
    # input image
    img = cv2.imread(train_images[n])

    # Detect the markers.
    corners, ids, rejected = aruco.detectMarkers(image=img, dictionary=aruco_dict, parameters=parameters,
                                                 cameraMatrix=camera_matrix, distCoeff=camera_distortion)

    # setting the position of text box
    font = cv2.FONT_HERSHEY_PLAIN
    # get the rotation and translation vectors
    rvecs, tvecs, _objPonits = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
    # identify each ArUco Marker
    for i in range(ids.size):

        # Do the output for each ArUco Marker
        if ids[i] == end_effector_id:

            [str_position, str_attitude, trans, rotates] = read_inf(ids, i)

            cv2.putText(img, str_position, (0, 25), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.putText(img, str_attitude, (0, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            loc_e.append(trans[0:3])
            rot_e.append(rotates)


        if ids[i] == palm_id:

            [str_position, str_attitude, trans, rotates] = read_inf(ids, i)

            cv2.putText(img, str_position, (0, 75), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.putText(img, str_attitude, (0, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            loc_p.append(trans[0:3])
            rot_p.append(rotates)


loc_p = np.array(loc_p)
loc_e = np.array(loc_e)
rot_p = np.array(rot_p)
rot_e = np.array(rot_e)

T, R = rigid_transform_3D(loc_e, loc_p, rot_e, rot_p)

print('EE_to_Palm_Translation_Matrix')
print(T)
print('\nEE_to_Palm_Rotation_Matrix')
print(R)

# save matrices to csv files
np.savetxt('EE_to_Palm_Translation_Matrix.csv', T, delimiter=',')
np.savetxt('EE_to_Palm_Rotation_Matrix.csv', R, delimiter=',')



