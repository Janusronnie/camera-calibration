import cv2
from cv2 import aruco
import numpy as np
import math

img = cv2.imread('Test_10.jpg')

marker_size = 3.6  # cm

# Marker IDs
top_right_id = 12
top_left_id = 11
bottom_left_id = 10
bottom_right_id = 13
end_effector_id = 14
palm_id = 1
center_id = 0

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
parameters = aruco.DetectorParameters_create()

# --- Get the camera calibration path
calib_path = ""
camera_matrix = np.load(calib_path + 'camera_mtx.npy')
camera_distortion = np.load(calib_path + 'dist_mtx.npy')

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, camera_matrix, camera_distortion, None, newcameramtx)

# crop the image
x, y, w, h = roi
img = dst[y:y + h, x:x + w]

# Detect the markers.
corners, ids, rejected = aruco.detectMarkers(image=img, dictionary=aruco_dict, parameters=parameters,
                                             cameraMatrix=camera_matrix, distCoeff=camera_distortion)

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

def read_inf(ids, i):

    id = [0, 1, 10, 11, 12, 13, 14]
    location = ['center', 'palm', 'bottom_left', 'top_left', 'top_right', 'bottom_right', 'end_effector']
    dict={}
    for n in range(len(id)):
        dict[id[n]] = location[n]

    name = dict[ids[i, 0]]

    # --- 180 deg rotation matrix around the x axis
    R_flip = np.zeros((3, 3), dtype=np.float32)
    R_flip[0, 0] = 1.0
    R_flip[1, 1] = -1.0
    R_flip[2, 2] = -1.0

    trans = tvecs[i]
    str_position = name + " Position x=%4.1f  y=%4.1f  z=%4.1f" % (
        trans[0, 0], trans[0, 1], trans[0, 2])

    # -- Obtain the rotation matrix tag->camera
    R_ct = np.matrix(cv2.Rodrigues(rvecs[i])[0])
    R_tc = R_ct.T

    # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
    roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

    str_attitude = name + " Euler Angle x=%4.1f  y=%4.1f  z=%4.1f" % (
        math.degrees(roll_marker), math.degrees(pitch_marker),
        math.degrees(yaw_marker))

    aruco.drawAxis(img, camera_matrix, camera_distortion, rvecs[i], tvecs[i], 5)

    return str_position, str_attitude

font = cv2.FONT_HERSHEY_PLAIN

rvecs, tvecs, _objPonits = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

for i in range(ids.size):

    # Save end-effector marker pose
    if ids[i] == end_effector_id:

        [str_position, str_attitude] = read_inf(ids, i)

        cv2.putText(img, str_position, (0, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(img, str_attitude, (0, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if ids[i] == palm_id:

        [str_position, str_attitude] = read_inf(ids, i)

        cv2.putText(img, str_position, (0, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(img, str_attitude, (0, 200), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if ids[i] == center_id:

        [str_position, str_attitude] = read_inf(ids, i)

        cv2.putText(img, str_position, (0, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if ids[i] == top_right_id:

        [str_position, str_attitude] = read_inf(ids, i)

        cv2.putText(img, str_position, (0, 300), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if ids[i] == top_left_id:

        [str_position, str_attitude] = read_inf(ids, i)

        cv2.putText(img, str_position, (0, 350), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if ids[i] == bottom_left_id:

        [str_position, str_attitude] = read_inf(ids, i)

        cv2.putText(img, str_position, (0, 400), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if ids[i] == bottom_right_id:

        [str_position, str_attitude] = read_inf(ids, i)

        cv2.putText(img, str_position, (0, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


out = aruco.drawDetectedMarkers(img, corners)
cv2.imwrite('result.png', out)
cv2.imshow("out",out)
cv2.waitKey(0)
cv2.destroyAllWindows()

