import numpy as np
import cv2
import cv2.aruco as aruco
import math


marker_size = 3.6  # cm

# Marker IDs
top_right_id = 12
top_left_id = 11
bottom_left_id = 10
bottom_right_id = 13
end_effector_id = 14
palm_id = 1
center_id = 0

# --- Get the camera calibration path
calib_path = ""
camera_matrix = np.load(calib_path + 'camera_mtx.npy')
camera_distortion = np.load(calib_path + 'dist_mtx.npy')

# Define Aruco Dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
parameters = aruco.DetectorParameters_create()

# Lists for storing marker positions
top_right = []
top_left = []
bottom_left = []
bottom_right = []
end_effector = []
center = []
palm = []

# --- 180 deg rotation matrix around the x axis
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

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

    #tranfermation matrix
    tranf = np.eye(4)
    tranf[1, 1] = -1
    tranf[2, 2] = -1
    tranf[2, 3] = 80

    pos = np.insert(tvecs[i],[3], 1, axis=None)
    trans = np.dot(tranf, pos)

    str_position = name + " Position x=%4.1f  y=%4.1f  z=%4.1f" % (
        trans[0], trans[1], trans[2])

    # -- Obtain the rotation matrix tag->camera
    R_ct = np.matrix(cv2.Rodrigues(rvecs[i])[0])
    R_tc = R_ct.T

    # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
    roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

    str_attitude = name + " Euler Angle x=%4.1f  y=%4.1f  z=%4.1f" % (
        math.degrees(roll_marker), math.degrees(pitch_marker),
        math.degrees(yaw_marker))

    aruco.drawAxis(frame, camera_matrix, camera_distortion, rvecs[i], tvecs[i], 5)

    return str_position, str_attitude

# --- Capture the videocamera (this may also be a video or a picture)
cap = cv2.VideoCapture(0)
# -- Set the camera size as the one it was calibrated with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# -- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

while True:

    # -- Read the camera frame
    ret, frame = cap.read()

    # -- Convert in gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # -- remember, OpenCV stores color images in Blue, Green, Red

    # -- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                                                 cameraMatrix=camera_matrix, distCoeff=camera_distortion)

    if ids is not None:

        rvecs, tvecs, _objPonits = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        for i in range(ids.size):

            # Save end-effector marker pose
            if ids[i] == end_effector_id:
                [str_position, str_attitude] = read_inf(ids, i)

                cv2.putText(frame, str_position, (0, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(frame, str_attitude, (0, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if ids[i] == palm_id:
                [str_position, str_attitude] = read_inf(ids, i)

                cv2.putText(frame, str_position, (0, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(frame, str_attitude, (0, 200), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if ids[i] == center_id:
                [str_position, str_attitude] = read_inf(ids, i)

                cv2.putText(frame, str_position, (0, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if ids[i] == top_right_id:
                [str_position, str_attitude] = read_inf(ids, i)

                cv2.putText(frame, str_position, (0, 300), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if ids[i] == top_left_id:
                [str_position, str_attitude] = read_inf(ids, i)

                cv2.putText(frame, str_position, (0, 350), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if ids[i] == bottom_left_id:
                [str_position, str_attitude] = read_inf(ids, i)

                cv2.putText(frame, str_position, (0, 400), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if ids[i] == bottom_right_id:
                [str_position, str_attitude] = read_inf(ids, i)

                cv2.putText(frame, str_position, (0, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        aruco.drawDetectedMarkers(frame, corners)

    # --- Display the frame
    cv2.imshow('frame', frame)

    # --- use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
