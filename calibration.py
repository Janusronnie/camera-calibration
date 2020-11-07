#!/usr/bin/env python

import numpy as np
import cv2
import glob

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

#prepare object points,like (0,0,0), (2,0,0) ...., (6,5,0)
object_p = np.zeros((6*9,3), np.float32)
object_p[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

#Array to store object points and image points from all the images
object_points = [] # 3D point in real world space
img_points = [] # 2D points in image plane

#Square size
size = 0.025 #m or 25 cm

images = glob.glob('C:/Users/35552/Desktop/Courses at OSU/ROB514/Final Project/camera calibration/image_processing/*.jpg')

for fname in images:
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

	# If found, add object points, image points (after refining them)
	if ret == True:
		object_points.append(object_p*size)

		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		img_points.append(corners2)

		#Draw and display the corners
		img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
		cv2.imshow('img',img)
		cv2.waitKey(500)

		# Get all the relevant matrices
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, gray.shape[::-1],None,None)

		print('Camera Matrix:')
		print(mtx)
		np.save('camera_mtx.npy', mtx)

		print('Distortion Matrix:')
		print(dist)
		np.save('dist_mtx.npy', dist)

		print('Rotation Matrix:')
		print(rvecs)
		np.save('rotation_mtx.npy', rvecs)

		print('Translation Matrix:')
		print(mtx)
		np.save('translation_mtx.npy', tvecs)

		h, w = img.shape[:2]
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

		# undistort
		dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

		# crop the image
		x, y, w, h = roi
		dst = dst[y:y + h, x:x + w]
		cv2.imwrite('calibresult.png', dst)

