import cv2
import numpy as np
from PIL import Image
#畸变矩阵
distortionCoefficientsL = np.array([-0.478552209587669,1.505571974393130,0.002748662121190,0.004881929481332,-4.821846238142730])
#内部参数
cameraMatrixL = np.array([[811.345016085786,-1.91510694838280,323.972460745020],
[0.,	811.412182617847,	258.526911475435],
[0., 0., 1.]])
newCameraMatrixL = cameraMatrixL
#畸变矩阵
distortionCoefficientsR = np.array([-0.404274787167942,1.507595421004656,0.003123359694831,0.003252164386124,-13.324876208011640])
#内部参数
cameraMatrixR = np.array([[834.250616292289,0.0362831681933432,309.178417154997],
[0,832.689896197984,247.552076110513],[0.,0.,1.]])
newCameraMatrixR = cameraMatrixR

# Stereo params from MATLAB
Rot = np.array([[0.999963131347512,-0.001442506890614,0.008464934704390],[0.001436043240002,0.999998672740830,7.696079459699394e-04],[-0.008466033633993,-7.574235593027199e-04,0.999963875639545]]).T
Trns = np.array([[-62.852988353155716], [-0.220718628615023], [1.124658293578811]])
imgSize = (640,480)

R_L, R_R, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, imgSize, Rot, Trns, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, R_L, proj_mat_l, imgSize, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, R_R, proj_mat_r, imgSize, cv2.CV_32FC1)