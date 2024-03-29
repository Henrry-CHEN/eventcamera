import cv2
import numpy as np
from PIL import Image


#畸变矩阵
distortionCoefficientsL = np.array([-0.048031442223833355, 0.011330957517194437, -0.055378166304281135,0.021500973881459395])
#distortionCoefficientsL = np.array([0.0,0.0,0.0,0.0])
#内部参数
cameraMatrixL = np.array([[226.38018519795807,0.,173.6470807871759],
                        [0.,226.15002947047415,133.73271487507847],
                        [0., 0., 1.]])
newCameraMatrixL = cameraMatrixL
#畸变矩阵
distortionCoefficientsR = np.array([-0.04846669832871334, 0.010092844338123635, -0.04293073765014637,0.005194706897326005])
#distortionCoefficientsR = np.array([0.0,0.0,0.0,0.0])
#内部参数
cameraMatrixR = np.array([[226.0181418548734,0.,174.5433576736815],
                            [0.,225.7869434267677,124.21627572590607],
                          [0.,0.,1.]])
newCameraMatrixR = cameraMatrixR
imgSize = (346,260)
proj_mat_l_1=np.array([    [199.6530123165822, 0.0, 177.43276376280926, 0.0]
  , [0.0, 199.6530123165822, 126.81215684365904, 0.0]
  , [0.0, 0.0, 1.0, 0.0]])
proj_mat_r_1=np.array(    [[199.6530123165822, 0.0, 177.43276376280926, -19.941771812941038]
  , [0.0, 199.6530123165822, 126.81215684365904, 0.0]
  , [0.0, 0.0, 1.0, 0.0]])

R_L= np.array( [[0.999877311526236, 0.015019439766575743, -0.004447282784398257]
  , [-0.014996983873604017, 0.9998748347535599, 0.005040367172759556]
  , [0.004522429630305261, -0.004973052949604937, 0.9999774079320989]])

R_R=np.array(    [[0.9999922706537476, 0.003931701344419404, -1.890238450965101e-05]
  , [-0.003931746704476347, 0.9999797362744968, -0.005006836150689904]
  , [-7.83382948021244e-07, 0.0050068717705076754, 0.9999874655386736]])

leftMapX, leftMapY = cv2.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, R_L, newCameraMatrixL, imgSize, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, R_R, newCameraMatrixL, imgSize, cv2.CV_32FC1)