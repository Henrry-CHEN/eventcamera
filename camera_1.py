import cv2
import numpy as np
from PIL import Image
#畸变矩阵
distortionCoefficientsL = np.array([-0.2935,-0.3440,-1.7559e-04, -0.0015,1.5067])
#内部参数
cameraMatrixL = np.array([[789.3270,0.,332.1650],
[0.,789.9218,261.8494],
[0., 0., 1.]])
newCameraMatrixL = cameraMatrixL
#畸变矩阵
distortionCoefficientsR = np.array([-0.3435,0.0741,-8.6562e-04,-9.9706e-04,0.4352])
#内部参数
cameraMatrixR = np.array([[788.2710,0.,324.8369],
[0.,788.6607,248.3408],[0.,0.,1.]])
newCameraMatrixR = cameraMatrixR

# Stereo params from MATLAB
Rot = np.array([[1.0000, 0.0011, 0.0090],[-0.0011, 1.0000, -4.9004e-04],[-0.0090, 4.8040e-04, 1.0000]]).T
Trns = np.array([[-62.5813], [-0.3437], [0.0488]])
imgSize = (640,480)

R_L, R_R, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(newCameraMatrixL, distortionCoefficientsL, newCameraMatrixR, distortionCoefficientsR, imgSize, Rot, Trns, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

leftMapX, leftMapY = cv2.initUndistortRectifyMap(newCameraMatrixL, distortionCoefficientsL, R_L, proj_mat_l, imgSize, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(newCameraMatrixR, distortionCoefficientsR, R_R, proj_mat_r, imgSize, cv2.CV_32FC1)

width=640
height=480
#两张图片水平拼接
def merge_images(image1, image2):
    # 打开两张图片


    # 创建一个新的空白图片，尺寸为两张图片横向拼接后的尺寸
    merged_image = Image.new('RGB', (width * 2, height))
    # merged_image.save("ttt.jpg")全黑图片

    # 将两张图片拼接到新图片上
    merged_image.paste(image1, (0, 0))#左上角
    merged_image.paste(image2, (width, 0))
    # 保存合成后的图片
    return merged_image

# 示例用法

