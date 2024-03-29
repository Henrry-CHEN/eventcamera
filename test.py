import h5py
import numpy
import numpy as np
import cv2
from EventFrameIterator import EventFrameIterator
import ST_ex
import ST_before
import camera
import datetime
from PIL import Image
import time
from dv import AedatFile
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import alphashape
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.colors as mcolors
filename_last ="C:/Users/12816/PycharmProjects/pythonProject/event7.aedat4"
filename = 'indoor_flying1_data.hdf5'
'''
num_disp = 32
block_size = 3
stereo_SGBM = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 1 * block_size ** 2,
    P2=32 * 1 * block_size ** 2,
    disp12MaxDiff=-1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=64,
)
'''
def frametoimage(frame):
    img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    # 归一化正值和负值
    max_positive = np.max(frame[frame > 0])
    max_negative = np.min(frame[frame < 0])

    # 映射颜色：正值映射到红色，负值映射到绿色
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i, j] > 0:
                # 根据正值的大小调整红色的强度
                intensity = int(255 * (frame[i, j] / max_positive))
                img[i, j] = [0, 0, intensity]  # 红色
            elif frame[i, j] < 0:
                # 根据负值的大小调整绿色的强度
                intensity = int(255 * (-frame[i, j] / max_negative))
                img[i, j] = [0, intensity, 0]  # 绿色
    return img


def remove_outliers(data, threshold_factor=0.8):
    """
    从给定数据中剔除相差过大的点。

    参数:
    - data: 数据序列 (list或numpy数组)。
    - threshold_factor: 用于定义异常值的标准差倍数阈值。

    返回:
    - 剔除异常值后的数据序列。
    """
    # 转换为numpy数组以便进行计算
    data = np.array(data)

    # 计算平均值和标准差
    mean = np.mean(data)
    std_dev = np.std(data)

    # 计算阈值
    threshold = threshold_factor * std_dev

    # 筛选出不是异常的点
    filtered_data = data[np.abs(data - mean) <= threshold]

    return filtered_data
def main():
    with AedatFile(filename_last) as f:
        '''
        event_left_file = 'map_left_array.npy'
        event_left_array = np.load(event_left_file, allow_pickle=True)
        event_right_file = 'map_right_array.npy'
        event_right_array = np.load(event_right_file, allow_pickle=True)
        #event_left_frames = EventFrameIterator(event_left_array, 30, (260, 346))
        #event_right_frames = EventFrameIterator(event_right_array, 30, (260, 346))
        mapLx = np.loadtxt('indoor_flying_left_x_map.txt')
        mapLx = mapLx.astype(np.float32)
        mapLy = np.loadtxt('indoor_flying_left_y_map.txt')
        mapLy = mapLy.astype(np.float32)
        mapRx = np.loadtxt('indoor_flying_right_x_map.txt')
        mapRx = mapRx.astype(np.float32)
        mapRy = np.loadtxt('indoor_flying_right_y_map.txt')
        mapRy = mapRy.astype(np.float32)
        '''

        #np.save('map_left_array.npy', event_left_array)
        #np.save('map_right_array.npy', event_right_array)

        #event_matched=ST_ex.ST_EventMatcher(f['undistortedEvents'],f['undistortedEvents_1'],2,(480,640),2)
        event_matched = ST_before.ST_EventMatcher(f['undistortedEvents'], f['undistortedEvents_1'], 5, (480, 640), 2)
        #event_left_frames = EventFrameIterator(event_left_array, 60, (260, 346))
        #event_right_frames = EventFrameIterator(event_right_array, 60, (260, 346))
        #for frame1,frame2  in zip(event_left_frames,event_right_frames):
        i=0
        for frame1, frame2 in event_matched:
            starttime = time.perf_counter()
            gray_left_im = frametoimage(frame1)
            gray_right_im = frametoimage(frame2)
            alpha = 0.05  # Alpha值的选择取决于点集和期望的形状精细度
            endtime = time.perf_counter()
            timediff = endtime - starttime
            print(timediff)
            cmap = plt.get_cmap('viridis')


            # 可视化结果
            #gray_left_im = cv2.normalize(frame1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            #gray_right_im = cv2.normalize(frame2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            #smoothed_depth_map = cv2.GaussianBlur(gray_right_im, (5, 5), 0)
            #disparity = stereo_SGBM.compute(gray_left_im, gray_right_im)
            #disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #disparity = colormap(disparity / disparity.max())

            cv2.imshow('left', gray_left_im)
            cv2.imshow('right', gray_right_im)
            #cv2.imshow('disp', disparity)
            cv2.waitKey(1)



    '''
    event_left_frames = EventFrameIterator(event_left_array, 30, (260, 346))
    event_right_frames = EventFrameIterator(event_right_array, 30, (260, 346))
    for event_left_frame ,event_right_frame in zip(event_left_frames,event_right_frames):
        gray_left_im = cv2.normalize(event_left_frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        gray_right_im = cv2.normalize(event_right_frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        disparity = stereo_SGBM.compute(gray_left_im, gray_right_im)
        # disparity = stereo_BM.compute(img_rtf_l, img_rtf_r)
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
        cv2.imshow('disparity', disparity)
        cv2.imshow('left', gray_left_im)
        cv2.imshow('right', gray_right_im)
        cv2.waitKey(1)
    '''

if __name__ == "__main__":
    main()