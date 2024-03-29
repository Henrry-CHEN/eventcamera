import h5py
import numpy
import numpy as np
import cv2
from EventFrameIterator import EventFrameIterator
import camera
from PIL import Image
import camera_1
import time
class ST_EventMatcher:
    def __init__(self,  events_l,events_r,frameTime, shape, time_threshold=2, spatial_threshold=1,disparity=50):
        self.time_threshold = time_threshold*1000  # 时间戳差小于1ms的阈值
        self.spatial_threshold = spatial_threshold  # 到基线的距离小于1的阈值
        self.frameTime = frameTime
        self.shape = shape
        self.events_l=events_l
        self.events_r = events_r
        self.disaprity=disparity
        self.spike_map = np.zeros((shape[0], shape[1], 2, 2), dtype=np.int64)
        self._iterator_l = iter(events_l)
        self._iterator_r = iter(events_r)
    def __iter__(self):
        return self

    def __next__(self):
        frame1 = np.zeros(self.shape)
        frame2 = np.zeros(self.shape)
        first_event_time = -1
        count=0
        frame_l=[]
        frame_r=[]
        frame_l_match=[]
        for event_r in self._iterator_r:
            #starttime_1 = time.perf_counter()
            x_r_u=event_r.x
            y_r_u=event_r.y
            x_r,y_r=int(np.round(camera_1.leftMapX[event_r.y, event_r.x])),int(np.round(camera_1.leftMapY[event_r.y, event_r.x]))
            if first_event_time < 0:
                first_event_time = event_r.timestamp
            if x_r< self.shape[1] and y_r< self.shape[0] and x_r> 0 and y_r > 0:#保存时间尖峰图
                self.spike_map[y_r,x_r][0][0]=event_r.timestamp
                self.spike_map[y_r,x_r][0][1]=event_r.polarity
            if y_r<self.shape[0] and x_r<self.shape[1] and x_r>0 and y_r>0:
                frame2[y_r, x_r] += 1 if event_r.polarity else -1 #保存在一个frame里面
                count+=1
            if event_r.timestamp - first_event_time >= self.frameTime * 1000:
                break
            frame_r.append(event_r)
            #endtime_1 = time.perf_counter()
            #timediff_1 = endtime_1 - starttime_1
            #print(timediff_1)
        first_event_time = -1

        for event_l in self._iterator_l:
            starttime=time.perf_counter()
            if first_event_time < 0:
                first_event_time = event_l.timestamp
            #if event_l.y < self.shape[0] and event_l.x < self.shape[1] and event_l.y > 0 and event_l.x > 0:
                #event_match=self.match_event(event_l)
            #else:
                #event_match=[0,0,0,0]
            if event_l.y<self.shape[0] and event_l.x<self.shape[1] and event_l.y>0 and event_l.x>0:
                frame1[event_l.y, event_l.x] += 1 if event_l.polarity else -1
             #   count+=1
            if event_l.timestamp - first_event_time >= self.frameTime * 1000:
                break
            frame_l.append(event_l)
            #if event_match[2]!=0:
                #frame_l_match.append(event_match)
            endtime = time.perf_counter()
            timediff=endtime-starttime
            print(timediff)
       # same_frame=find_matching_coordinates(frame1,frame2)
        #match_list=self.match_events(frame_l,frame_r)
        len1=len(frame_l)
        lenr=len(frame_r)
        #leng=len(match_list)
        lenm=len(frame_l_match)
        #for event_m in frame_l_match:
            #if event_m[1] < self.shape[0] and event_m[0] < self.shape[1] and event_m[1] > 0 and event_m[0] > 0:
                #frame2[event_m[1], event_m[0]] += 1 if event_m[3] else -1
        #for event_ma_l,event_ma_r in match_list:
            #if event_ma_l[1] < self.shape[0] and event_ma_l[0] < self.shape[1] and event_ma_l[1] > 0 and event_ma_l[0] > 0:
                #frame1[event_ma_l[1], event_ma_l[0]] += 1 if event_ma_l[3] else -1
            #if event_ma_r[1] < self.shape[0] and event_ma_r[0] < self.shape[1] and event_ma_r[1] > 0 and event_ma_r[0] > 0:
                #frame2[event_ma_r[1], event_ma_r[0]] += 1 if event_ma_r[3] else -1
        if first_event_time < 0:
            raise StopIteration
        return frame1,frame2

    def match_event(self, left_event):
        x = left_event.x
        y = left_event.y
        t = left_event.timestamp
        p = left_event.polarity
        all_cost = 4
        match_event = [0, 0, 0, 0]
        for i in range(-1, 1):
            spike_x = 0
            for spike in self.spike_map[y + i]:
                time_difference = abs(spike[0][0] - t)
                disparity_distance = abs(x - spike_x)
                if p == spike[0][1] and disparity_distance <= self.disaprity and time_difference <= self.time_threshold:
                    cost = i + float(time_difference / 1000000)
                    if cost < all_cost:
                        all_cost = cost
                        match_event[0] = spike_x
                        match_event[1] = y + i
                        match_event[2] = spike[0][0]
                        match_event[3] = p
                spike_x += 1
        return match_event

