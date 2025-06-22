import os
import sys
import numpy as np
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)

from common.types_my import IMU


class ImuFileLoader:
    def __init__(self, filename:str, columns:int, rate:int):
        self.dt_ = 1.0 / float(rate)
        self.imu_ = IMU()
        self.data_ = np.genfromtxt(filename, delimiter=None)
        self.index = 0

    def next(self):

        if self.index >= self.data_.shape[0]:
            return None
        data_ = self.data_[self.index, :]
        pre_time = self.imu_.time
        self.imu_.time = data_[0]
        self.imu_.dtheta = np.array(data_[1:4])
        self.imu_.dvel = np.array(data_[4:7])
        dt = self.imu_.time - pre_time
        if dt < 0.1:
            self.imu_.dt = dt
        else:
            self.imu_.dt = self.dt_

        self.index += 1

        return self.imu_
    
    def starttime(self):
        return self.data_[0, 0]
    
    def endtime(self):
        return self.data_[-1, 0]
    
    def isEof(self):
        return self.index >= self.data_.shape[0]
