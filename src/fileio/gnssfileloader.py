import os
import sys
import numpy as np
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)

from common.types import GNSS
from common.angle import Angle

class GnssFileLoader:
    def __init__(self, filename: str):
        self.data_ = np.genfromtxt(filename, delimiter=None)
        self.index = 0

    def next(self):
        if self.index >= self.data_.shape[0]:
            return None
        data_ = self.data_[self.index, :]
        gnss_ = GNSS()
        gnss_.time = data_[0]
        gnss_.blh = np.array(data_[1:4])
        gnss_.std = np.array(data_[4:7])
        gnss_.blh[0] *= Angle.D2R
        gnss_.blh[1] *= Angle.D2R
        self.index += 1

        return gnss_

    def starttime(self):
        return self.data_[0, 0]

    def endtime(self):
        return self.data_[-1, 0]
        
    def isEof(self):
        return self.index >= self.data_.shape[0]
