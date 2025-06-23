import numpy as np
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)

from common.angle import Angle

class Attitude:
    def __init__(self):
        self.qbn = np.zeros(4)
        self.cbn = np.zeros((3,3))
        self.euler = np.zeros(3)

class PVA:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.att = Attitude()

class ImuError:
    def __init__(self):
        self.gyrbias = np.zeros(3)
        self.accbias = np.zeros(3)
        self.gyrscale = np.zeros(3)
        self.accscale=  np.zeros(3)

class NavState:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.imuerror = ImuError()

class ImuNoise:
    def __init__(self):
        self.gyr_arw = np.zeros(3)
        self.acc_vrw = np.zeros(3)
        self.gyrbias_std = np.zeros(3)
        self.accbias_std = np.zeros(3)
        self.gyrscale_std = np.zeros(3)
        self.accscale_std = np.zeros(3)
        self.corr_time = 0.0

class GINSOptions:
    def __init__(self):
        # 初始状态和状态标准差
        # initial state and state standard deviation
        self.initstate = NavState()
        self.initstate_std = NavState()

        # IMU噪声参数
        # imu noise parameters
        self.imunoise = ImuNoise()
        
        # 安装参数
        # install parameters
        self.antlever = np.zeros(3)
    
    def print_options(self):
        print("---------------KF-GINS Options:---------------")

        # 打印初始状态
        # print initial state
        print(" - Initial State: ")
        print(f"\t- initial position: [{self.initstate.pos[0] * Angle.R2D:.12f}  {self.initstate.pos[1] * Angle.R2D:.12f}  {self.initstate.pos[2]:.6f}] [deg, deg, m]")
        print(f"\t- initial velocity: {self.initstate.vel} [m/s]")
        print(f"\t- initial attitude: {self.initstate.euler * Angle.R2D} [deg]")
        print(f"\t- initial gyrbias : {self.initstate.imuerror.gyrbias * Angle.R2D * 3600} [deg/h]")
        print(f"\t- initial accbias : {self.initstate.imuerror.accbias * 1e5} [mGal]")
        print(f"\t- initial gyrscale: {self.initstate.imuerror.gyrscale * 1e6} [ppm]")
        print(f"\t- initial accscale: {self.initstate.imuerror.accscale * 1e6} [ppm]")

        # 打印初始状态标准差
        # print initial state STD
        print(" - Initial State STD: ")
        print(f"\t- initial position std: {self.initstate_std.pos} [m]")
        print(f"\t- initial velocity std: {self.initstate_std.vel} [m/s]")
        print(f"\t- initial attitude std: {self.initstate_std.euler * Angle.R2D} [deg]")
        print(f"\t- initial gyrbias std: {self.initstate_std.imuerror.gyrbias * Angle.R2D * 3600} [deg/h]")
        print(f"\t- initial accbias std: {self.initstate_std.imuerror.accbias * 1e5} [mGal]")
        print(f"\t- initial gyrscale std: {self.initstate_std.imuerror.gyrscale * 1e6} [ppm]")
        print(f"\t- initial accscale std: {self.initstate_std.imuerror.accscale * 1e6} [ppm]")

        # 打印IMU噪声参数
        # print IMU noise parameters
        print(" - IMU noise: ")
        print(f"\t- arw: {self.imunoise.gyr_arw * Angle.R2D * 60} [deg/sqrt(h)]")
        print(f"\t- vrw: {self.imunoise.acc_vrw * 60} [m/s/sqrt(h)]")
        print(f"\t- gyrbias  std: {self.imunoise.gyrbias_std * Angle.R2D * 3600} [deg/h]")
        print(f"\t- accbias  std: {self.imunoise.accbias_std * 1e5} [mGal]")
        print(f"\t- gyrscale std: {self.imunoise.gyrscale_std * 1e6} [ppm]")
        print(f"\t- accscale std: {self.imunoise.accscale_std * 1e6} [ppm]")
        print(f"\t- correlation time: {self.imunoise.corr_time / 3600.0} [h]")

        # 打印GNSS天线杆臂
        # print GNSS antenna leverarm
        print(f" - Antenna leverarm: {self.antlever} [m]\n")
