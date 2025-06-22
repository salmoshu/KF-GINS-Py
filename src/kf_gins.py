import os
import sys
import yaml
import numpy as np
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)

import kfgins.kf_gins_types as kf
from common.angle import Angle
import common.types_my as ty
import kfgins.gi_engine as gi

with open('../dataset/kf-gins.yaml', 'r',encoding='utf-8') as file:
    config = yaml.safe_load(file)

def LoadOptions():
    ## 读取初始位置(纬度 经度 高程)、(北向速度 东向速度 垂向速度)、姿态(欧拉角，ZYX旋转顺序, 横滚角、俯仰角、航向角)
    options = kf.GINSOptions()
    options.initstate.pos = np.array(config['initpos']) * Angle.D2R
    options.initstate.vel = np.array(config['initvel'])
    options.initstate.euler = np.array(config['initatt']) * Angle.D2R
    options.initstate.pos[2] *= Angle.R2D

    ## 读取IMU误差初始值(零偏和比例因子)
    options.initstate.imuerror.gyrbias = np.array(config['initgyrbias']) * Angle.D2R/3600.0
    options.initstate.imuerror.accbias = np.array(config['initaccbias']) * 1e-5
    options.initstate.imuerror.gyrscale = np.array(config['initgyrscale']) * 1e-6
    options.initstate.imuerror.accscale = np.array(config['initaccscale']) * 1e-6

    ## 读取初始位置、速度、姿态(欧拉角)的标准差
    options.initstate_std.pos = np.array(config['initposstd'])
    options.initstate_std.vel = np.array(config['initvelstd'])
    options.initstate_std.euler = np.array(config['initattstd']) * Angle.D2R

    ## 读取IMU噪声参数
    options.imunoise.gyr_arw = np.array(config['imunoise']['arw'])
    options.imunoise.acc_vrw = np.array(config['imunoise']['vrw'])
    options.imunoise.gyrbias_std = np.array(config['imunoise']['gbstd'])
    options.imunoise.accbias_std = np.array(config['imunoise']['abstd'])
    options.imunoise.gyrscale_std = np.array(config['imunoise']['gsstd'])
    options.imunoise.accscale_std = np.array(config['imunoise']['asstd'])
    options.imunoise.corr_time = config['imunoise']['corrtime']

    ## 读取IMU误差初始标准差,如果配置文件中没有设置，则采用IMU噪声参数中的零偏和比例因子的标准差
    options.initstate_std.imuerror.gyrbias = np.array(config['imunoise']['gbstd']) * Angle.D2R / 3600.0
    options.initstate_std.imuerror.accbias = np.array(config['imunoise']['abstd']) * 1e-5
    options.initstate_std.imuerror.gyrscale = np.array(config['imunoise']['gsstd']) * 1e-6
    options.initstate_std.imuerror.accscale = np.array(config['imunoise']['asstd']) * 1e-6

    ## IMU噪声参数转换为标准单位
    options.imunoise.gyr_arw *= (Angle.D2R / 60.0)
    options.imunoise.acc_vrw /= 60.0
    options.imunoise.gyrbias_std *= (Angle.D2R / 3600.0)
    options.imunoise.accbias_std *= 1e-5
    options.imunoise.gyrscale_std *= 1e-6
    options.imunoise.accscale_std *= 1e-6
    options.imunoise.corr_time *= 3600

    ## GNSS天线杆臂, GNSS天线相位中心在IMU坐标系下位置
    options.antlever = np.array(config['antlever'])
    return options

def imuload(data_,rate,pre_time):
    dt_ = 1.0 / rate
    imu_ = ty.IMU()
    imu_.time = data_[0]
    imu_.dtheta = np.array(data_[1:4])
    imu_.dvel = np.array(data_[4:7])
    dt = imu_.time - pre_time
    pre_time = imu_.time
    if dt < 0.1:
        imu_.dt = dt
    else:
        imu_.dt = dt_
    return imu_,pre_time

def gnssload(data_):
    gnss_ = ty.GNSS()
    gnss_.time = data_[0]
    gnss_.blh = np.array(data_[1:4])
    gnss_.std = np.array(data_[4:7])
    gnss_.blh[0] *= Angle.D2R
    gnss_.blh[1] *= Angle.D2R
    return(gnss_)

def align(imu_data,gnss_data,starttime):
    imu_cur = ty.IMU()
    gnss = ty.GNSS()
    imu_index = 0
    gnss_index = 0
    pre_time = starttime
    p_t = 0
    for index,row in enumerate(imu_data):
        imu_cur,p_t = imuload(row,imudatarate,pre_time)
        imu_index = index
        if row[0] > starttime:
            break  
    for index,row in enumerate(gnss_data):
        gnss = gnssload(row)
        gnss_index = index
        if row[0] > starttime:
            break
    return imu_cur,gnss,imu_index,gnss_index,p_t

nav_result = np.empty((0, 11))
error_result = np.empty((0, 13))
options = LoadOptions()

giengine = gi.GIEngine()
giengine.GIFunction(options)

imudatarate = config['imudatarate']
starttime = config['starttime']
endtime = config['endtime']
pre_time = starttime
imu_data = np.genfromtxt(config['imupath'],delimiter=None)
gnss_data = np.genfromtxt(config['gnsspath'],delimiter=None)
if endtime < 0 :
    endtime = gnss_data[-1, 0]

imu_cur,gnss,is_index,gs_index,pre_time = align(imu_data,gnss_data,starttime)

giengine.addImuData(imu_cur, True)
giengine.addGnssData(gnss)
for row in imu_data[is_index+1:]:

    if gnss.time < imu_cur.time and gnss.time+1!= endtime:
        gnss = gnssload(gnss_data[gs_index])
        gs_index += 1
        giengine.addGnssData(gnss)
    
    imu_cur,pre_time = imuload(row,imudatarate,pre_time)
    if imu_cur.time > endtime:
        break
    giengine.addImuData(imu_cur)

    giengine.newImuProcess()

    timestamp = giengine.timestamp()
    navstate  = giengine.getNavState()
    imuerr = navstate.imuerror
    # cov       = giengine.getCovariance()

    # print(np.round(navstate.pos[0]* Angle.R2D,10),
    #       np.round(navstate.pos[1]* Angle.R2D,10))

    # raise 'system pause'

    result1 = np.array([
        np.round(0,9),
        np.round(timestamp,9),
        np.round(navstate.pos[0]* Angle.R2D,9),
        np.round(navstate.pos[1]* Angle.R2D,9),
        np.round(navstate.pos[2],9),
        np.round(navstate.vel[0],9),
        np.round(navstate.vel[1],9),
        np.round(navstate.vel[2],9),
        np.round(navstate.euler[0]* Angle.R2D,9),
        np.round(navstate.euler[1]* Angle.R2D,9),
        np.round(navstate.euler[2]* Angle.R2D,9)])
    # result2 = np.array([np.round(timestamp,9),np.round(imuerr.gyrbias[0]* Angle.R2D*3600,9),np.round(imuerr.gyrbias[1]* Angle.R2D*3600,9),  np.round(imuerr.gyrbias[2]* Angle.R2D*3600,9),np.round(imuerr.accbias[0]* 1e5,9), np.round(imuerr.accbias[1]* 1e5,9), np.round(imuerr.accbias[2]* 1e5,9),np.round(imuerr.gyrscale[0] * 1e6,9),np.round(imuerr.gyrscale[1] * 1e6,9),np.round(imuerr.gyrscale[2] * 1e6,9),np.round(imuerr.accscale[0] * 1e6,9),np.round(imuerr.accscale[1] * 1e6,9),np.round(imuerr.accscale[2] * 1e6,9)])

    nav_result = np.vstack((nav_result, result1))
    # error_result = np.vstack((error_result, result2))

    progress = (timestamp - starttime) / (endtime - starttime) * 100
    # print('\r[{:.2f}%]'.format(progress) + str(timestamp))
    sys.stdout.write('\r[{:.2f}%]'.format(progress) + str(timestamp))
    sys.stdout.flush()
    
np.savetxt(config['outputpath']+'/KF_GINS_Navresult-py.nav', nav_result, delimiter=" ",fmt="%.9f")    
# np.savetxt(config['outputpath_error'], error_result, delimiter=" ",fmt="%6f")    





