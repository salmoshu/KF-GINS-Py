import os
import sys
import argparse
import yaml
import time
import numpy as np
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)

from kfgins.kf_gins_types import GINSOptions
from common.angle import Angle
from common.types import IMU, GNSS
from fileio.gnssfileloader import GnssFileLoader
from fileio.imufileloader import ImuFileLoader
from kfgins.gi_engine import GIEngine

# 从配置文件中读取GIEngine相关的初始状态，并转换为标准单位
# Load initial states of GIEngine from configuration file and convert them to standard units
def loadConfig(config, options:GINSOptions):
    # 读取初始位置(纬度 经度 高程)、(北向速度 东向速度 垂向速度)、姿态(欧拉角，ZYX旋转顺序, 横滚角、俯仰角、航向角)
    # load initial position(latitude longitude altitude)
    #              velocity(speeds in the directions of north, east and down)
    #              attitude(euler angle, ZYX, roll, pitch and yaw)
    vec1 = (np.array(config["initpos"])).astype(np.double) 
    vec2 = (np.array(config["initvel"])).astype(np.double)
    vec3 = (np.array(config["initatt"])).astype(np.double)

    options.initstate.pos   = np.array([vec1[0], vec1[1], vec1[2]]) * Angle.D2R
    options.initstate.vel   = np.array([vec2[0], vec2[1], vec2[2]])
    options.initstate.euler = np.array([vec3[0], vec3[1], vec3[2]]) * Angle.D2R
    options.initstate.pos[2] *= Angle.R2D

    # 读取IMU误差初始值(零偏和比例因子)
    # load initial imu error (bias and scale factor)
    vec1 = (np.array(config["initgyrbias"])).astype(np.double)
    vec2 = (np.array(config["initaccbias"])).astype(np.double)
    vec3 = (np.array(config["initgyrscale"])).astype(np.double)
    vec4 = (np.array(config["initaccscale"])).astype(np.double)

    options.initstate.imuerror.gyrbias = np.array([vec1[0], vec1[1], vec1[2]])  * Angle.D2R / 3600.0
    options.initstate.imuerror.accbias = np.array([vec2[0], vec2[1], vec2[2]]) * 1e-5
    options.initstate.imuerror.gyrscale = np.array([vec3[0], vec3[1], vec3[2]]) * 1e-6
    options.initstate.imuerror.accscale = np.array([vec4[0], vec4[1], vec4[2]]) * 1e-6

    # 读取初始位置、速度、姿态(欧拉角)的标准差
    # load initial position std, velocity std and attitude(euler angle) std
    vec1 = (np.array(config["initposstd"])).astype(np.double)
    vec2 = (np.array(config["initvelstd"])).astype(np.double)
    vec3 = (np.array(config["initattstd"])).astype(np.double)

    options.initstate_std.pos = np.array([vec1[0], vec1[1], vec1[2]])
    options.initstate_std.vel = np.array([vec2[0], vec2[1], vec2[2]])
    options.initstate_std.euler = np.array([vec3[0], vec3[1], vec3[2]]) * Angle.D2R
    
    # 读取IMU噪声参数
    # load imu noise parameters
    vec1 = (np.array(config["imunoise"]["arw"])).astype(np.double)
    vec2 = (np.array(config["imunoise"]["vrw"])).astype(np.double)
    vec3 = (np.array(config["imunoise"]["gbstd"])).astype(np.double)
    vec4 = (np.array(config["imunoise"]["abstd"])).astype(np.double)
    vec5 = (np.array(config["imunoise"]["gsstd"])).astype(np.double)
    vec6 = (np.array(config["imunoise"]["asstd"])).astype(np.double)

    options.imunoise.corr_time = (np.array(config["imunoise"]["corrtime"])).astype(np.double)
    options.imunoise.gyr_arw = np.array([vec1[0], vec1[1], vec1[2]])
    options.imunoise.acc_vrw = np.array([vec2[0], vec2[1], vec2[2]])
    options.imunoise.gyrbias_std = np.array([vec3[0], vec3[1], vec3[2]])
    options.imunoise.accbias_std = np.array([vec4[0], vec4[1], vec4[2]])
    options.imunoise.gyrscale_std = np.array([vec5[0], vec5[1], vec5[2]])
    options.imunoise.accscale_std = np.array([vec6[0], vec6[1], vec6[2]])
    
    # 读取IMU误差初始标准差,如果配置文件中没有设置，则采用IMU噪声参数中的零偏和比例因子的标准差
    # Load initial imu bias and scale std, set to bias and scale instability std if load failed
    try:
        vec1 = config['initbgstd']
    except:
        vec1 = [options.imunoise.gyrbias_std[0], options.imunoise.gyrbias_std[1], options.imunoise.gyrbias_std[2]]
    
    try:
        vec2 = config['initbastd']
    except:
        vec2 = [options.imunoise.accbias_std[0], options.imunoise.accbias_std[1], options.imunoise.accbias_std[2]]
    
    try:
        vec3 = config['initsgstd']
    except:
        vec3 = [options.imunoise.gyrscale_std[0], options.imunoise.gyrscale_std[1], options.imunoise.gyrscale_std[2]]
    
    try:
        vec4 = config['initsastd']
    except:
        vec4 = [options.imunoise.accscale_std[0], options.imunoise.accscale_std[1], options.imunoise.accscale_std[2]]
    
    # IMU初始误差转换为标准单位
    # convert initial IMU errors' units to standard units
    options.initstate_std.imuerror.gyrbias = np.array([vec1[0], vec1[1], vec1[2]]) * Angle.D2R / 3600.0
    options.initstate_std.imuerror.accbias = np.array([vec2[0], vec2[1], vec2[2]]) * 1e-5
    options.initstate_std.imuerror.gyrscale = np.array([vec3[0], vec3[1], vec3[2]]) * 1e-6
    options.initstate_std.imuerror.accscale = np.array([vec4[0], vec4[1], vec4[2]]) * 1e-6

    # IMU噪声参数转换为标准单位
    # convert imu noise parameters' units to standard units
    options.imunoise.gyr_arw *= (Angle.D2R / 60.0)
    options.imunoise.acc_vrw /= 60.0
    options.imunoise.gyrbias_std *= (Angle.D2R / 3600.0)
    options.imunoise.accbias_std *= 1e-5
    options.imunoise.gyrscale_std *= 1e-6
    options.imunoise.accscale_std *= 1e-6
    options.imunoise.corr_time *= 3600

    # GNSS天线杆臂, GNSS天线相位中心在IMU坐标系下位置
    # gnss antenna leverarm, position of GNSS antenna phase center in IMU frame
    if("antlever" in config):
        vec1 = (np.array(config["antlever"])).astype(np.double)
        options.antlever = vec1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KF-GINS')
    parser.add_argument('--conf', type=str, help='configuration file path')
    args = parser.parse_args()
    
    print("\033[1m" + "KF-GINS: An EKF-Based GNSS/INS Integrated Navigation System\n" + "\033[0m")

    try:
        filename = None
        if args.conf is None:
            filename = '../dataset/kf-gins.yaml'
        else:
            filename = args.conf
        with open(filename, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception("Failed to read configuration file. Please check the path and format of the configuration file!")

    options = GINSOptions()
    loadConfig(config, options)

    imupath    = config['imupath']
    gnsspath   = config['gnsspath']
    outputpath = config['outputpath']
    imudatalen  = int(config["imudatalen"])
    imudatarate = int(config["imudatarate"])
    starttime   = float(config["starttime"])
    endtime     = float(config["endtime"])

    # 加载GNSS文件和IMU文件
    # load GNSS file and IMU file
    gnssfile = GnssFileLoader(gnsspath)
    imufile = ImuFileLoader(imupath, imudatalen, imudatarate)

    # 构造GIEngine
    # Construct GIEngine
    giengine = GIEngine(options)

    if endtime < 0 :
        endtime = imufile.endtime()

    if (endtime > 604800 or starttime < imufile.starttime() or starttime > endtime):
        print("Process time ERROR!")
    
    # 数据对齐
    # data alignment
    imu_cur = IMU()
    while True:
        imu_cur = imufile.next()
        if imu_cur.time >= starttime:
            break

    gnss = GNSS()
    while True:
        gnss = gnssfile.next()
        if gnss.time >= starttime:
            break
    
    # 添加IMU和GNSS数据到GIEngine中，补偿IMU误差
    # add imudata and gnssdata to GIEngine, compensate IMU error
    giengine.addImuData(imu_cur, True)
    giengine.addGnssData(gnss)

    nav_result = np.empty((0, 11))
    error_result = np.empty((0, 13))

    process_time = time.time()

    f_nav = open(config['outputpath']+'/KF_GINS_Navresult.nav', 'w') 
    f_err = open(config['outputpath']+'/KF_GINS_IMU_ERR.txt', 'w')

    while True:
        # 当前IMU状态时间新于GNSS时间时，读取并添加新的GNSS数据到GIEngine
        # load new gnssdata when current state time is newer than GNSS time and add it to GIEngine
        if gnss.time < imu_cur.time and not gnssfile.isEof():
            gnss = gnssfile.next()
            giengine.addGnssData(gnss)

        # 读取并添加新的IMU数据到GIEngine
        # load new imudata and add it to GIEngine
        imu_cur = imufile.next()
        if imu_cur.time > endtime or imufile.isEof():
            break
        giengine.addImuData(imu_cur)

        # 处理新的IMU数据
        # process new imudata
        giengine.newImuProcess()

        timestamp = giengine.timestamp()
        navstate  = giengine.getNavState()
        imuerr = navstate.imuerror
        # cov       = giengine.getCovariance()

        result1 = np.array([
            np.round(0, 9),
            np.round(timestamp, 9),
            np.round(navstate.pos[0]*Angle.R2D, 9),
            np.round(navstate.pos[1]*Angle.R2D, 9),
            np.round(navstate.pos[2], 9),
            np.round(navstate.vel[0], 9),
            np.round(navstate.vel[1], 9),
            np.round(navstate.vel[2], 9),
            np.round(navstate.euler[0]*Angle.R2D, 9),
            np.round(navstate.euler[1]*Angle.R2D, 9),
            np.round(navstate.euler[2]*Angle.R2D, 9)])

        result2 = np.array([
            np.round(timestamp, 9),
            np.round(imuerr.gyrbias[0]*Angle.R2D*3600, 9),
            np.round(imuerr.gyrbias[1]*Angle.R2D*3600, 9),
            np.round(imuerr.gyrbias[2]*Angle.R2D*3600, 9),
            np.round(imuerr.accbias[0]*1e5, 9),
            np.round(imuerr.accbias[1]*1e5, 9),
            np.round(imuerr.accbias[2]*1e5, 9),
            np.round(imuerr.gyrscale[0]*1e6, 9),
            np.round(imuerr.gyrscale[1]*1e6, 9),
            np.round(imuerr.gyrscale[2]*1e6, 9),
            np.round(imuerr.accscale[0]*1e6, 9),
            np.round(imuerr.accscale[1]*1e6, 9),
            np.round(imuerr.accscale[2]*1e6, 9)])

        np.savetxt(f_nav, [result1], delimiter=" ", fmt="%.9f")
        np.savetxt(f_err, [result2], delimiter=" ", fmt="%.9f")

        progress = (timestamp - starttime) / (endtime - starttime) * 100.0
        sys.stdout.write('\r[{:.2f}%]'.format(progress) + str(timestamp))
        sys.stdout.flush()
    
    print("\nFinished in {:.2f} seconds".format(time.time() - process_time))
    f_nav.close()
    f_err.close()
