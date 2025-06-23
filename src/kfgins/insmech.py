import numpy as np
from scipy.spatial.transform import Rotation
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)

from common.types import IMU
from kfgins.kf_gins_types import PVA
from common.earth import Earth
from common.rotation import RotationUtils as RU

class INSMech:
    def insMech(pvapre:PVA, imupre:IMU, imucur:IMU):
        pvacur = PVA()
        # 依次进行速度更新、位置更新、姿态更新, 不可调换顺序
        # perform velocity update, position updata and attitude update in sequence, irreversible order
        pvacur = INSMech.velUpdate(pvapre, pvacur, imupre, imucur)
        pvacur = INSMech.posUpdate(pvapre, pvacur, imupre, imucur)
        pvacur = INSMech.attUpdate(pvapre, pvacur, imupre, imucur)
        return pvacur
    
    def velUpdate(pvapre:PVA, pvacur:PVA, imupre:IMU, imucur:IMU):
        d_vfb, d_vfn, d_vgn, gl, midvel, midpos = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
        temp1, temp2, temp3 = np.zeros(3),np.zeros(3),np.zeros(3)
        cnn = np.zeros((3, 3))
        I33 = np.eye(3)
        qne, qee, qnn = np.zeros(4),np.zeros(4),np.zeros(4)

        # 计算地理参数，子午圈半径和卯酉圈半径，地球自转角速度投影到n系, n系相对于e系转动角速度投影到n系，重力值
        # calculate geographic parameters, Meridian and Mao unitary radii,
        # earth rotational angular velocity projected to n-frame,
        # rotational angular velocity of n-frame to e-frame projected to n-frame, and gravity
        rmrn = Earth.meridianPrimeVerticalRadius(pvapre.pos[0])
        wie_n = np.array([Earth.WGS84_WIE * np.cos(pvapre.pos[0]), 0, -Earth.WGS84_WIE * np.sin(pvapre.pos[0])])
        wen_n = np.array([pvapre.vel[1] / (rmrn[1] + pvapre.pos[2]), -pvapre.vel[0] / (rmrn[0] + pvapre.pos[2]), 
                          -pvapre.vel[1] * np.tan(pvapre.pos[0]) / (rmrn[1] + pvapre.pos[2])])
        gravity = Earth.gravity(pvapre.pos)
        
        # 旋转效应和双子样划桨效应
        # rotational and sculling motion
        temp1 = np.cross(imucur.dtheta, imucur.dvel) / 2
        temp2 = np.cross(imupre.dtheta, imucur.dvel) / 12
        temp3 = np.cross(imupre.dvel, imucur.dtheta) / 12

        # b系比力积分项
        # velocity increment due to the specific force
        d_vfb = imucur.dvel + temp1 + temp2 + temp3

        # 比力积分项投影到n系
        # velocity increment dut to the specfic force projected to the n-frame
        temp1 = (wie_n + wen_n) * imucur.dt / 2
        cnn   = I33 - RU.skewSymmetric(temp1)
        d_vfn = cnn @ pvapre.att.cbn @ d_vfb
        
        # 计算重力/哥式积分项
        # velocity increment due to the gravity and Coriolis force
        gl = np.array([0.0,0.0,gravity])
        d_vgn = (gl - np.cross((2 * wie_n + wen_n), pvapre.vel)) * imucur.dt
        
        # 得到中间时刻速度
        # velocity at k-1/2
        midvel = pvapre.vel + (d_vfn + d_vgn) / 2
        
        # 外推得到中间时刻位置
        # position extrapolation to k-1/2
        qnn = RU.rotvec2quaternion(temp1)
        temp2 = np.array([0.0,0.0,-Earth.WGS84_WIE * imucur.dt / 2])
        qee = RU.rotvec2quaternion(temp2)
        qne = Earth.qne(pvapre.pos)
        qne = Rotation.from_quat(qee) * Rotation.from_quat(qne) * Rotation.from_quat(qnn)
        qne = qne.as_quat()
        midpos[2] = pvapre.pos[2] - midvel[2] * imucur.dt / 2
        midpos = Earth.blh(qne, midpos[2])
        
        # 重新计算中间时刻的rmrn, wie_e, wen_n
        # recompute rmrn, wie_n, and wen_n at k-1/2
        rmrn = Earth.meridianPrimeVerticalRadius(midpos[0])
        wie_n =np.array([Earth.WGS84_WIE * np.cos(midpos[0]), 0, -Earth.WGS84_WIE * np.sin(midpos[0])])
        wen_n = np.array([midvel[1] / (rmrn[1] + midpos[2]), -midvel[0] / (rmrn[0] + midpos[2]), -midvel[1] * np.tan(midpos[0]) / (rmrn[1] + midpos[2])])
        
        # 重新计算n系下平均比力积分项
        # recompute d_vfn
        temp3 = (wie_n + wen_n) * imucur.dt / 2
        cnn   = I33 - RU.skewSymmetric(temp3)
        d_vfn = cnn @ pvapre.att.cbn @ d_vfb

        # 重新计算重力、哥式积分项
        # recompute d_vgn
        a = Earth.gravity(midpos)
        gl = np.array([0.0,0.0,a])
        d_vgn = (gl - np.cross((2 * wie_n + wen_n), midvel)) * imucur.dt
         
        # 速度更新完成
        # velocity update finish
        pvacur.vel = pvapre.vel + d_vfn + d_vgn
        
        return pvacur
    
    def posUpdate(pvapre:PVA, pvacur:PVA, imupre:IMU, imucur:IMU):
        temp1, temp2, midvel, midpos = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
        qne, qee, qnn =  np.zeros(4) ,np.zeros(4) ,np.zeros(4)

        # 重新计算中间时刻的速度和位置
        # recompute velocity and position at k-1/2
        midvel = (pvacur.vel + pvapre.vel) / 2
        midpos = pvapre.pos + Earth.DRi(pvapre.pos) @ midvel * imucur.dt / 2

        # 重新计算中间时刻地理参数
        # recompute rmrn, wie_n, wen_n at k-1/2
        rmrn = Earth.meridianPrimeVerticalRadius(midpos[0])
        wie_n = np.array([Earth.WGS84_WIE * np.cos(midpos[0]), 0, -Earth.WGS84_WIE * np.sin(midpos[0])])
        wen_n = np.array([midvel[1] / (rmrn[1] + midpos[2]), -midvel[0] / (rmrn[0] + midpos[2]),-midvel[1] * np.tan(midpos[0]) / (rmrn[1] + midpos[2])])

        # 重新计算 k时刻到k-1时刻 n系旋转矢量
        # recompute n-frame rotation vector (n(k) with respect to n(k-1)-frame)
        temp1 = (wie_n + wen_n) * imucur.dt
        qnn   = RU.rotvec2quaternion(temp1)

        # e系转动等效旋转矢量 (k-1时刻k时刻，所以取负号)
        # e-frame rotation vector (e(k-1) with respect to e(k)-frame)
        temp2 = np.array([ 0.0, 0.0, -Earth.WGS84_WIE * imucur.dt])
        qee = RU.rotvec2quaternion(temp2)

        # 位置更新完成
        # position update finish
        qne = Earth.qne(pvapre.pos)
        qne = Rotation.from_quat(qee) * Rotation.from_quat(qne) * Rotation.from_quat(qnn)
        qne = qne.as_quat()
        pvacur.pos[2] = pvapre.pos[2] - midvel[2] * imucur.dt
        pvacur.pos = Earth.blh(qne, pvacur.pos[2])
        return pvacur
    
    def attUpdate(pvapre:PVA, pvacur:PVA, imupre:IMU, imucur:IMU):
        qne_pre, qne_cur, qne_mid, qnn, qbb = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)
        temp1, midpos, midvel = np.zeros(3),np.zeros(3),np.zeros(3)

        # 重新计算中间时刻的速度和位置
        # recompute velocity and position at k-1/2
        midvel = (pvapre.vel + pvacur.vel) / 2
        qne_pre   = Earth.qne(pvapre.pos)
        qne_cur   = Earth.qne(pvacur.pos)
        q = Rotation.from_quat(qne_cur).inv()* Rotation.from_quat(qne_pre)
        q = q.as_quat()
        temp1     = RU.quaternion2vector(q)
        qne_mid   = Rotation.from_quat(qne_pre) * Rotation.from_quat(RU.rotvec2quaternion(temp1 / 2)).inv()
        midpos[2] = (pvacur.pos[2] + pvapre.pos[2]) / 2
        midpos    = Earth.blh(qne_mid.as_quat(), midpos[2])

        # 重新计算中间时刻地理参数
        # recompute rmrn, wie_n, wen_n at k-1/2
        rmrn = Earth.meridianPrimeVerticalRadius(midpos[0])
        wie_n = np.array([Earth.WGS84_WIE * np.cos(midpos[0]), 0, -Earth.WGS84_WIE * np.sin(midpos[0])])
        wen_n = np.array([midvel[1] / (rmrn[1] + midpos[2]), -midvel[0] / (rmrn[0] + midpos[2]),
        -midvel[1] * np.tan(midpos[0]) / (rmrn[1] + midpos[2])])

        # 计算n系的旋转四元数 k-1时刻到k时刻变换
        # n-frame rotation vector (n(k-1) with respect to n(k)-frame)
        temp1 = -(wie_n + wen_n) * imucur.dt
        qnn   = RU.rotvec2quaternion(temp1)

        # 计算b系旋转四元数 补偿二阶圆锥误差
        # b-frame rotation vector (b(k) with respect to b(k-1)-frame)
        # compensate the second-order coning correction term.
        temp1 = imucur.dtheta + np.cross(imupre.dtheta,imucur.dtheta) / 12
        qbb   = RU.rotvec2quaternion(temp1)

        # 姿态更新完成
        # attitude update finish
        if np.all(pvapre.att.qbn == 0):
            pvacur.att.qbn   = Rotation.from_quat(qnn) * Rotation.from_quat(qbb)
        else:
            pvacur.att.qbn   = Rotation.from_quat(qnn) * Rotation.from_quat(pvapre.att.qbn) * Rotation.from_quat(qbb)
        
        pvacur.att.qbn = pvacur.att.qbn.as_quat()
        pvacur.att.cbn   = RU.quaternion2matrix(pvacur.att.qbn)
        pvacur.att.euler = RU.matrix2euler(pvacur.att.cbn)

        return pvacur
        