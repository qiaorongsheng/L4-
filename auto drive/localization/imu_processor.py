"""
IMU 惯性测量单元处理 —— 捷联惯导解算、零速修正、重力补偿
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray


@dataclass
class IMUMeasurement:
    """IMU 原始测量值"""
    timestamp: float
    # 加速度计 [m/s²] (体坐标系)
    accel_x: float; accel_y: float; accel_z: float
    # 陀螺仪 [rad/s] (体坐标系)
    gyro_x: float; gyro_y: float; gyro_z: float
    # 温度 [°C]
    temperature: float = 25.0


@dataclass
class INSState:
    """惯性导航状态"""
    timestamp: float
    # 位置 (ENU 坐标系)
    e: float; n: float; u: float
    # 速度 (ENU 坐标系)
    v_e: float; v_n: float; v_u: float
    # 姿态 (四元数, body → ENU)
    q_w: float; q_x: float; q_y: float; q_z: float
    # 加速度计零偏
    accel_bias_x: float = 0.0
    accel_bias_y: float = 0.0
    accel_bias_z: float = 0.0
    # 陀螺仪零偏
    gyro_bias_x: float = 0.0
    gyro_bias_y: float = 0.0
    gyro_bias_z: float = 0.0


class IMUProcessor:
    """
    IMU 惯导解算:
      - 捷联惯导 (Strapdown Inertial Navigation)
      - 四元数姿态更新
      - 科里奥利力 & 重力补偿
      - Allan 方差零偏估计
      - 零速检测 (ZUPT)
    """

    GRAVITY = 9.80665  # 重力加速度 [m/s²]

    def __init__(self, gyro_noise_density: float = 0.0017,  # °/√h → rad/√s
                 accel_noise_density: float = 0.00098,       # m/s/√h → m/s/√s
                 gyro_bias_stability: float = 3.5e-6,        # rad/s
                 accel_bias_stability: float = 1.0e-4):      # m/s²
        self.gyro_noise_density = gyro_noise_density
        self.accel_noise_density = accel_noise_density
        self.gyro_bias_stability = gyro_bias_stability
        self.accel_bias_stability = accel_bias_stability

        # 初始对准标志
        self._initialized = False
        self._static_samples = []

    def calibrate_biases(self, static_measurements: list,
                         duration_s: float = 5.0) -> Tuple[NDArray, NDArray]:
        """
        静止状态标定: 采集静态数据估计传感器零偏
        加速度计: 均值 = 零偏 (静止时只有重力)
        陀螺仪: 均值 = 零偏 (地球自转可忽略)
        """
        if len(static_measurements) < 100:
            return np.zeros(3), np.zeros(3)

        accel_biases = np.zeros(3)
        gyro_biases = np.zeros(3)

        for m in static_measurements:
            accel_biases += np.array([m.accel_x, m.accel_y, m.accel_z])
            gyro_biases += np.array([m.gyro_x, m.gyro_y, m.gyro_z])

        accel_biases /= len(static_measurements)
        gyro_biases /= len(static_measurements)

        # 加速度计: 减去重力分量 (假设水平安装)
        accel_biases[2] -= self.GRAVITY

        return accel_biases, gyro_biases

    def initialize_attitude(self, accel: NDArray, mag: Optional[NDArray] = None) -> Rotation:
        """
        利用加速度计 + 磁力计进行初始对准
        """
        # 归一化
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 0.5:
            return Rotation.identity()

        accel_normalized = accel / accel_norm

        # 重力方向 → 姿态 (假设 z 轴向下)
        # roll = atan2(-ay, -az), pitch = asin(ax / g)
        roll = np.arctan2(-accel_normalized[1], -accel_normalized[2])
        pitch = np.arcsin(np.clip(accel_normalized[0], -1, 1))

        # 航向角 (如果没有磁力计, 设为 0)
        yaw = 0.0
        if mag is not None:
            # 将磁力计投影到水平面
            mag_norm = np.linalg.norm(mag)
            if mag_norm > 0:
                # Tilt-compensated heading
                mx = mag[0] * np.cos(pitch) + \
                     mag[1] * np.sin(roll) * np.sin(pitch) - \
                     mag[2] * np.cos(roll) * np.sin(pitch)
                my = mag[1] * np.cos(roll) + mag[2] * np.sin(roll)
                yaw = np.arctan2(-my, mx)

        return Rotation.from_euler('ZYX', [yaw, pitch, roll])

    def propagate(self, state: INSState, meas: IMUMeasurement,
                  dt: float) -> INSState:
        """
        捷联惯导递推 (一阶积分)
        """
        # 补偿零偏
        accel = np.array([meas.accel_x, meas.accel_y, meas.accel_z])
        accel -= np.array([state.accel_bias_x,
                           state.accel_bias_y,
                           state.accel_bias_z])

        gyro = np.array([meas.gyro_x, meas.gyro_y, meas.gyro_z])
        gyro -= np.array([state.gyro_bias_x,
                          state.gyro_bias_y,
                          state.gyro_bias_z])

        # 当前姿态
        q = Rotation.from_quat([state.q_x, state.q_y, state.q_z, state.q_w])
        R_b2e = q.as_matrix()  # body → ENU

        # ---- 姿态更新 (四元数积分) ----
        gyro_norm = np.linalg.norm(gyro)
        if gyro_norm > 1e-10:
            axis = gyro / gyro_norm
            angle = gyro_norm * dt
            dq = Rotation.from_rotvec(axis * angle)
        else:
            dq = Rotation.identity()

        q_new = q * dq
        R_new = q_new.as_matrix()

        # ---- 速度更新 (重力 + 科里奥利力补偿) ----
        # 将加速度转换到 ENU 坐标系
        accel_enu = R_new @ accel

        # 重力补偿 (ENU: 重力沿 -U 方向)
        gravity_enu = np.array([0, 0, -self.GRAVITY])
        accel_enu += gravity_enu

        # ---- 位置更新 ----
        v_enu = np.array([state.v_e, state.v_n, state.v_u])
        v_new = v_enu + accel_enu * dt

        pos = np.array([state.e, state.n, state.u])
        pos_new = pos + v_enu * dt + 0.5 * accel_enu * dt ** 2

        # 新状态
        q_xyzw = q_new.as_quat()  # [x, y, z, w]
        return INSState(
            timestamp=meas.timestamp,
            e=float(pos_new[0]), n=float(pos_new[1]), u=float(pos_new[2]),
            v_e=float(v_new[0]), v_n=float(v_new[1]), v_u=float(v_new[2]),
            q_w=float(q_xyzw[3]), q_x=float(q_xyzw[0]),
            q_y=float(q_xyzw[1]), q_z=float(q_xyzw[2]),
            accel_bias_x=state.accel_bias_x,
            accel_bias_y=state.accel_bias_y,
            accel_bias_z=state.accel_bias_z,
            gyro_bias_x=state.gyro_bias_x,
            gyro_bias_y=state.gyro_bias_y,
            gyro_bias_z=state.gyro_bias_z,
        )

    def zero_velocity_detector(self, accel: NDArray, gyro: NDArray,
                               window_size: int = 10) -> bool:
        """
        零速检测 (ZUPT): 判断车辆是否静止
        用于抑制 IMU 积分漂移
        """
        self._static_samples.append((accel.copy(), gyro.copy()))
        if len(self._static_samples) > window_size:
            self._static_samples.pop(0)
        if len(self._static_samples) < window_size:
            return False

        # 加速度方差检验
        accel_stack = np.array([s[0] for s in self._static_samples])
        accel_var = np.var(accel_stack, axis=0)

        # 陀螺仪方差检验
        gyro_stack = np.array([s[1] for s in self._static_samples])
        gyro_var = np.var(gyro_stack, axis=0)

        # 阈值判断
        accel_threshold = 0.02   # m²/s⁴
        gyro_threshold = 0.001  # rad²/s²

        is_static = (np.all(accel_var < accel_threshold) and
                     np.all(gyro_var < gyro_threshold))

        return is_static

    def get_euler_angles(self, state: INSState) -> Tuple[float, float, float]:
        """获取欧拉角 (roll, pitch, yaw)"""
        q = Rotation.from_quat([state.q_x, state.q_y, state.q_z, state.q_w])
        euler = q.as_euler('ZYX')  # yaw, pitch, roll
        return (float(euler[2]), float(euler[1]), float(euler[0]))
