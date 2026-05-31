"""
误差状态卡尔曼滤波 (ES-EKF) —— 融合 GNSS + IMU + 轮速 + 地图匹配
"""
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray


@dataclass
class VehicleState:
    """车辆全状态估计"""
    timestamp: float
    # 位置 (ENU)
    x: float; y: float; z: float
    # 速度 (体坐标系)
    vx: float; vy: float; vz: float
    # 姿态 (欧拉角)
    roll: float; pitch: float; yaw: float
    # IMU 零偏
    accel_bias: NDArray   # (3,)
    gyro_bias: NDArray    # (3,)
    # 协方差
    covariance: NDArray   # (15, 15) 或 (21, 21)


class EKFLocalizer:
    """
    误差状态扩展卡尔曼滤波 (Error-State EKF) 多传感器融合定位器

    状态向量 (15维):
      [δp (3), δv (3), δθ (3), δba (3), δbg (3)]

    观测源:
      - GNSS 位置/速度 (ENU)
      - 轮速计 (纵向速度)
      - 地图匹配位置
      - 零速约束
    """

    STATE_DIM = 15  # 误差状态维度

    # 索引常量
    IDX_P = slice(0, 3)    # 位置误差
    IDX_V = slice(3, 6)    # 速度误差
    IDX_THETA = slice(6, 9)  # 姿态误差
    IDX_BA = slice(9, 12)  # 加速度计零偏误差
    IDX_BG = slice(12, 15) # 陀螺仪零偏误差

    def __init__(self):
        # 名义状态 (全状态, 不直接参与滤波)
        self._nominal_state = np.zeros(21)  # [p(3), v(3), q(4), ba(3), bg(3), g(3)]

        # 初始姿态
        q0 = Rotation.identity().as_quat()  # [x, y, z, w]
        self._nominal_state[6:10] = q0

        # 重力向量 (ENU: [0, 0, -g])
        self._nominal_state[18:21] = np.array([0, 0, -9.80665])

        # 误差状态协方差
        self._P = np.eye(self.STATE_DIM) * 0.1

        # 过程噪声协方差
        self._Q = np.eye(12)  # 噪声驱动矩阵为 12 维
        self._Q[:3, :3] *= 0.01   # 加速度计噪声
        self._Q[3:6, :3] *= 0.001  # 陀螺仪噪声
        self._Q[6:9, :3] *= 1e-6   # 加速度计零偏随机游走
        self._Q[9:12, :3] *= 1e-8  # 陀螺仪零偏随机游走

    def predict(self, accel: NDArray, gyro: NDArray, dt: float):
        """
        IMU 预测步骤 (ES-EKF)
        accel: 体坐标系加速度 [ax, ay, az]
        gyro:  体坐标系角速度  [gx, gy, gz]
        """
        # ---- 名义状态更新 ----
        p = self._nominal_state[:3]
        v = self._nominal_state[3:6]
        q = Rotation.from_quat(self._nominal_state[6:10])
        ba = self._nominal_state[10:13]
        bg = self._nominal_state[13:16]
        g = self._nominal_state[18:21]

        # 补偿零偏
        accel_corrected = accel - ba
        gyro_corrected = gyro - bg

        R = q.as_matrix()  # body → ENU

        # 速度更新
        dv = R @ accel_corrected + g
        v_new = v + dv * dt

        # 位置更新
        p_new = p + v * dt + 0.5 * dv * dt ** 2

        # 姿态更新
        angle = np.linalg.norm(gyro_corrected) * dt
        if angle > 1e-10:
            axis = gyro_corrected / np.linalg.norm(gyro_corrected)
            dq = Rotation.from_rotvec(axis * angle)
        else:
            dq = Rotation.identity()
        q_new = q * dq

        self._nominal_state[:3] = p_new
        self._nominal_state[3:6] = v_new
        self._nominal_state[6:10] = q_new.as_quat()

        # ---- 误差状态协方差传播 ----
        # Fx: 误差状态转移矩阵 (15x15)
        Fx = np.eye(self.STATE_DIM)

        # 位置对速度的偏导
        Fx[self.IDX_P, self.IDX_V] = np.eye(3) * dt

        # 速度对姿态的偏导
        accel_skew = self._skew_symmetric(R @ accel_corrected)
        Fx[self.IDX_V, self.IDX_THETA] = -accel_skew * dt

        # 速度对加速度计零偏的偏导
        Fx[self.IDX_V, self.IDX_BA] = -R * dt

        # 姿态对陀螺仪零偏的偏导
        Fx[self.IDX_THETA, self.IDX_BG] = -R * dt

        # 姿态转移
        gyro_skew = self._skew_symmetric(gyro_corrected)
        Fx[self.IDX_THETA, self.IDX_THETA] = np.eye(3) - gyro_skew * dt

        # 噪声驱动矩阵 Fi (15x12)
        Fi = np.zeros((self.STATE_DIM, 12))
        Fi[self.IDX_V, :3] = R
        Fi[self.IDX_THETA, 3:6] = R
        Fi[self.IDX_BA, 6:9] = np.eye(3)
        Fi[self.IDX_BG, 9:12] = np.eye(3)

        # 协方差传播
        self._P = Fx @ self._P @ Fx.T + Fi @ self._Q @ Fi.T

    def update_gnss_position(self, gnss_pos: NDArray, gnss_cov: NDArray):
        """
        GNSS 位置观测更新
        gnss_pos: (3,) ENU 位置
        gnss_cov: (3, 3) 位置协方差
        """
        # 观测矩阵 H: 只观测位置
        H = np.zeros((3, self.STATE_DIM))
        H[:, self.IDX_P] = np.eye(3)

        # 观测噪声
        R = gnss_cov

        # 残差
        z = gnss_pos - self._nominal_state[:3]

        self._kalman_update(H, R, z)

    def update_gnss_velocity(self, gnss_vel: NDArray, vel_cov: NDArray):
        """
        GNSS 速度观测更新 (ENU 坐标系)
        """
        H = np.zeros((3, self.STATE_DIM))
        H[:, self.IDX_V] = np.eye(3)

        z = gnss_vel - self._nominal_state[3:6]

        self._kalman_update(H, vel_cov, z)

    def update_wheel_speed(self, wheel_speed: float, speed_std: float):
        """
        轮速计观测 (纵向速度约束)
        假设车辆无侧滑: body 坐标系下 vy ≈ 0, vz ≈ 0
        """
        # 体坐标系速度观测: [vx, 0, 0]
        q = Rotation.from_quat(self._nominal_state[6:10])
        R = q.as_matrix()
        v_body_expected = R.T @ self._nominal_state[3:6]  # ENU → body

        # 只观测纵向和横向 (vx = wheel_speed, vy = 0)
        H = np.zeros((2, self.STATE_DIM))
        H[0, self.IDX_V] = R[0, :]  # vx in ENU wrt body x
        H[1, self.IDX_V] = R[1, :]  # vy in ENU wrt body y

        z = np.array([wheel_speed - v_body_expected[0],
                      -v_body_expected[1]])

        R_mat = np.diag([speed_std ** 2, 0.5 ** 2])  # vy ≈ 0, 允许 0.5m/s

        self._kalman_update(H, R_mat, z)

    def update_map_match(self, map_pos: NDArray, map_cov: NDArray):
        """
        地图匹配位置观测 (NDT/ICP 结果)
        """
        H = np.zeros((3, self.STATE_DIM))
        H[:, self.IDX_P] = np.eye(3)

        z = map_pos - self._nominal_state[:3]

        self._kalman_update(H, map_cov, z)

    def _kalman_update(self, H: NDArray, R: NDArray, z: NDArray):
        """
        标准卡尔曼更新
        """
        # 卡尔曼增益
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)

        # 误差状态更新
        delta_x = K @ z

        # 注入名义状态
        self._inject_error_state(delta_x)

        # 协方差更新 (Joseph 形式, 数值稳定)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

    def _inject_error_state(self, delta_x: NDArray):
        """
        将误差状态注入名义状态, 然后重置误差状态
        """
        # 位置
        self._nominal_state[:3] += delta_x[self.IDX_P]

        # 速度
        self._nominal_state[3:6] += delta_x[self.IDX_V]

        # 姿态 (通过四元数乘法)
        delta_theta = delta_x[self.IDX_THETA]
        angle = np.linalg.norm(delta_theta)
        if angle > 1e-10:
            axis = delta_theta / angle
            dq = Rotation.from_rotvec(axis * angle)
            q = Rotation.from_quat(self._nominal_state[6:10])
            q_new = q * dq
            self._nominal_state[6:10] = q_new.as_quat()

        # 零偏
        self._nominal_state[10:13] += delta_x[self.IDX_BA]
        self._nominal_state[13:16] += delta_x[self.IDX_BG]

        # 重置误差状态协方差 (G 矩阵近似为单位阵)
        # 在实际 ES-EKF 中需要更复杂的重置, 这里简化处理

    def get_state(self) -> VehicleState:
        """获取当前状态估计"""
        q = self._nominal_state[6:10]
        euler = Rotation.from_quat(q).as_euler('ZYX')

        return VehicleState(
            timestamp=0.0,
            x=self._nominal_state[0],
            y=self._nominal_state[1],
            z=self._nominal_state[2],
            vx=self._nominal_state[3],
            vy=self._nominal_state[4],
            vz=self._nominal_state[5],
            roll=euler[2], pitch=euler[1], yaw=euler[0],
            accel_bias=self._nominal_state[10:13].copy(),
            gyro_bias=self._nominal_state[13:16].copy(),
            covariance=self._P.copy(),
        )

    @staticmethod
    def _skew_symmetric(v: NDArray) -> NDArray:
        """向量 → 反对称矩阵"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
