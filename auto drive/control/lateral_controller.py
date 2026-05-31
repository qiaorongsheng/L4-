"""
横向控制器 —— Stanley / 纯追踪 (Pure Pursuit) / LQR 横向控制
"""
import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray


class LateralController:
    """
    横向控制器:
      - Stanley 控制器 (高速 + 低速自适应)
      - 纯追踪 (Pure Pursuit) 作为 fallback
      - 前馈 + 反馈结构

    Stanley 控制律:
      δ = θ_e + arctan(k * e_lat / v)

    其中:
      θ_e = 航向误差
      e_lat = 横向误差 (前轴处)
      k = 增益系数
      v = 车速
    """

    def __init__(self, wheelbase: float = 2.85,
                 k_stanley: float = 0.5,
                 k_soft: float = 1.0,
                 max_steering_angle: float = 0.65,
                 max_steering_rate: float = 0.52):
        self.wheelbase = wheelbase
        self.k_stanley = k_stanley
        self.k_soft = k_soft
        self.max_steering_angle = max_steering_angle
        self.max_steering_rate = max_steering_rate

        self._prev_steering = 0.0

    def stanley_control(self, ego_state,
                        reference_trajectory: NDArray,
                        current_idx: int) -> Tuple[float, float]:
        """
        Stanley 横向控制

        Args:
            ego_state: 车辆状态 (含 x, y, yaw, v)
            reference_trajectory: (N, 4+) [x, y, heading, velocity, ...]
            current_idx: 当前参考轨迹索引

        Returns:
            (steering_angle [rad], lateral_error [m])
        """
        # 当前参考点
        ref = reference_trajectory[current_idx]
        ref_x, ref_y = ref[0], ref[1]
        ref_heading = ref[2]

        # ---- 航向误差 ----
        yaw = getattr(ego_state, 'yaw', ego_state[2])
        heading_error = ref_heading - yaw
        # 归一化到 [-π, π]
        heading_error = np.arctan2(np.sin(heading_error),
                                   np.cos(heading_error))

        # ---- 横向误差 (前轴处) ----
        ego_x = getattr(ego_state, 'x', ego_state[0])
        ego_y = getattr(ego_state, 'y', ego_state[1])

        # 横向误差向量
        dx = ego_x - ref_x
        dy = ego_y - ref_y

        # 横向误差 = 点积 (位置误差向量, 参考路径的法向量)
        normal = np.array([-np.sin(ref_heading), np.cos(ref_heading)])
        position_error = np.array([dx, dy])
        lateral_error = np.dot(position_error, normal)

        # 前轴横向误差 (考虑轴距)
        front_axle_lateral_error = lateral_error + \
            self.wheelbase / 2 * np.sin(heading_error)

        # ---- Stanley 控制律 ----
        v = getattr(ego_state, 'vx', ego_state[3]) if \
            hasattr(ego_state, 'vx') else ego_state[3]
        if hasattr(ego_state, 'vy'):
            v = np.sqrt(ego_state.vx ** 2 + ego_state.vy ** 2)
        v = max(v, 0.5)  # 防止低速除零

        # 自适应增益
        k = self.k_stanley

        # 横向误差项
        lateral_term = np.arctan(k * front_axle_lateral_error /
                                 (self.k_soft + v))

        # 总转向角 = 航向误差 + 横向修正
        steering = heading_error + lateral_term

        # 限幅
        steering = np.clip(steering, -self.max_steering_angle,
                           self.max_steering_angle)

        # 转向速率限制
        max_delta_per_step = self.max_steering_rate * 0.01  # 假设 100Hz
        steering = np.clip(steering,
                           self._prev_steering - max_delta_per_step,
                           self._prev_steering + max_delta_per_step)

        self._prev_steering = steering

        return float(steering), float(front_axle_lateral_error)

    def pure_pursuit_control(self, ego_state,
                             reference_path: NDArray,
                             lookahead_distance: float = None
                             ) -> Tuple[float, float]:
        """
        纯追踪 (Pure Pursuit) 横向控制 (fallback)

        δ = arctan(2 * L * sin(α) / ld)

        其中:
          L = 轴距
          α = 目标点与车辆航向的夹角
          ld = 预瞄距离
        """
        ego_x = getattr(ego_state, 'x', ego_state[0])
        ego_y = getattr(ego_state, 'y', ego_state[1])
        ego_yaw = getattr(ego_state, 'yaw', ego_state[2])

        if lookahead_distance is None:
            v = getattr(ego_state, 'vx', 5.0) if \
                hasattr(ego_state, 'vx') else 5.0
            lookahead_distance = np.clip(v * 0.8, 8.0, 30.0)

        # 找预瞄点 (沿参考路径距离 ego 最近的 forward 点)
        target_idx = self._find_lookahead_point(
            ego_x, ego_y, reference_path, lookahead_distance
        )

        if target_idx is None or target_idx >= len(reference_path):
            return 0.0, 0.0

        target = reference_path[target_idx]
        target_x, target_y = target[0], target[1]

        # 目标点在车辆坐标系中的位置
        dx = target_x - ego_x
        dy = target_y - ego_y

        # 转换到车辆坐标系
        x_veh = dx * np.cos(ego_yaw) + dy * np.sin(ego_yaw)
        y_veh = -dx * np.sin(ego_yaw) + dy * np.cos(ego_yaw)

        # 纯追踪曲率
        if abs(x_veh) < 0.01:
            return 0.0, y_veh

        curvature = 2 * y_veh / (lookahead_distance ** 2)
        steering = np.arctan(self.wheelbase * curvature)

        steering = np.clip(steering, -self.max_steering_angle,
                           self.max_steering_angle)

        return float(steering), float(y_veh)

    def lqr_lateral_control(self, ego_state,
                            reference_trajectory: NDArray,
                            current_idx: int) -> Tuple[float, float]:
        """
        LQR 横向控制 (最优控制)
        状态: [e_lat, e_lat_dot, e_heading, e_heading_dot]
        """
        # 提取误差状态
        ref = reference_trajectory[current_idx]
        ref_heading = ref[2]
        ref_curvature = ref[4] if reference_trajectory.shape[1] > 4 else 0.0

        ego_yaw = getattr(ego_state, 'yaw', ego_state[2])
        ego_v = getattr(ego_state, 'vx', ego_state[3]) if \
            hasattr(ego_state, 'vx') else 5.0
        ego_v = max(ego_v, 1.0)

        # 横向误差
        ego_x = getattr(ego_state, 'x', ego_state[0])
        ego_y = getattr(ego_state, 'y', ego_state[1])
        dx = ego_x - ref[0]
        dy = ego_y - ref[1]
        normal = np.array([-np.sin(ref_heading), np.cos(ref_heading)])
        e_lat = dx * normal[0] + dy * normal[1]

        # 横向误差变化率
        e_lat_dot = ego_v * np.sin(ego_yaw - ref_heading)

        # 航向误差
        e_heading = ego_yaw - ref_heading
        e_heading = np.arctan2(np.sin(e_heading), np.cos(e_heading))

        # 航向误差变化率 (简化 = 横摆角速度 - 参考曲率*速度)
        ego_yaw_rate = getattr(ego_state, 'yaw_rate', 0.0)
        e_heading_dot = ego_yaw_rate - ref_curvature * ego_v

        # LQR 状态
        x_err = np.array([e_lat, e_lat_dot, e_heading, e_heading_dot])

        # LQR 增益矩阵 (离线预计算, 这里为示例值)
        # 实际应根据车辆模型 + 代价矩阵求解 Riccati 方程
        K = np.array([-0.15, -0.08, -0.45, -0.12])

        # 前馈项 (补偿道路曲率)
        L = self.wheelbase
        m = 2100.0
        Cf = Cr = 155494.0
        Kv = L * (1 + m * ego_v ** 2 * (L / (2 * Cr) - L / (2 * Cf)) / L)

        ff_steering = L * ref_curvature + Kv * ref_curvature

        # 反馈 + 前馈
        steering = -K @ x_err + ff_steering * self.wheelbase

        steering = np.clip(steering, -self.max_steering_angle,
                           self.max_steering_angle)

        return float(steering), float(e_lat)

    def compute_steering(self, ego_state,
                         reference_trajectory: NDArray,
                         current_idx: int,
                         method: str = 'stanley') -> Tuple[float, float]:
        """
        横向控制统一入口
        """
        v = getattr(ego_state, 'vx', ego_state[3]) if \
            hasattr(ego_state, 'vx') else 5.0

        if method == 'lqr':
            return self.lqr_lateral_control(
                ego_state, reference_trajectory, current_idx
            )
        elif method == 'pure_pursuit':
            return self.pure_pursuit_control(
                ego_state, reference_trajectory
            )
        else:  # stanley (默认)
            return self.stanley_control(
                ego_state, reference_trajectory, current_idx
            )

    def _find_lookahead_point(self, ego_x: float, ego_y: float,
                              path: NDArray,
                              lookahead_distance: float) -> Optional[int]:
        """
        在路径上寻找距离车辆 lookahead_distance 的前方最近点
        """
        best_idx = None
        best_dist_diff = float('inf')

        for i in range(len(path)):
            pt = path[i]
            dist = np.sqrt((pt[0] - ego_x) ** 2 + (pt[1] - ego_y) ** 2)
            diff = abs(dist - lookahead_distance)

            # 确保是前方点
            if i > 0:
                prev_pt = path[i - 1]
                dot = (pt[0] - prev_pt[0]) * (pt[0] - ego_x) + \
                      (pt[1] - prev_pt[1]) * (pt[1] - ego_y)
                if dot < 0:  # 在后方
                    continue

            if diff < best_dist_diff:
                best_dist_diff = diff
                best_idx = i

        return best_idx
