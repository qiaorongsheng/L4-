"""
紧急处理模块 —— 最小风险策略 (MRM) 执行
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray


class MRMType(Enum):
    """最小风险策略类型"""
    SAFE_STOP = "safe_stop"          # 在车道内安全停车
    PULL_OVER = "pull_over"          # 靠边停车
    SLOW_DOWN = "slow_down"          # 减速慢行
    HANDOVER = "handover"            # 请求接管 (L4→L3退坡)
    RETURN_TO_STATION = "return"     # 返回站点


@dataclass
class EmergencyTrajectory:
    """紧急轨迹"""
    trajectory: NDArray         # (N, 4) [x, y, heading, v]
    mrm_type: MRMType
    stop_point: Tuple[float, float]
    estimated_stop_time: float
    safety_margin: float


class EmergencyHandler:
    """
    紧急情况处理:
      - 最小风险策略 (MRM) 规划
      - 安全停车轨迹生成
      - 靠边停车轨迹
      - 紧急制动触发
      - 故障降级管理
    """

    def __init__(self, max_deceleration: float = -6.0,
                 comfort_deceleration: float = -3.0,
                 emergency_deceleration: float = -9.0):
        self.max_deceleration = max_deceleration
        self.comfort_deceleration = comfort_deceleration
        self.emergency_deceleration = emergency_deceleration

        self._mrm_active = False
        self._mrm_type: Optional[MRMType] = None

    def trigger_emergency_brake(self, ego_state,
                                dt: float = 0.1,
                                horizon: float = 3.0) -> EmergencyTrajectory:
        """
        触发紧急制动: 在当前位置沿当前方向以最大减速度停车
        """
        num_steps = int(horizon / dt)
        v0 = getattr(ego_state, 'vx', 5.0)
        x0 = getattr(ego_state, 'x', 0.0)
        y0 = getattr(ego_state, 'y', 0.0)
        heading = getattr(ego_state, 'yaw', 0.0)

        trajectory = np.zeros((num_steps, 4))

        for i in range(num_steps):
            t = (i + 1) * dt
            v = max(0, v0 + self.emergency_deceleration * t)

            # 停止距离
            if v > 0:
                t_stop = v0 / abs(self.emergency_deceleration)
                s = v0 * min(t, t_stop) + 0.5 * self.emergency_deceleration * \
                    min(t, t_stop) ** 2
            else:
                s = v0 ** 2 / (2 * abs(self.emergency_deceleration))

            trajectory[i, 0] = x0 + s * np.cos(heading)
            trajectory[i, 1] = y0 + s * np.sin(heading)
            trajectory[i, 2] = heading
            trajectory[i, 3] = v

        stop_dist = v0 ** 2 / (2 * abs(self.emergency_deceleration))
        stop_x = x0 + stop_dist * np.cos(heading)
        stop_y = y0 + stop_dist * np.sin(heading)

        self._mrm_active = True
        self._mrm_type = MRMType.SAFE_STOP

        return EmergencyTrajectory(
            trajectory=trajectory,
            mrm_type=MRMType.SAFE_STOP,
            stop_point=(stop_x, stop_y),
            estimated_stop_time=v0 / abs(self.emergency_deceleration),
            safety_margin=0.0,
        )

    def plan_pull_over(self, ego_state, lane_boundaries,
                       dt: float = 0.1,
                       horizon: float = 10.0) -> EmergencyTrajectory:
        """
        规划靠边停车轨迹
        策略: 逐步减速 + 横向移动到路肩
        """
        num_steps = int(horizon / dt)
        v0 = getattr(ego_state, 'vx', 5.0)
        x0 = getattr(ego_state, 'x', 0.0)
        y0 = getattr(ego_state, 'y', 0.0)
        heading0 = getattr(ego_state, 'yaw', 0.0)

        trajectory = np.zeros((num_steps, 4))

        # 目标横向位置: 右侧路肩 (右侧车道线外 1m)
        right_lane_boundary = -1.75  # 右车道边界 (简化为 -3.5/2)
        target_y = right_lane_boundary - 1.5  # 路肩上

        for i in range(num_steps):
            t = (i + 1) * dt

            # 减速到舒适停车
            v = max(0, v0 + self.comfort_deceleration * t)

            # 横向平滑过渡到路肩
            alpha = min(1.0, t / 4.0)  # 4秒横向过渡
            traj_y = y0 + (target_y - y0) * alpha

            # 纵向
            avg_v = (v0 + v) / 2
            traj_x = x0 + avg_v * t * np.cos(heading0)

            trajectory[i, 0] = traj_x
            trajectory[i, 1] = traj_y
            trajectory[i, 2] = heading0
            trajectory[i, 3] = v

        stop_time = v0 / abs(self.comfort_deceleration)
        stop_dist = v0 * stop_time + 0.5 * self.comfort_deceleration * stop_time ** 2
        stop_x = x0 + stop_dist * np.cos(heading0)
        stop_y = target_y

        self._mrm_active = True
        self._mrm_type = MRMType.PULL_OVER

        return EmergencyTrajectory(
            trajectory=trajectory,
            mrm_type=MRMType.PULL_OVER,
            stop_point=(stop_x, stop_y),
            estimated_stop_time=stop_time + 4.0,  # + 横向移动时间
            safety_margin=1.5,
        )

    def plan_slow_down(self, ego_state,
                       target_speed: float = 5.0,  # 5 m/s ≈ 18 km/h
                       dt: float = 0.1,
                       horizon: float = 8.0) -> EmergencyTrajectory:
        """
        降速行驶: 逐步减速到安全速度
        """
        num_steps = int(horizon / dt)
        v0 = getattr(ego_state, 'vx', 10.0)
        x0 = getattr(ego_state, 'x', 0.0)
        y0 = getattr(ego_state, 'y', 0.0)
        heading0 = getattr(ego_state, 'yaw', 0.0)

        decel = self.comfort_deceleration
        trajectory = np.zeros((num_steps, 4))

        for i in range(num_steps):
            t = (i + 1) * dt
            v = max(target_speed, v0 + decel * t)

            avg_v = (v0 + v) / 2
            traj_x = x0 + avg_v * t * np.cos(heading0)
            traj_y = y0

            trajectory[i, 0] = traj_x
            trajectory[i, 1] = traj_y
            trajectory[i, 2] = heading0
            trajectory[i, 3] = v

        return EmergencyTrajectory(
            trajectory=trajectory,
            mrm_type=MRMType.SLOW_DOWN,
            stop_point=(trajectory[-1, 0], trajectory[-1, 1]),
            estimated_stop_time=0.0,  # 不停止
            safety_margin=2.0,
        )

    def execute_mrm(self, mrm_type: MRMType, ego_state,
                    lane_boundaries=None) -> EmergencyTrajectory:
        """
        执行最小风险策略
        """
        if mrm_type == MRMType.SAFE_STOP:
            return self.trigger_emergency_brake(ego_state)
        elif mrm_type == MRMType.PULL_OVER:
            return self.plan_pull_over(ego_state, lane_boundaries)
        elif mrm_type == MRMType.SLOW_DOWN:
            return self.plan_slow_down(ego_state)
        elif mrm_type == MRMType.HANDOVER:
            # 降速 + 请求接管
            self._mrm_active = True
            self._mrm_type = MRMType.HANDOVER
            return self.plan_slow_down(ego_state, target_speed=8.0)
        else:
            # 默认: 安全停车
            return self.trigger_emergency_brake(ego_state)

    def cancel_mrm(self):
        """取消 MRM (仅在故障恢复后)"""
        self._mrm_active = False
        self._mrm_type = None

    @property
    def is_mrm_active(self) -> bool:
        return self._mrm_active
