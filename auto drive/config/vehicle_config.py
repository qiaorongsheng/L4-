"""
L4 自动驾驶车辆物理参数与运动学约束配置
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class VehicleConfig:
    """车辆运动学与动力学参数"""

    # ---- 尺寸 ----
    wheelbase: float = 2.85             # 轴距 [m]
    track_width: float = 1.62           # 轮距 [m]
    length: float = 4.85                # 车长 [m]
    width: float = 1.95                 # 车宽 [m]
    mass: float = 2100.0                # 整备质量 [kg]
    iz: float = 3580.0                  # 横摆转动惯量 [kg·m²]

    # ---- 轮胎 ----
    cornering_stiffness_front: float = 155494.0   # 前轮侧偏刚度 [N/rad]
    cornering_stiffness_rear: float = 155494.0    # 后轮侧偏刚度 [N/rad]
    tire_radius: float = 0.355                   # 轮胎滚动半径 [m]

    # ---- 运动学约束 ----
    max_steering_angle: float = 0.65     # 最大转向角 [rad] ≈ 37°
    max_steering_rate: float = 0.52      # 最大转向速率 [rad/s] ≈ 30°/s
    max_acceleration: float = 3.0        # 最大加速度 [m/s²]
    max_deceleration: float = -6.0       # 最大减速度 [m/s²]
    max_jerk: float = 10.0              # 最大加加速度 [m/s³]
    max_speed: float = 33.3             # 最高车速 [m/s] ≈ 120 km/h

    # ---- 执行器延迟 ----
    steering_delay: float = 0.08         # 转向延迟 [s]
    throttle_delay: float = 0.05         # 油门延迟 [s]
    brake_delay: float = 0.03            # 制动延迟 [s]

    # ---- 传感器布置 ----
    lidar_position: Tuple[float, float, float] = (0.0, 0.0, 1.75)   # 车顶
    camera_positions: Tuple = (
        (0.0, 0.0, 1.60),       # 前视 (车前挡风玻璃)
        (0.0, 0.0, 1.55),       # 前视广角
        (0.0, 0.0, 1.55),       # 后视
        (1.95, -0.75, 0.90),    # 左后侧视
        (-1.95, -0.75, 0.90),   # 右后侧视
    )
    radar_positions: Tuple = (
        (0.0, 1.20, 0.45),      # 前向长距雷达
        (0.0, -1.20, 0.45),     # 后向长距雷达
        (1.85, 0.30, 0.45),     # 左前角雷达
        (-1.85, 0.30, 0.45),    # 右前角雷达
    )

    @property
    def wheelbase_half(self) -> float:
        return self.wheelbase / 2

    @property
    def lf(self) -> float:
        """前轴到质心距离 (假设质心居中)"""
        return self.wheelbase_half

    @property
    def lr(self) -> float:
        """后轴到质心距离"""
        return self.wheelbase_half
