"""
L4 自动驾驶系统级运行参数配置
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SystemConfig:
    """系统运行参数 —— 可运行时热更新"""

    # ---- 运行频率 ----
    perception_hz: float = 30.0          # 感知模块频率 [Hz]
    localization_hz: float = 100.0       # 定位模块频率 [Hz]
    prediction_hz: float = 20.0          # 预测模块频率 [Hz]
    planning_hz: float = 10.0            # 规划模块频率 [Hz]
    control_hz: float = 100.0            # 控制模块频率 [Hz]
    safety_check_hz: float = 50.0        # 安全检查频率 [Hz]

    # ---- 感知参数 ----
    lidar_range: float = 150.0           # LiDAR 最大探测距离 [m]
    lidar_points_per_scan: int = 128     # LiDAR 线数
    camera_image_width: int = 1920
    camera_image_height: int = 1208
    detection_confidence_threshold: float = 0.65
    tracking_max_age: int = 10           # 跟踪目标最大丢失帧数
    tracking_min_hits: int = 3           # 跟踪目标最小确认帧数

    # ---- 规划参数 ----
    planning_horizon_s: float = 8.0      # 运动规划时域 [s]
    planning_dt: float = 0.1             # 规划离散时间步长 [s]
    lane_change_min_gap: float = 25.0    # 变道最小安全间隙 [m]
    safe_following_time: float = 2.0     # 安全跟车时距 [s]
    emergency_brake_ttc: float = 2.5     # 紧急制动TTC阈值 [s]

    # ---- 控制参数 ----
    lookahead_distance_min: float = 8.0   # 最小预瞄距离 [m]
    lookahead_distance_max: float = 30.0  # 最大预瞄距离 [m]

    # ---- 安全参数 ----
    max_lateral_error: float = 0.3        # 最大横向误差 [m]
    max_longitudinal_error: float = 0.5   # 最大纵向误差 [m]
    system_latency_budget_ms: float = 100.0  # 系统延迟预算 [ms]

    # ---- ODD (Operational Design Domain) ----
    allowed_road_types: List[str] = field(default_factory=lambda: [
        "urban_arterial", "highway", "rural_road", "urban_local"
    ])
    max_rainfall_mmh: float = 50.0        # 最大允许降雨量
    max_fog_visibility_m: float = 100.0   # 最低雾天能见度
    min_ambient_light_lux: float = 0.5    # 最低环境光照
    operational_speed_range: List[float] = field(
        default_factory=lambda: [0.0, 33.3]  # 0 - 120 km/h
    )
