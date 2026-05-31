"""
L4 自动驾驶系统编排器 —— 多模块协调调度
"""
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

# 内部模块
from config import VehicleConfig, SystemConfig
from perception import (
    CameraProcessor, LidarProcessor, RadarProcessor,
    SensorFusion, ObjectDetector, ObjectTracker, LaneDetector,
)
from localization import GNSSLocalizer, IMUProcessor, MapMatcher, EKFLocalizer
from prediction import TrajectoryPredictor, BehaviorPredictor, InteractionModel
from planning import RoutePlanner, BehaviorPlanner, MotionPlanner, TrajectoryOptimizer
from control import LateralController, LongitudinalController, MPCController, VehicleInterface
from safety import SafetyMonitor, RedundancyChecker, EmergencyHandler
from safety.emergency_handler import MRMType
from hd_map import HDMapManager


class SystemState(Enum):
    """系统状态"""
    INIT = "init"
    STANDBY = "standby"
    RUNNING = "running"
    DEGRADED = "degraded"
    MRM = "mrm"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


@dataclass
class SystemHealth:
    """系统健康报告"""
    state: SystemState
    module_status: Dict[str, bool]
    sensor_health: Dict[str, bool]
    latencies: Dict[str, float]
    errors: List[str]
    fps: Dict[str, float]
    timestamp: float


class AutonomousDrivingSystem:
    """
    L4 自动驾驶系统主编排器

    模块调度:
      Perception  (30 Hz) → Localization (100 Hz) → Prediction (20 Hz)
                                                        ↓
      HD Map         →         Planning (10 Hz)    →   Control (100 Hz)
                                                        ↓
                          Safety Monitor (50 Hz) ← ← ← ←
    """

    def __init__(self, vehicle_config: VehicleConfig = None,
                 system_config: SystemConfig = None):
        self.vehicle_cfg = vehicle_config or VehicleConfig()
        self.sys_cfg = system_config or SystemConfig()

        # ---- 系统状态 ----
        self._state = SystemState.INIT
        self._running = False
        self._threads: List[threading.Thread] = []
        self._lock = threading.RLock()

        # ---- 数据缓冲 ----
        self._sensor_data: Dict[str, Any] = {}
        self._perception_output: Dict[str, Any] = {}
        self._localization_output: Dict[str, Any] = {}
        self._prediction_output: Dict[str, Any] = {}
        self._planning_output: Dict[str, Any] = {}
        self._control_output: Dict[str, Any] = {}
        self._safety_output: Dict[str, Any] = {}

        # ---- 延迟测量 ----
        self._latencies: Dict[str, deque] = {
            name: deque(maxlen=100)
            for name in ['perception', 'localization', 'prediction',
                         'planning', 'control', 'safety']
        }

        # ---- 错误计数 ----
        self._error_counts: Dict[str, int] = {}
        self._max_errors_before_degrade = 10

        # ---- 初始化模块 ----
        self._init_modules()

    def _init_modules(self):
        """初始化所有子模块"""
        print("[System] Initializing L4 autonomous driving modules...")

        # Perception
        self.camera = CameraProcessor([], [])
        self.lidar = LidarProcessor()
        self.radar = RadarProcessor()
        self.fusion = SensorFusion()
        self.detector = ObjectDetector(
            confidence_threshold=self.sys_cfg.detection_confidence_threshold
        )
        self.tracker = ObjectTracker(
            max_age=self.sys_cfg.tracking_max_age,
            min_hits=self.sys_cfg.tracking_min_hits,
        )
        self.lane_detector = LaneDetector()

        # Localization
        self.gnss = GNSSLocalizer()
        self.imu = IMUProcessor()
        self.map_matcher = MapMatcher()
        self.ekf = EKFLocalizer()

        # Prediction
        self.behavior_predictor = BehaviorPredictor()
        self.trajectory_predictor = TrajectoryPredictor(
            prediction_horizon=self.sys_cfg.planning_horizon_s,
            prediction_dt=self.sys_cfg.planning_dt,
        )
        self.interaction_model = InteractionModel()

        # Planning
        self.route_planner = RoutePlanner()
        self.behavior_planner = BehaviorPlanner(
            safe_following_time=self.sys_cfg.safe_following_time,
            lane_change_min_gap=self.sys_cfg.lane_change_min_gap,
        )
        self.motion_planner = MotionPlanner(
            planning_horizon=self.sys_cfg.planning_horizon_s,
            planning_dt=self.sys_cfg.planning_dt,
        )
        self.traj_optimizer = TrajectoryOptimizer(wheelbase=self.vehicle_cfg.wheelbase)

        # Control
        self.lateral_ctrl = LateralController(
            wheelbase=self.vehicle_cfg.wheelbase,
            max_steering_angle=self.vehicle_cfg.max_steering_angle,
            max_steering_rate=self.vehicle_cfg.max_steering_rate,
        )
        self.longitudinal_ctrl = LongitudinalController(
            vehicle_mass=self.vehicle_cfg.mass,
            max_accel=self.vehicle_cfg.max_acceleration,
            max_decel=self.vehicle_cfg.max_deceleration,
            max_jerk=self.vehicle_cfg.max_jerk,
        )
        self.mpc_ctrl = MPCController(wheelbase=self.vehicle_cfg.wheelbase)
        self.vehicle_iface = VehicleInterface(
            max_steering_angle=self.vehicle_cfg.max_steering_angle,
            max_steering_rate=self.vehicle_cfg.max_steering_rate,
        )

        # Safety
        self.safety_monitor = SafetyMonitor()
        self.redundancy_checker = RedundancyChecker()
        self.emergency_handler = EmergencyHandler()

        # HD Map
        self.hd_map = HDMapManager()

        # Data Logger
        self.data_logger = DataLogger()

        print("[System] All modules initialized.")

    def start(self):
        """启动自动驾驶系统"""
        print("[System] Starting L4 autonomous driving system...")
        self._running = True
        self._state = SystemState.RUNNING

        # 启动各模块线程
        self._threads = [
            threading.Thread(target=self._perception_loop, daemon=True,
                             name="Perception"),
            threading.Thread(target=self._localization_loop, daemon=True,
                             name="Localization"),
            threading.Thread(target=self._prediction_loop, daemon=True,
                             name="Prediction"),
            threading.Thread(target=self._planning_loop, daemon=True,
                             name="Planning"),
            threading.Thread(target=self._control_loop, daemon=True,
                             name="Control"),
            threading.Thread(target=self._safety_loop, daemon=True,
                             name="Safety"),
        ]

        for t in self._threads:
            t.start()

        print("[System] All threads started. System is RUNNING.")

    def stop(self):
        """停止系统"""
        print("[System] Stopping autonomous driving system...")
        self._running = False
        self._state = SystemState.SHUTDOWN

        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)

        print("[System] System stopped.")

    def _perception_loop(self):
        """
        感知主循环 (30 Hz)
        Pipeline: Camera → LiDAR → Radar → Fusion → Detection → Tracking → Lane
        """
        period = 1.0 / self.sys_cfg.perception_hz
        print(f"[Perception] Started at {self.sys_cfg.perception_hz} Hz")

        while self._running:
            t_start = time.perf_counter()

            try:
                # 获取传感器数据
                raw_images = self._sensor_data.get('camera_images', [])
                lidar_pc = self._sensor_data.get('lidar_pointcloud',
                                                  np.array([]).reshape(0, 4))
                radar_adc = self._sensor_data.get('radar_adc', None)

                # Camera processing
                camera_output = {}
                if raw_images:
                    camera_output = self.camera.process(raw_images)

                # LiDAR processing
                lidar_output = self.lidar.process(lidar_pc)

                # Radar processing
                radar_output = self.radar.process(
                    radar_adc, dt=period
                ) if radar_adc is not None else {'tracks': []}

                # Sensor fusion → detection
                detections = self.detector.detect(
                    lidar_pc, lidar_output['clusters'],
                    camera_output.get('bev', np.zeros((1, 1, 3))),
                    camera_output.get('features', {}),
                    radar_output.get('tracks', []),
                )

                # Multi-object tracking
                tracked = self.tracker.update(detections, dt=period)

                # Lane detection
                camera_lane_pts = self.lane_detector.extract_lane_points_from_bev(
                    camera_output.get('lane_mask', np.zeros((512, 512)))
                )
                lidar_lane_pts = self.lane_detector.extract_lane_points_from_lidar(
                    lidar_output.get('ground_points', np.array([]).reshape(0, 4))
                )
                lane_boundaries = self.lane_detector.detect_lanes(
                    camera_lane_pts, lidar_lane_pts
                )

                # 多传感器融合
                fusion_output = self.fusion.fuse(
                    detections, [], radar_output.get('tracks', []),
                    sensor_timestamps={
                        'lidar': time.time(),
                        'camera': time.time(),
                        'radar': time.time(),
                    },
                    sensor_extrinsics={},
                )

                # 输出
                with self._lock:
                    self._perception_output = {
                        'camera': camera_output,
                        'lidar': lidar_output,
                        'radar': radar_output,
                        'detections': detections,
                        'tracked_objects': tracked,
                        'lane_boundaries': lane_boundaries,
                        'fused_objects': fusion_output['fused_objects'],
                        'sensor_degradation': fusion_output['sensor_degradation'],
                        'timestamp': time.time(),
                    }

            except Exception as e:
                self._record_error('perception', str(e))

            # 频率控制
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0, period - elapsed)
            time.sleep(sleep_time)

    def _localization_loop(self):
        """
        定位主循环 (100 Hz)
        Pipeline: GNSS → IMU(Propagation) → Map Matching → EKF(Update)
        """
        period = 1.0 / self.sys_cfg.localization_hz
        print(f"[Localization] Started at {self.sys_cfg.localization_hz} Hz")

        while self._running:
            t_start = time.perf_counter()

            try:
                gnss_obs = self._sensor_data.get('gnss_observation')
                imu_meas = self._sensor_data.get('imu_measurement')
                wheel_speed = self._sensor_data.get('wheel_speed', 0.0)

                dt = period

                # IMU 预测
                if imu_meas is not None:
                    accel = np.array([imu_meas.accel_x, imu_meas.accel_y,
                                      imu_meas.accel_z])
                    gyro = np.array([imu_meas.gyro_x, imu_meas.gyro_y,
                                     imu_meas.gyro_z])
                    self.ekf.predict(accel, gyro, dt)

                # GNSS 更新
                if gnss_obs is not None:
                    gnss_quality = self.gnss.assess_solution_quality(gnss_obs)
                    if gnss_quality > 0.3:
                        e, n, u, ve, vn, vu = self.gnss.get_enu_observation(gnss_obs)
                        gnss_pos = np.array([e, n, u])
                        gnss_vel = np.array([ve, vn, vu])

                        pos_cov = np.eye(3) * (1.0 / max(gnss_quality, 0.1)) ** 2
                        self.ekf.update_gnss_position(gnss_pos, pos_cov)
                        self.ekf.update_gnss_velocity(gnss_vel, pos_cov * 0.5)

                # 轮速更新
                if wheel_speed > 0.1:
                    self.ekf.update_wheel_speed(wheel_speed, speed_std=0.5)

                # 获取状态估计
                vehicle_state = self.ekf.get_state()

                with self._lock:
                    self._localization_output = {
                        'vehicle_state': vehicle_state,
                        'gnss_quality': gnss_quality if gnss_obs else 0.0,
                        'timestamp': time.time(),
                    }

            except Exception as e:
                self._record_error('localization', str(e))

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0, period - elapsed))

    def _prediction_loop(self):
        """
        预测主循环 (20 Hz)
        Pipeline: Behavior Prediction → Trajectory Prediction → Interaction Model
        """
        period = 1.0 / self.sys_cfg.prediction_hz
        print(f"[Prediction] Started at {self.sys_cfg.prediction_hz} Hz")

        while self._running:
            t_start = time.perf_counter()

            try:
                with self._lock:
                    tracked = self._perception_output.get('tracked_objects', [])
                    lane_bounds = self._perception_output.get('lane_boundaries', [])
                    veh_state = self._localization_output.get('vehicle_state')

                if veh_state is None:
                    time.sleep(period)
                    continue

                # Behavior prediction
                behaviors = self.behavior_predictor.predict_behavior(
                    tracked, veh_state, lane_bounds
                )

                # Trajectory prediction
                trajectories = self.trajectory_predictor.predict(
                    tracked, behaviors, lane_bounds
                )

                # Interaction graph
                interaction_graph = self.interaction_model.build_interaction_graph(
                    veh_state, tracked, behaviors
                )
                interaction_analysis = \
                    self.interaction_model.analyze_ego_interactions(
                        interaction_graph
                    )

                with self._lock:
                    self._prediction_output = {
                        'behaviors': behaviors,
                        'trajectories': trajectories,
                        'interaction_graph': interaction_graph,
                        'interaction_analysis': interaction_analysis,
                        'timestamp': time.time(),
                    }

            except Exception as e:
                self._record_error('prediction', str(e))

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0, period - elapsed))

    def _planning_loop(self):
        """
        规划主循环 (10 Hz)
        Pipeline: Route → Behavior → Motion → Trajectory Optimization
        """
        period = 1.0 / self.sys_cfg.planning_hz
        print(f"[Planning] Started at {self.sys_cfg.planning_hz} Hz")

        while self._running:
            t_start = time.perf_counter()

            try:
                with self._lock:
                    veh_state = self._localization_output.get('vehicle_state')
                    tracked = self._perception_output.get('tracked_objects', [])
                    fused = self._perception_output.get('fused_objects', [])
                    lane_bounds = self._perception_output.get('lane_boundaries', [])
                    pred_output = self._prediction_output
                    interaction = pred_output.get('interaction_analysis', {})
                    trajectories = pred_output.get('trajectories', [])

                if veh_state is None:
                    time.sleep(period)
                    continue

                # Route planning (全局, 低频更新)
                # 简化: 构建参考线
                reference_line = np.zeros((100, 4))
                for i in range(100):
                    reference_line[i] = [i * 10.0, 0.0, 0.0, 0.0]  # 直路

                # Behavior decision
                behavior = self.behavior_planner.decide(
                    veh_state, fused, lane_bounds,
                    route_maneuvers=[{'maneuver': 'follow_lane'}],
                    interaction_graph=interaction,
                    dt=period,
                )

                # Motion planning
                motion_plan = self.motion_planner.plan(
                    veh_state, reference_line, fused,
                    trajectories, behavior, lane_bounds,
                )

                # Trajectory optimization
                initial_state = np.array([veh_state.x, veh_state.y,
                                          veh_state.yaw, veh_state.vx])
                ref_traj_4 = np.array([
                    motion_plan.best_trajectory.x,
                    motion_plan.best_trajectory.y,
                    motion_plan.best_trajectory.heading,
                    motion_plan.best_trajectory.v,
                ])

                opt_states, opt_controls, opt_success = \
                    self.traj_optimizer.optimize(ref_traj_4, initial_state)

                with self._lock:
                    self._planning_output = {
                        'behavior': behavior,
                        'motion_plan': motion_plan,
                        'optimized_trajectory': opt_states,
                        'optimized_controls': opt_controls,
                        'planning_success': opt_success,
                        'reference_line': reference_line,
                        'timestamp': time.time(),
                    }

            except Exception as e:
                self._record_error('planning', str(e))

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0, period - elapsed))

    def _control_loop(self):
        """
        控制主循环 (100 Hz)
        Pipeline: Lateral Control → Longitudinal Control → Vehicle Command
        """
        period = 1.0 / self.sys_cfg.control_hz
        print(f"[Control] Started at {self.sys_cfg.control_hz} Hz")

        while self._running:
            t_start = time.perf_counter()

            try:
                with self._lock:
                    veh_state = self._localization_output.get('vehicle_state')
                    planning = self._planning_output
                    safety = self._safety_output

                if veh_state is None or not planning:
                    time.sleep(period)
                    continue

                # 安全检查: MRM 优先
                if safety and safety.get('mrm_active', False):
                    mrm_traj = safety.get('mrm_trajectory')
                    if mrm_traj is not None:
                        # 执行紧急轨迹
                        steering, _ = self.lateral_ctrl.compute_steering(
                            veh_state, mrm_traj.trajectory, 0, 'stanley'
                        )
                        throttle = 0.0
                        brake = 1.0  # 全力制动
                        self.vehicle_iface.send_command(
                            self.vehicle_iface.build_command(
                                steering, throttle, brake,
                                gear=1, turn_signal=0,
                                timestamp=time.time(),
                            ),
                            time.time(),
                        )
                        continue

                # 正常控制
                opt_traj = planning.get('optimized_trajectory')
                behavior = planning.get('behavior')

                if opt_traj is None or behavior is None:
                    time.sleep(period)
                    continue

                # 横向控制
                steering, lateral_err = self.lateral_ctrl.compute_steering(
                    veh_state, opt_traj, 0, method='stanley'
                )

                # 纵向控制
                target_speed = behavior.target_speed
                current_speed = veh_state.vx

                throttle, brake, target_accel = self.longitudinal_ctrl.control(
                    target_speed=target_speed,
                    current_speed=current_speed,
                    dt=period,
                )

                # 构建并发送指令
                cmd = self.vehicle_iface.build_command(
                    steering_angle=steering,
                    throttle=throttle,
                    brake=brake,
                    gear=1,
                    turn_signal=0,
                    timestamp=time.time(),
                )

                success = self.vehicle_iface.send_command(cmd, time.time())

                # 看门狗
                watchdog_cmd = self.vehicle_iface.check_watchdog(time.time())
                if watchdog_cmd:
                    self.vehicle_iface.send_command(watchdog_cmd, time.time())

                with self._lock:
                    self._control_output = {
                        'command': cmd,
                        'success': success,
                        'lateral_error': lateral_err,
                        'target_accel': target_accel,
                        'timestamp': time.time(),
                    }

            except Exception as e:
                self._record_error('control', str(e))

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0, period - elapsed))

    def _safety_loop(self):
        """
        安全监控主循环 (50 Hz)
        """
        period = 1.0 / self.sys_cfg.safety_check_hz
        print(f"[Safety] Started at {self.sys_cfg.safety_check_hz} Hz")

        while self._running:
            t_start = time.perf_counter()

            try:
                with self._lock:
                    veh_state = self._localization_output.get('vehicle_state')
                    planning = self._planning_output
                    prediction = self._prediction_output
                    fusion = self._perception_output

                if veh_state is None:
                    time.sleep(period)
                    continue

                interaction = prediction.get('interaction_analysis', {})
                lateral_err = self._control_output.get('lateral_error', 0.0)
                ekf_cov = veh_state.covariance if hasattr(veh_state, 'covariance') \
                    else np.eye(15)
                gnss_quality = self._localization_output.get('gnss_quality', 0.0)
                sensor_health = fusion.get('sensor_health',
                                           {'lidar': True, 'camera': True, 'radar': True})
                sensor_ages = {'lidar': 0.05, 'camera': 0.05, 'radar': 0.05}
                motion_plan = planning.get('motion_plan')
                planning_success = planning.get('planning_success', False)
                planning_latency = 0.05  # 假设

                # 安全评估
                assessment = self.safety_monitor.assess(
                    interaction_graph=interaction,
                    lateral_error=lateral_err,
                    current_speed=veh_state.vx,
                    speed_limit=33.3,
                    ekf_covariance=ekf_cov,
                    gnss_quality=gnss_quality,
                    sensor_health=sensor_health,
                    sensor_data_ages=sensor_ages,
                    motion_plan=motion_plan,
                    planning_success=planning_success,
                    planning_latency=planning_latency,
                )

                mrm_trajectory = None
                mrm_active = False

                # MRM 触发
                if assessment.trigger_mrm:
                    print(f"[Safety] MRM triggered! Level={assessment.level.name}, "
                          f"Type={assessment.mrm_type}")

                    self._state = SystemState.MRM

                    mrm_type_map = {
                        'safe_stop': MRMType.SAFE_STOP,
                        'pull_over': MRMType.PULL_OVER,
                    }

                    mrm_traj = self.emergency_handler.execute_mrm(
                        mrm_type_map.get(assessment.mrm_type,
                                        MRMType.SAFE_STOP),
                        veh_state,
                    )
                    mrm_trajectory = mrm_traj
                    mrm_active = True

                with self._lock:
                    self._safety_output = {
                        'assessment': assessment,
                        'mrm_active': mrm_active,
                        'mrm_trajectory': mrm_trajectory,
                        'timestamp': time.time(),
                    }

                # 降级处理
                if assessment.level.value >= 4:  # WARNING
                    self._state = SystemState.DEGRADED
                elif not mrm_active:
                    self._state = SystemState.RUNNING

            except Exception as e:
                self._record_error('safety', str(e))

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0, period - elapsed))

    def feed_sensor_data(self, sensor_type: str, data: Any):
        """外部接口: 注入传感器数据"""
        with self._lock:
            self._sensor_data[sensor_type] = data

    def get_system_output(self) -> Dict:
        """获取系统当前输出"""
        with self._lock:
            return {
                'state': self._state,
                'perception': {
                    'num_tracked': len(
                        self._perception_output.get('tracked_objects', [])
                    ),
                },
                'localization': {
                    'x': self._localization_output.get('vehicle_state').x
                    if self._localization_output.get('vehicle_state') else 0,
                    'y': self._localization_output.get('vehicle_state').y
                    if self._localization_output.get('vehicle_state') else 0,
                },
                'planning': {
                    'behavior': self._planning_output.get('behavior').state.name
                    if self._planning_output.get('behavior') else 'N/A',
                },
                'safety': {
                    'level': self._safety_output.get('assessment').level.name
                    if self._safety_output.get('assessment') else 'N/A',
                    'score': self._safety_output.get('assessment').overall_score
                    if self._safety_output.get('assessment') else 0,
                },
                'control': self._control_output,
            }

    def get_health(self) -> SystemHealth:
        """获取系统健康状态"""
        with self._lock:
            return SystemHealth(
                state=self._state,
                module_status={
                    'perception': len(self._perception_output) > 0,
                    'localization': len(self._localization_output) > 0,
                    'prediction': len(self._prediction_output) > 0,
                    'planning': len(self._planning_output) > 0,
                    'control': len(self._control_output) > 0,
                    'safety': len(self._safety_output) > 0,
                },
                sensor_health=self._perception_output.get(
                    'sensor_degradation', {}
                ),
                latencies={
                    name: np.mean(list(d)) if d else 0
                    for name, d in self._latencies.items()
                },
                errors=list(self._error_counts.keys()),
                fps={
                    name: len(d) / max(np.mean(list(d)), 0.01)
                    for name, d in self._latencies.items() if d
                },
                timestamp=time.time(),
            )

    def _record_error(self, module: str, error_msg: str):
        """记录模块错误"""
        key = f"{module}:{error_msg[:50]}"
        self._error_counts[key] = self._error_counts.get(key, 0) + 1

        # 错误过多 → 降级
        total_errors = sum(self._error_counts.values())
        if total_errors > self._max_errors_before_degrade:
            self._state = SystemState.DEGRADED
            print(f"[System] Too many errors ({total_errors}) — DEGRADING")


class DataLogger:
    """数据记录器"""

    def __init__(self, max_buffer: int = 10000):
        self._buffer: deque = deque(maxlen=max_buffer)
        self._recording = False

    def log(self, data: Dict):
        if self._recording:
            self._buffer.append({
                'timestamp': time.time(),
                'data': data,
            })

    def start_recording(self):
        self._recording = True

    def stop_recording(self):
        self._recording = False

    def export(self, filepath: str):
        import json
        with open(filepath, 'w') as f:
            json.dump(list(self._buffer), f, default=str)
