"""
纵向控制器 —— 双 PID + 前馈 + 坡度补偿
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PIDGains:
    """PID 参数"""
    kp: float
    ki: float
    kd: float
    integral_max: float = 5.0
    output_max: float = 1.0
    output_min: float = -1.0


class LongitudinalController:
    """
    纵向控制器:
      - 级联 PID (位置环 + 速度环)
      - 前馈补偿 (坡度 + 风阻)
      - 加速度 → 油门/制动映射
      - 舒适性约束 (jerk limiting)
    """

    def __init__(self, vehicle_mass: float = 2100.0,
                 wheel_radius: float = 0.355,
                 max_accel: float = 3.0,
                 max_decel: float = -6.0,
                 max_jerk: float = 10.0):
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_jerk = max_jerk

        # 速度环 PID
        self._speed_pid = PIDGains(
            kp=0.8, ki=0.1, kd=0.05,
            integral_max=3.0, output_max=max_accel,
            output_min=max_decel,
        )

        # 位置环 PID (用于停车/跟车)
        self._position_pid = PIDGains(
            kp=0.3, ki=0.02, kd=0.1,
            integral_max=2.0, output_max=10.0,
            output_min=-10.0,
        )

        # PID 积分项
        self._speed_integral = 0.0
        self._position_integral = 0.0
        self._prev_error_speed = 0.0
        self._prev_error_position = 0.0
        self._prev_accel = 0.0

    def reset(self):
        """重置 PID 积分项"""
        self._speed_integral = 0.0
        self._position_integral = 0.0
        self._prev_error_speed = 0.0
        self._prev_error_position = 0.0
        self._prev_accel = 0.0

    def speed_control(self, target_speed: float, current_speed: float,
                      dt: float = 0.01) -> float:
        """
        速度环 PID 控制
        返回: 目标加速度 [m/s²]
        """
        error = target_speed - current_speed

        # 积分 (带 anti-windup)
        self._speed_integral += error * dt
        self._speed_integral = np.clip(self._speed_integral,
                                       -self._speed_pid.integral_max,
                                       self._speed_pid.integral_max)

        # 微分
        derivative = (error - self._prev_error_speed) / max(dt, 0.001)
        self._prev_error_speed = error

        # PID 输出
        pid = self._speed_pid
        accel = (pid.kp * error +
                 pid.ki * self._speed_integral +
                 pid.kd * derivative)

        # 限幅
        accel = np.clip(accel, pid.output_min, pid.output_max)

        # Jerk 限制
        jerk = (accel - self._prev_accel) / dt
        if abs(jerk) > self.max_jerk:
            accel = self._prev_accel + np.sign(jerk) * self.max_jerk * dt

        self._prev_accel = accel
        return float(accel)

    def distance_control(self, target_distance: float,
                         current_distance: float,
                         ego_speed: float, target_speed: float,
                         dt: float = 0.01) -> float:
        """
        距离环 PID 控制 (跟车/停车)
        返回: 目标速度 [m/s]
        """
        error = target_distance - current_distance

        self._position_integral += error * dt
        self._position_integral = np.clip(self._position_integral,
                                          -self._position_pid.integral_max,
                                          self._position_pid.integral_max)

        derivative = (error - self._prev_error_position) / max(dt, 0.001)
        self._prev_error_position = error

        pid = self._position_pid
        speed_correction = (pid.kp * error +
                            pid.ki * self._position_integral +
                            pid.kd * derivative)

        desired_speed = target_speed + speed_correction
        desired_speed = np.clip(desired_speed, 0.0, 33.3)

        return float(desired_speed)

    def adaptive_cruise_control(self, ego_speed: float,
                                lead_distance: float,
                                lead_speed: float,
                                safe_time_gap: float = 2.0,
                                dt: float = 0.01) -> float:
        """
        自适应巡航 (ACC) 控制
        恒时距 (Constant Time Gap) 策略
        """
        # 目标安全距离 = 当前速度 × 安全时距 + 最小间隙
        desired_gap = ego_speed * safe_time_gap + 5.0

        # 距离误差
        gap_error = lead_distance - desired_gap

        if gap_error > 5.0:
            # 距离充裕 → 加速到巡航速度
            return self.speed_control(33.3, ego_speed, dt)
        elif gap_error > 0:
            # 适当距离 → 跟随前车速度
            return self.speed_control(lead_speed, ego_speed, dt)
        else:
            # 距离不足 → 减速
            desired_speed = self.distance_control(
                desired_gap, lead_distance, ego_speed, lead_speed, dt
            )
            return self.speed_control(desired_speed, ego_speed, dt)

    def feedforward_compensation(self, slope_angle: float = 0.0,
                                 drag_coefficient: float = 0.3,
                                 frontal_area: float = 2.2,
                                 air_density: float = 1.225,
                                 current_speed: float = 0.0) -> float:
        """
        前馈补偿:
          - 坡度补偿: g * sin(θ)
          - 空气阻力: 0.5 * ρ * Cd * A * v²
        返回: 补偿加速度 [m/s²]
        """
        g = 9.80665

        # 坡度补偿
        slope_compensation = g * np.sin(slope_angle)

        # 风阻补偿
        drag_force = 0.5 * air_density * drag_coefficient * \
                     frontal_area * current_speed ** 2
        drag_compensation = drag_force / self.vehicle_mass

        return float(slope_compensation + drag_compensation)

    def accel_to_throttle_brake(self, target_accel: float,
                                current_speed: float,
                                slope_angle: float = 0.0
                                ) -> Tuple[float, float]:
        """
        加速度 → 油门/制动指令映射
        返回: (throttle [0,1], brake [0,1])
        """
        # 前馈补偿
        ff = self.feedforward_compensation(slope_angle,
                                           current_speed=current_speed)
        net_accel = target_accel + ff

        # 车辆阻力功率 (简化查表模型)
        # 怠速爬行力
        coastdown_force = 100 + 5 * current_speed + 0.3 * current_speed ** 2

        if net_accel > 0:
            # 加速 → 油门控制
            required_force = self.vehicle_mass * net_accel + coastdown_force
            max_engine_force = 8000  # N (发动机最大牵引力)
            throttle = np.clip(required_force / max_engine_force, 0.0, 1.0)
            brake = 0.0
        else:
            # 减速 → 制动控制 (先松油门, 再制动)
            # 发动机制动
            engine_brake_force = 500 + 2 * current_speed

            if abs(net_accel) * self.vehicle_mass < engine_brake_force:
                throttle = 0.0
                brake = 0.0  # 发动机制动足够
            else:
                throttle = 0.0
                # 所需制动力
                required_brake_force = abs(net_accel) * self.vehicle_mass - \
                                       engine_brake_force
                max_brake_force = 25000  # N
                brake = np.clip(required_brake_force / max_brake_force,
                                0.0, 1.0)

        return (float(throttle), float(brake))

    def emergency_brake(self) -> Tuple[float, float]:
        """紧急制动 (最大制动)"""
        self.reset()
        return (0.0, 1.0)  # (throttle=0, brake=1)

    def control(self, target_speed: float, current_speed: float,
                target_distance: Optional[float] = None,
                current_distance: Optional[float] = None,
                lead_speed: Optional[float] = None,
                slope_angle: float = 0.0,
                dt: float = 0.01) -> Tuple[float, float, float]:
        """
        纵向控制统一入口
        返回: (throttle, brake, target_accel)
        """
        # Step 1: 计算目标加速度
        if target_distance is not None and current_distance is not None and \
           lead_speed is not None:
            # 跟车模式 (ACC)
            target_accel = self.adaptive_cruise_control(
                current_speed, current_distance, lead_speed,
                dt=dt,
            )
        elif target_speed < 0.5:
            # 停车模式 → 距离控制
            if target_distance is not None and current_distance is not None:
                desired_speed = self.distance_control(
                    target_distance, current_distance,
                    current_speed, 0.0, dt
                )
            else:
                desired_speed = 0.0
            target_accel = self.speed_control(desired_speed, current_speed, dt)
        else:
            # 速度控制模式
            target_accel = self.speed_control(target_speed, current_speed, dt)

        # Step 2: 加速度 → 油门/制动
        throttle, brake = self.accel_to_throttle_brake(
            target_accel, current_speed, slope_angle
        )

        return (throttle, brake, target_accel)
