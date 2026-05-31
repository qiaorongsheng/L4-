"""
车辆接口抽象层 —— 统一控制指令输出与车辆状态反馈
"""
import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class ControlCommand:
    """标准化控制指令"""
    # 转向
    steering_angle: float       # 方向盘转角 [rad]
    steering_rate: float        # 转向速率 [rad/s]

    # 纵向
    throttle: float             # 油门开度 [0, 1]
    brake: float                # 制动压力 [0, 1]

    # 档位
    gear: int                   # -1=R, 0=N, 1=D

    # 信号灯
    turn_signal: int            # -1=左, 0=关, 1=右
    hazard_lights: bool = False
    horn: bool = False

    # 时间戳 & 生命周期
    timestamp: float = 0.0
    valid_until: float = 0.1    # 指令有效期 [s]


@dataclass
class VehicleFeedback:
    """车辆状态反馈"""
    # 底盘状态
    steering_angle: float       # 实际方向盘转角 [rad]
    wheel_speed_fl: float       # 左前轮速 [m/s]
    wheel_speed_fr: float       # 右前轮速 [m/s]
    wheel_speed_rl: float       # 左后轮速 [m/s]
    wheel_speed_rr: float       # 右后轮速 [m/s]

    # 动力系统
    throttle_pedal: float       # 实际油门踏板位置 [0,1]
    brake_pressure: float       # 实际制动压力 [MPa]
    gear: int                   # 实际档位
    engine_rpm: float           # 发动机转速 [rpm]
    motor_torque: float         # 电机扭矩 [Nm] (电动车)

    # 状态标志
    abs_active: bool = False
    esc_active: bool = False
    tcs_active: bool = False

    timestamp: float = 0.0


class VehicleInterface:
    """
    车辆接口层:
      - 控制指令验证 (合理性检查)
      - 指令平滑与限幅
      - CAN 总线通信模拟
      - 安全看门狗 (指令超时自动制动)
    """

    def __init__(self, max_steering_angle: float = 0.65,
                 max_steering_rate: float = 0.52,
                 command_timeout: float = 0.2):
        self.max_steering_angle = max_steering_angle
        self.max_steering_rate = max_steering_rate
        self.command_timeout = command_timeout

        # 上一个有效指令
        self._last_command: Optional[ControlCommand] = None
        self._last_command_time: float = 0.0

        # 回调函数 (实际部署对接 CAN 驱动)
        self._can_send_callback: Optional[Callable] = None
        self._can_recv_callback: Optional[Callable] = None

        # 统计
        self._command_count: int = 0

    def register_can_callbacks(self, send_cb: Callable,
                               recv_cb: Callable):
        """注册 CAN 总线回调函数"""
        self._can_send_callback = send_cb
        self._can_recv_callback = recv_cb

    def validate_command(self, cmd: ControlCommand) -> bool:
        """
        控制指令合理性验证:
          - 转向角范围
          - 转向速率范围
          - 油门/制动互斥
          - 物理范围检查
        """
        # 转向角范围
        if abs(cmd.steering_angle) > self.max_steering_angle * 1.1:
            return False

        # 转向速率范围
        if abs(cmd.steering_rate) > self.max_steering_rate * 1.5:
            return False

        # 油门/制动互斥 (安全: 不可同时)
        if cmd.throttle > 0.05 and cmd.brake > 0.05:
            return False

        # 范围检查
        if not (0.0 <= cmd.throttle <= 1.0):
            return False
        if not (0.0 <= cmd.brake <= 1.0):
            return False
        if cmd.gear not in (-1, 0, 1):
            return False
        if cmd.turn_signal not in (-1, 0, 1):
            return False

        return True

    def smooth_command(self, cmd: ControlCommand,
                       prev_cmd: Optional[ControlCommand] = None,
                       dt: float = 0.01) -> ControlCommand:
        """
        指令平滑 (防止突变)
        """
        if prev_cmd is None:
            return cmd

        # 转向平滑
        max_delta = self.max_steering_rate * dt
        steer_diff = cmd.steering_angle - prev_cmd.steering_angle
        if abs(steer_diff) > max_delta:
            cmd.steering_angle = prev_cmd.steering_angle + \
                                 np.sign(steer_diff) * max_delta

        # 油门/制动平滑 (一阶低通)
        alpha = 0.3  # 平滑系数
        cmd.throttle = alpha * cmd.throttle + (1 - alpha) * prev_cmd.throttle
        cmd.brake = alpha * cmd.brake + (1 - alpha) * prev_cmd.brake

        return cmd

    def build_command(self, steering_angle: float,
                      throttle: float, brake: float,
                      gear: int = 1,
                      turn_signal: int = 0,
                      timestamp: float = 0.0) -> ControlCommand:
        """
        构建标准化控制指令
        """
        return ControlCommand(
            steering_angle=float(np.clip(steering_angle,
                                         -self.max_steering_angle,
                                         self.max_steering_angle)),
            steering_rate=0.0,
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
            gear=gear,
            turn_signal=turn_signal,
            timestamp=timestamp,
            valid_until=timestamp + self.command_timeout,
        )

    def send_command(self, cmd: ControlCommand,
                     current_time: float) -> bool:
        """
        发送控制指令到车辆 (通过 CAN)
        返回: 是否发送成功
        """
        # 指令验证
        if not self.validate_command(cmd):
            return False

        # 指令平滑
        if self._last_command is not None:
            dt = current_time - self._last_command_time
            cmd = self.smooth_command(cmd, self._last_command,
                                      max(dt, 0.001))

        # 通过 CAN 发送 (模拟)
        if self._can_send_callback:
            self._can_send_callback(cmd)

        self._last_command = cmd
        self._last_command_time = current_time
        self._command_count += 1

        return True

    def check_watchdog(self, current_time: float) -> Optional[ControlCommand]:
        """
        看门狗检查: 如果指令超时, 自动紧急制动
        """
        if self._last_command is None:
            return None

        if current_time - self._last_command_time > self.command_timeout:
            # 指令超时 → 紧急制动
            return ControlCommand(
                steering_angle=self._last_command.steering_angle,  # 保持方向
                steering_rate=0.0,
                throttle=0.0,
                brake=1.0,
                gear=1,
                turn_signal=0,
                hazard_lights=True,
                timestamp=current_time,
            )

        return None
