#!/usr/bin/env python3
"""
L4 自动驾驶系统 — 主入口
===========================

L4 级别: 在定义的 ODD (Operational Design Domain) 内完全无需人类干预。
本系统覆盖 L4 全栈模块:

  Perception    → 多传感器融合目标检测与跟踪
  Localization  → GNSS+IMU+Map Matching 多源融合定位
  Prediction    → 多模态轨迹预测 + 行为意图识别 + 交互建模
  Planning      → 全局路由 + 行为决策 + Lattice 运动规划 + 轨迹优化
  Control       → MPC + Stanley + 双 PID 横纵向控制
  Safety        → 多层安全监控 + 冗余检查 + MRM 紧急处理
  HD Map        → 高精地图加载/查询/拓扑分析

运行方式:
  python main.py                  # 启动完整系统 (需要传感器输入)
  python main.py --sim            # 模拟模式
  python main.py --replay <file>  # 回放模式
"""

import sys
import time
import argparse
import signal
import threading
import numpy as np

from config import VehicleConfig, SystemConfig
from system import AutonomousDrivingSystem


# ============================================================
# 全局系统句柄
# ============================================================
g_system: AutonomousDrivingSystem = None


def signal_handler(sig, frame):
    """优雅退出"""
    print("\n[Main] Received shutdown signal...")
    if g_system:
        g_system.stop()
    sys.exit(0)


# ============================================================
# 模拟传感器数据生成器 (用于测试)
# ============================================================
class SimulatedSensors:
    """
    模拟传感器数据生成器:
      模拟相机图像、LiDAR 点云、雷达信号、GNSS 观测、IMU 数据
      用于开发测试 — 不需要真实硬件
    """

    def __init__(self):
        self._t = 0.0
        self._ego_x = 0.0
        self._ego_y = 0.0
        self._ego_v = 10.0  # 10 m/s = 36 km/h
        self._ego_yaw = 0.0

    def generate_camera_images(self):
        """生成模拟相机图像 (随机噪声 + 车道线)"""
        images = []
        for _ in range(5):
            # 512x512 RGB 噪声图像 + 模拟车道线
            img = np.random.randn(512, 512, 3) * 0.1 + 0.5
            # 在中间画一条白线 (模拟车道线)
            img[:, 250:260, :] = 0.9
            images.append(img.astype(np.float32))
        return images

    def generate_lidar_pointcloud(self):
        """生成模拟 LiDAR 点云"""
        num_points = 120000  # 128 线
        points = np.zeros((num_points, 4))

        # 随机点散布
        points[:, 0] = np.random.uniform(0, 80, num_points)     # x
        points[:, 1] = np.random.uniform(-30, 30, num_points)    # y
        points[:, 2] = np.random.uniform(-2, 3, num_points)      # z
        points[:, 3] = np.random.uniform(0.1, 0.9, num_points)   # intensity

        # 模拟地面点 (低 z)
        ground_mask = np.random.rand(num_points) < 0.3
        points[ground_mask, 2] = np.random.uniform(-0.1, 0.1, ground_mask.sum())

        # 模拟前方车辆 (x~30m, y~0m)
        vehicle_mask = (np.abs(points[:, 0] - 30) < 2) & \
                       (np.abs(points[:, 1]) < 1) & \
                       (points[:, 2] > 0.3)
        # 通过增大特定区域点密度模拟
        extra_points = np.random.randn(500, 4)
        extra_points[:, 0] = extra_points[:, 0] * 1.5 + 30
        extra_points[:, 1] = extra_points[:, 1] * 0.8
        extra_points[:, 2] = np.abs(extra_points[:, 2]) * 1.5
        extra_points[:, 3] = 0.7

        return np.vstack([points, extra_points])

    def generate_radar_adc(self):
        """生成模拟雷达 ADC 数据"""
        return np.random.randn(128, 256, 4) * 0.01

    def generate_gnss(self):
        """生成模拟 GNSS 观测"""
        from localization.gnss_localizer import GNSSObservation, GNSSFixType

        self._ego_x += self._ego_v * 0.01 * np.cos(self._ego_yaw)
        self._ego_y += self._ego_v * 0.01 * np.sin(self._ego_yaw)

        return GNSSObservation(
            timestamp=self._t,
            latitude=31.2304 + self._ego_y / 111320.0,
            longitude=121.4737 + self._ego_x / (111320.0 *
                                                  np.cos(np.radians(31.2304))),
            altitude=5.0,
            v_east=self._ego_v * np.sin(self._ego_yaw),
            v_north=self._ego_v * np.cos(self._ego_yaw),
            hdop=0.8,
            fix_type=GNSSFixType.RTK_FIXED,
            num_satellites=22,
        )

    def generate_imu(self):
        """生成模拟 IMU 测量"""
        from localization.imu_processor import IMUMeasurement

        return IMUMeasurement(
            timestamp=self._t,
            accel_x=0.0,
            accel_y=0.0,
            accel_z=9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
        )

    def step(self, dt: float = 0.01):
        """推进模拟时间"""
        self._t += dt
        self._ego_x += self._ego_v * dt * np.cos(self._ego_yaw)
        self._ego_y += self._ego_v * dt * np.sin(self._ego_yaw)


# ============================================================
# 模拟模式
# ============================================================
def run_simulation_mode(duration_s: float = 60.0):
    """
    模拟模式: 使用模拟传感器数据运行完整自动驾驶管线
    """
    global g_system

    print("=" * 60)
    print("  L4 Autonomous Driving System — Simulation Mode")
    print("=" * 60)

    # 初始化系统
    vehicle_cfg = VehicleConfig()
    sys_cfg = SystemConfig()

    g_system = AutonomousDrivingSystem(vehicle_cfg, sys_cfg)

    # 启动传感器模拟器
    sim = SimulatedSensors()

    # 启动系统
    g_system.start()

    # 主循环: 注入传感器数据
    start_time = time.time()
    iteration = 0

    try:
        while time.time() - start_time < duration_s:
            dt = 0.01
            sim.step(dt)

            # 生成并注入传感器数据
            g_system.feed_sensor_data('camera_images',
                                      sim.generate_camera_images())
            g_system.feed_sensor_data('lidar_pointcloud',
                                      sim.generate_lidar_pointcloud())
            g_system.feed_sensor_data('radar_adc',
                                      sim.generate_radar_adc())
            g_system.feed_sensor_data('gnss_observation',
                                      sim.generate_gnss())
            g_system.feed_sensor_data('imu_measurement',
                                      sim.generate_imu())
            g_system.feed_sensor_data('wheel_speed', sim._ego_v)

            # 每秒输出状态
            iteration += 1
            if iteration % 100 == 0:
                output = g_system.get_system_output()
                health = g_system.get_health()
                print(f"\n[{time.time() - start_time:.1f}s] "
                      f"State={output['state'].name}, "
                      f"Pos=({output['localization']['x']:.1f}, "
                      f"{output['localization']['y']:.1f}), "
                      f"Tracked={output['perception']['num_tracked']}, "
                      f"Behavior={output['planning']['behavior']}, "
                      f"Safety={output['safety']['level']} "
                      f"({output['safety']['score']:.2f})")

            time.sleep(0.005)  # 降低 CPU 占用

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")

    finally:
        g_system.stop()

    # 输出最终健康报告
    final_health = g_system.get_health()
    print("\n" + "=" * 60)
    print("  Final System Health Report")
    print("=" * 60)
    print(f"  State:       {final_health.state.name}")
    print(f"  Modules:     {final_health.module_status}")
    print(f"  Latencies:   {final_health.latencies}")
    print(f"  Errors:      {final_health.errors}")
    print("=" * 60)


# ============================================================
# 回放模式
# ============================================================
def run_replay_mode(filepath: str):
    """
    回放模式: 从记录文件中回放传感器数据
    """
    global g_system
    import json

    print(f"[Main] Loading replay file: {filepath}")

    with open(filepath, 'r') as f:
        replay_data = json.load(f)

    g_system = AutonomousDrivingSystem()
    g_system.start()

    try:
        for frame in replay_data:
            sensor_data = frame['data']
            for sensor_type, data in sensor_data.items():
                g_system.feed_sensor_data(sensor_type, data)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Main] Replay interrupted.")

    finally:
        g_system.stop()


# ============================================================
# 交互演示模式
# ============================================================
def run_interactive_demo():
    """
    交互演示: 逐步展示各模块输出
    """
    global g_system

    print("=" * 60)
    print("  L4 Autonomous Driving System — Interactive Demo")
    print("=" * 60)
    print("  Demonstrating each module's output...")
    print("=" * 60)

    vehicle_cfg = VehicleConfig()
    sys_cfg = SystemConfig()

    g_system = AutonomousDrivingSystem(vehicle_cfg, sys_cfg)
    sim = SimulatedSensors()

    g_system.start()

    print("\n[Demo] System started. Press Ctrl+C to stop.")
    print("[Demo] Watching system output...\n")

    try:
        for step in range(500):
            sim.step(0.01)

            g_system.feed_sensor_data('camera_images',
                                      sim.generate_camera_images())
            g_system.feed_sensor_data('lidar_pointcloud',
                                      sim.generate_lidar_pointcloud())
            g_system.feed_sensor_data('radar_adc',
                                      sim.generate_radar_adc())
            g_system.feed_sensor_data('gnss_observation',
                                      sim.generate_gnss())
            g_system.feed_sensor_data('imu_measurement',
                                      sim.generate_imu())
            g_system.feed_sensor_data('wheel_speed', sim._ego_v)

            if step % 200 == 0:
                output = g_system.get_system_output()
                print(f"\n--- Step {step} ---")
                print(f"  System State:     {output['state'].name}")
                print(f"  Vehicle Position: ({output['localization']['x']:.2f}, "
                      f"{output['localization']['y']:.2f})")
                print(f"  Tracked Objects:  {output['perception']['num_tracked']}")
                print(f"  Behavior:         {output['planning']['behavior']}")
                print(f"  Safety Level:     {output['safety']['level']} "
                      f"(score={output['safety']['score']:.3f})")

                if output['control']:
                    ctrl = output['control']
                    cmd = ctrl.get('command')
                    if cmd:
                        print(f"  Steer: {cmd.steering_angle:.3f} rad, "
                              f"Throttle: {cmd.throttle:.3f}, "
                              f"Brake: {cmd.brake:.3f}")

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[Demo] Stopped by user.")

    finally:
        g_system.stop()

    print("\n[Demo] Session complete.")


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="L4 Autonomous Driving System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Interactive demo mode
  python main.py --sim           # Full simulation (60s)
  python main.py --sim --dur 120 # Simulation for 120 seconds
  python main.py --replay data.json  # Replay recorded data
        """,
    )

    parser.add_argument('--sim', action='store_true',
                        help='Run in simulation mode')
    parser.add_argument('--dur', type=float, default=60.0,
                        help='Simulation duration in seconds')
    parser.add_argument('--replay', type=str, metavar='FILE',
                        help='Replay recorded sensor data')
    parser.add_argument('--export-config', action='store_true',
                        help='Export default configuration and exit')

    args = parser.parse_args()

    # 信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.export_config:
        import json
        from dataclasses import asdict
        config = {
            'vehicle': asdict(VehicleConfig()),
            'system': asdict(SystemConfig()),
        }
        with open('config_export.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print("[Main] Configuration exported to config_export.json")
        return

    if args.replay:
        run_replay_mode(args.replay)
    elif args.sim:
        run_simulation_mode(args.dur)
    else:
        run_interactive_demo()


if __name__ == '__main__':
    main()
