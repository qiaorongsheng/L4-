# L4 Autonomous Driving System

A comprehensive L4-level autonomous driving software stack covering **perception**, **localization**, **prediction**, **planning**, **control**, **safety**, and **HD maps**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SAFETY MONITOR (50 Hz)                     │
│   TTC Check │ Lateral Check │ Sensor Health │ ODD Compliance    │
│                  Redundancy Checker │ MRM Handler               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │PERCEPTION│   │LOCALIZAT.│   │PREDICTION│   │ PLANNING │    │
│  │  (30 Hz) │   │ (100 Hz) │   │ (20 Hz)  │   │ (10 Hz)  │    │
│  │          │   │          │   │          │   │          │    │
│  │ Camera   │   │ GNSS     │   │ Behavior │   │ Route    │    │
│  │ LiDAR    │   │ IMU      │   │ Traject. │   │ Behavior │    │
│  │ Radar    │   │ MapMatch │   │ Interact │   │ Motion   │    │
│  │ Fusion   │   │ EKF      │   │          │   │ Optimize │    │
│  │ Tracking │   │          │   │          │   │          │    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘    │
│       │              │              │              │           │
│       └──────────────┴──────────────┴──────────────┘           │
│                              │                                   │
│                       ┌──────┴──────┐                           │
│                       │   CONTROL   │                           │
│                       │  (100 Hz)   │                           │
│                       │             │                           │
│                       │ MPC/Stanley │                           │
│                       │ Lateral PID │                           │
│                       │ Longitud.PID│                           │
│                       │ Veh.Iface   │                           │
│                       └──────┬──────┘                           │
│                              │                                   │
│                       ┌──────┴──────┐                           │
│                       │   VEHICLE   │                           │
│                       │ Steer/Throt │                           │
│                       │ /Brake/Gear │                           │
│                       └─────────────┘                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                       HD MAP                              │   │
│  │    Lane Topology │ Spatial Index │ Local Map Query        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Module Details

### Perception (`perception/`)
| File | Description | Key Algorithms |
|------|-------------|----------------|
| `camera_processor.py` | Multi-camera BEV projection, undistortion, feature extraction | IPM, Multi-cam stitching |
| `lidar_processor.py` | Point cloud ground segmentation, clustering, 3D bbox fitting | RANSAC, DBSCAN, PCA |
| `radar_processor.py` | Range-Doppler map, CFAR detection, Kalman tracking | 2D-FFT, CA-CFAR, EKF |
| `object_detector.py` | Multi-modal 3D detection (PointPainting paradigm) | PointPainting, NMS |
| `object_tracker.py` | 3D multi-object tracking | EKF-CTRV, Hungarian match |
| `lane_detector.py` | Lane marking detection & polynomial fitting | RANSAC poly-fit |
| `sensor_fusion.py` | Spatio-temporal sensor fusion, degradation detection | Covariance intersection |

### Localization (`localization/`)
| File | Description | Key Algorithms |
|------|-------------|----------------|
| `gnss_localizer.py` | RTK-GNSS processing, WGS84↔ENU conversion | pyproj transforms |
| `imu_processor.py` | Strapdown INS propagation, ZUPT detection | Quaternion integration |
| `map_matcher.py` | LiDAR-to-HD Map registration | NDT + ICP (multi-resolution) |
| `ekf_fusion.py` | 15-state Error-State EKF sensor fusion | ES-EKF |

### Prediction (`prediction/`)
| File | Description | Key Algorithms |
|------|-------------|----------------|
| `behavior_predictor.py` | Intent classification, lane change prediction | HMM, rule-based inference |
| `trajectory_predictor.py` | Multi-modal trajectory prediction | Polynomial extrapolation, lane-constrained |
| `interaction_model.py` | Multi-agent interaction graph, game theory | Stackelberg game, social attention |

### Planning (`planning/`)
| File | Description | Key Algorithms |
|------|-------------|----------------|
| `route_planner.py` | Lane-level global routing | A\*, Yen's K-shortest paths |
| `behavior_planner.py` | Hierarchical FSM driving decisions | HFSM, rule-based cost |
| `motion_planner.py` | Frenet-frame lattice trajectory generation | Polynomial sampling, collision check |
| `trajectory_optimizer.py` | Nonlinear trajectory optimization | CasADi IPOPT, MPC |

### Control (`control/`)
| File | Description | Key Algorithms |
|------|-------------|----------------|
| `lateral_controller.py` | Steering control | Stanley, Pure Pursuit, LQR |
| `longitudinal_controller.py` | Speed/distance control | Cascade PID, ACC (CTH) |
| `mpc_controller.py` | Coupled lateral+longitudinal MPC | CasADi IPOPT, RTI |
| `vehicle_interface.py` | CAN bus abstraction, command validation | Watchdog, smoothing |

### Safety (`safety/`)
| File | Description | Key Algorithms |
|------|-------------|----------------|
| `__init__.py` | Multi-layer safety shell, ODD compliance | FTA, MRM trigger logic |
| `redundancy_checker.py` | Cross-sensor/cross-module consistency | Multi-source agreement |
| `emergency_handler.py` | Minimal Risk Maneuver execution | Safe stop, pull-over, slow-down |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Interactive demo mode (default)
python main.py

# Full simulation (60 seconds)
python main.py --sim

# Simulation for 120 seconds
python main.py --sim --dur 120

# Export default configuration
python main.py --export-config
```

## Configuration

Edit [config/vehicle_config.py](config/vehicle_config.py) for vehicle physics and [config/system_config.py](config/system_config.py) for operational parameters.

Key tunables:
- `perception_hz` / `planning_hz` / `control_hz` — module frequencies
- `safe_following_time` — ACC time gap (default 2.0s)
- `lane_change_min_gap` — minimum gap for lane change (default 25m)
- `emergency_brake_ttc` — TTC threshold for emergency braking (default 2.5s)

## L4 Key Safety Features

1. **Multi-sensor fusion** — LiDAR + Camera + Radar with redundancy
2. **Error-State EKF** — 15-state localization with GNSS/IMU/Wheel/Map
3. **Multi-modal prediction** — 4 modalities per tracked object
4. **Safety shell** — Continuous TTC/lateral/speed/ODD monitoring
5. **Minimal Risk Maneuvers** — Safe stop, pull-over, handover
6. **Redundancy checking** — Cross-validation of all sensor/planning outputs
7. **Watchdog** — Command timeout triggers automatic emergency brake
8. **Degradation management** — Graceful fallback on sensor/module failures

📊 项目概况
维度	数据
总文件数	37 个 Python 文件 + README + requirements.txt
总代码量	8,571 行
目录结构	8 个模块目录
🏗️ 系统架构

auto_drive/
├── main.py                          ← 主入口 (交互/模拟/回放 三种模式)
├── config/          (2 文件)         ← 车辆参数 + 系统运行参数
├── perception/      (7 文件)         ← 多传感器融合感知
│   ├── camera_processor.py          ← 多相机 BEV 投影、IPM、车道线检测
│   ├── lidar_processor.py           ← RANSAC 地面分割、DBSCAN 聚类、PCA BBox
│   ├── radar_processor.py           ← 2D-FFT、CFAR、DOA 估计、卡尔曼跟踪
│   ├── sensor_fusion.py             ← 时空对齐、3 传感器融合、退化检测
│   ├── object_detector.py           ← PointPainting 范式多模态检测
│   ├── object_tracker.py            ← CTRV-EKF + 匈牙利匹配 3D 跟踪
│   └── lane_detector.py             ← RANSAC 多项式车道线拟合
├── localization/    (4 文件)         ← 多源融合定位
│   ├── gnss_localizer.py            ← RTK-GNSS 解算、WGS84↔ENU
│   ├── imu_processor.py             ← 捷联惯导、四元数积分、ZUPT 零速检测
│   ├── map_matcher.py               ← NDT + ICP 多分辨率配准
│   └── ekf_fusion.py                ← 15维 误差状态 EKF (GNSS+IMU+轮速+地图)
├── prediction/      (3 文件)         ← 意图+轨迹+交互建模
│   ├── behavior_predictor.py        ← HMM 行为推理、变道意图、让行检测
│   ├── trajectory_predictor.py      ← 4模态轨迹生成 (keep/left/right/stop)
│   └── interaction_model.py         ← 交互图构建、Stackelberg 博弈求解
├── planning/        (4 文件)         ← 全局→行为→运动→优化
│   ├── route_planner.py             ← A* + Yen's K-最短路径车道级路由
│   ├── behavior_planner.py          ← 17态 HFSM 行为决策 + 安全门控
│   ├── motion_planner.py            ← Frenet 坐标系 Lattice 轨迹采样
│   └── trajectory_optimizer.py      ← CasADi IPOPT 非线性 MPC 优化
├── control/         (4 文件)         ← 横纵向解耦 + 耦合 MPC
│   ├── lateral_controller.py        ← Stanley / Pure Pursuit / LQR
│   ├── longitudinal_controller.py   ← 级联 PID + 前馈补偿 + ACC(CTH)
│   ├── mpc_controller.py            ← 运动学模型 MPC (CasADi 实时迭代)
│   └── vehicle_interface.py         ← CAN 抽象、指令验证、看门狗
├── safety/          (3 文件)         ← 多层安全壳
│   ├── __init__.py                   ← 7项安全检查、ODD合规、MRM触发
│   ├── redundancy_checker.py        ← 定位/速度/感知/规划4路冗余交叉验证
│   └── emergency_handler.py         ← Safe Stop/Pull Over/Slow Down/Handover
├── hd_map/          (2 文件)         ← HD Map 管理
│   ├── __init__.py                   ← 地图加载、R-Tree 查询、局部地图提取
│   └── lane_graph.py                ← 车道拓扑图、A* 路径搜索
├── system/
│   └── __init__.py                   ← 系统编排器 (6线程多频调度 + 数据记录)
└── requirements.txt
🔑 L4 关键特性
特性	实现
多传感器融合	LiDAR+Camera+Radar 三模态 PointPainting 融合
厘米级定位	15维 ES-EKF (GNSS/IMU/轮速/地图匹配)
多模态预测	每目标 4 条轨迹预测 + 交互建模
多层安全	TTC/横向/速度/定位/ODD 7项连续检查 → MRM
冗余验证	定位/速度/感知/规划 交叉一致性检验
紧急处理	Safe Stop / Pull Over / Slow Down / Handover
ODD 管理	天气/光照/能见度合规性检查
看门狗	控制指令超时自动紧急制动
🚀 运行

pip install -r requirements.txt
python main.py              # 交互演示
python main.py --sim        # 完整模拟 (60s)
python main.py --sim --dur 120  # 自定义时长
注意: 实际 L4 部署需替换各模块中的深度学习模型（3D 检测/分割/预测 Transformer），并接入真实传感器驱动与 CAN 总线。本代码提供了完整的算法框架、数学实现和系统架构，可作为研发基线。
