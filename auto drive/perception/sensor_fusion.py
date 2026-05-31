"""
多传感器融合模块 —— 时空对齐、异构传感器数据融合
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from numpy.typing import NDArray


@dataclass
class FusedObject:
    """融合后的目标对象 (所有传感器信息)"""
    object_id: int
    object_class: str
    confidence: float

    # 位置 & 运动
    x: float; y: float; z: float
    vx: float; vy: float
    ax: float; ay: float
    yaw: float; yaw_rate: float

    # 几何
    length: float; width: float; height: float

    # 传感器来源
    seen_by_lidar: bool = False
    seen_by_camera: bool = False
    seen_by_radar: bool = False

    # 协方差
    covariance: Optional[NDArray] = None

    # 时间戳
    timestamp: float = 0.0


class SensorFusion:
    """
    多传感器时空融合:
      - 时间戳同步 (线性插值)
      - 坐标系变换 (各传感器 → 车辆坐标系)
      - 传感器权重融合 (卡尔曼滤波 / 协方差交叉)
      - 传感器退化检测与降级策略
    """

    def __init__(self):
        # 各传感器信任权重
        self._sensor_weights = {
            'lidar_position': 0.8,       # LiDAR 定位精度最高
            'radar_velocity': 0.9,       # 雷达速度最准
            'camera_classification': 0.85,  # 视觉分类最好
        }

        # 传感器状态
        self._sensor_health = {
            'lidar': True,
            'camera': True,
            'radar': True,
            'gnss': True,
            'imu': True,
        }

    def _temporal_align(self, lidar_detections: List,
                        camera_detections: List,
                        radar_tracks: List,
                        lidar_ts: float,
                        camera_ts: float,
                        radar_ts: float) -> Tuple[List, List, List]:
        """
        时间戳对齐: 将所有传感器检测外推到统一时间戳 (取最新)
        """
        target_ts = max(lidar_ts, camera_ts, radar_ts)

        # LiDAR 检测外推
        dt_lidar = target_ts - lidar_ts
        if abs(dt_lidar) > 0.1:  # 100ms 以上视为过期
            lidar_detections = []
        else:
            for det in lidar_detections:
                det.center[0] += det.velocity[0] * dt_lidar
                det.center[1] += det.velocity[1] * dt_lidar

        # 雷达航迹外推
        for track in radar_tracks:
            track.x += track.vx * (target_ts - radar_ts)
            track.y += track.vy * (target_ts - radar_ts)

        return lidar_detections, camera_detections, radar_tracks

    def _coordinate_transform(self, detections: List,
                              sensor_extrinsic: Dict,
                              source: str) -> List:
        """
        将不同传感器的检测变换到统一的车辆坐标系
        """
        R = sensor_extrinsic.get('rotation', np.eye(3))
        t = sensor_extrinsic.get('translation', np.zeros(3))

        for det in detections:
            if hasattr(det, 'center'):
                det.center = R @ det.center + t
            if hasattr(det, 'velocity') and len(det.velocity) >= 2:
                det.velocity[:2] = (R[:2, :2] @ det.velocity[:2].reshape(2, 1)).flatten()

        return detections

    def _associate_and_fuse(self, lidar_dets: List,
                            camera_dets: List,
                            radar_tracks: List) -> List[FusedObject]:
        """
        多源目标关联与融合
        策略: LiDAR 主导定位, 雷达主导速度, 相机主导分类
        """
        fused_objects = []
        matched_radar = set()

        # 以 LiDAR 检测为锚点
        for i, ldet in enumerate(lidar_dets):
            center = ldet.center[:2]

            fused = FusedObject(
                object_id=i,
                object_class=ldet.object_class.value,
                confidence=ldet.confidence,
                x=center[0], y=center[1], z=ldet.center[2],
                vx=ldet.velocity[0], vy=ldet.velocity[1],
                ax=0.0, ay=0.0,
                yaw=ldet.yaw, yaw_rate=0.0,
                length=ldet.dimensions[0],
                width=ldet.dimensions[1],
                height=ldet.dimensions[2],
                seen_by_lidar=True,
            )

            # 融合雷达速度 (雷达多普勒精度高)
            for j, rtrack in enumerate(radar_tracks):
                rcenter = np.array([rtrack.x, rtrack.y])
                dist = np.linalg.norm(center - rcenter)

                if dist < 3.0 and j not in matched_radar:
                    alpha = self._sensor_weights['radar_velocity']
                    fused.vx = (1 - alpha) * fused.vx + alpha * rtrack.vx
                    fused.vy = (1 - alpha) * fused.vy + alpha * rtrack.vy
                    fused.seen_by_radar = True
                    matched_radar.add(j)
                    break

            # 融合相机分类信息 (如果存在)
            if camera_dets:
                for cdet in camera_dets:
                    ccenter = getattr(cdet, 'center',
                                      np.zeros(3))[:2]
                    if np.linalg.norm(center - ccenter) < 2.0:
                        if cdet.confidence > fused.confidence:
                            fused.object_class = cdet.object_class.value
                            fused.confidence = cdet.confidence
                        fused.seen_by_camera = True
                        break

            fused_objects.append(fused)

        # 处理纯雷达目标 (LiDAR 漏检的远距目标)
        for j, rtrack in enumerate(radar_tracks):
            if j not in matched_radar:
                fused_objects.append(FusedObject(
                    object_id=len(fused_objects),
                    object_class='unknown',
                    confidence=0.5,
                    x=rtrack.x, y=rtrack.y, z=0.0,
                    vx=rtrack.vx, vy=rtrack.vy,
                    ax=0.0, ay=0.0,
                    yaw=np.arctan2(rtrack.vy, rtrack.vx),
                    yaw_rate=0.0,
                    length=4.5, width=1.8, height=1.5,
                    seen_by_radar=True,
                ))

        return fused_objects

    def fuse(self,
             lidar_detections: List,
             camera_detections: List,
             radar_tracks: List,
             sensor_timestamps: Dict[str, float],
             sensor_extrinsics: Dict[str, Dict]) -> Dict:
        """
        主融合入口
        """
        # Step 1: 时间对齐
        lidar_dets, camera_dets, radar_trks = self._temporal_align(
            lidar_detections, camera_detections, radar_tracks,
            sensor_timestamps.get('lidar', 0),
            sensor_timestamps.get('camera', 0),
            sensor_timestamps.get('radar', 0),
        )

        # Step 2: 空间对齐 (坐标系变换)
        lidar_dets = self._coordinate_transform(
            lidar_dets, sensor_extrinsics.get('lidar', {}), 'lidar'
        )
        radar_trks = self._coordinate_transform(
            radar_trks, sensor_extrinsics.get('radar', {}), 'radar'
        )

        # Step 3: 目标关联与异构融合
        fused = self._associate_and_fuse(lidar_dets, camera_dets, radar_trks)

        # Step 4: 传感器退化检测
        degradation = self._check_sensor_degradation(
            lidar_dets, camera_dets, radar_trks
        )

        return {
            "fused_objects": fused,
            "num_objects": len(fused),
            "sensor_degradation": degradation,
            "sensor_health": self._sensor_health.copy(),
        }

    def _check_sensor_degradation(self, lidar_dets, camera_dets,
                                  radar_trks) -> Dict[str, bool]:
        """
        检测传感器退化状态 (用于触发降级策略)
        """
        degradation = {}

        # LiDAR 退化: 检测数为 0 但雷达有目标
        if len(lidar_dets) == 0 and len(radar_trks) > 3:
            degradation['lidar_degraded'] = True

        # 相机退化: 场景整体亮度过低 (需上游传入)
        degradation['camera_degraded'] = False

        # 雷达退化: 静止目标丢失 (雷达对静止目标不敏感)
        degradation['radar_degraded'] = False

        return degradation

    def set_sensor_weight(self, sensor: str, weight: float):
        """动态调整传感器信任权重"""
        key = f'{sensor}_position' if sensor in ['lidar', 'camera'] else f'{sensor}_velocity'
        if key in self._sensor_weights:
            self._sensor_weights[key] = np.clip(weight, 0.1, 1.0)

    def set_sensor_health(self, sensor: str, healthy: bool):
        """标记传感器健康状态"""
        if sensor in self._sensor_health:
            self._sensor_health[sensor] = healthy
