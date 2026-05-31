"""
3D 目标检测 —— 前融合 (PointPainting 范式) 多模态目标检测
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from numpy.typing import NDArray


class ObjectClass(Enum):
    """检测目标类别"""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    MOTORCYCLE = "motorcycle"
    TRAFFIC_CONE = "traffic_cone"
    TRAFFIC_SIGN = "traffic_sign"
    UNKNOWN = "unknown"


@dataclass
class Detection3D:
    """3D 目标检测框"""
    object_class: ObjectClass
    confidence: float
    # 3D 边界框 (车辆坐标系)
    center: NDArray         # (3,) x, y, z
    dimensions: NDArray     # (3,) length, width, height
    yaw: float              # 朝向角 [rad]
    # 运动状态
    velocity: NDArray = field(default_factory=lambda: np.zeros(2))  # (vx, vy)
    # 跟踪 ID (下游填充)
    track_id: int = -1
    # 协方差 (不确定性)
    covariance: Optional[NDArray] = None  # (7,7) [x,y,z,l,w,h,yaw]

    @property
    def corners(self) -> NDArray:
        """计算 3D 框的 8 个角点 (车辆坐标系)"""
        l, w, h = self.dimensions
        x, y, z = self.center
        yaw = self.yaw

        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        corners_local = np.array([
            [l / 2, w / 2, h / 2],
            [l / 2, w / 2, -h / 2],
            [l / 2, -w / 2, h / 2],
            [l / 2, -w / 2, -h / 2],
            [-l / 2, w / 2, h / 2],
            [-l / 2, w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2],
            [-l / 2, -w / 2, -h / 2],
        ])

        return corners_local @ R.T + np.array([x, y, z])


class ObjectDetector:
    """
    多模态 3D 目标检测器 (PointPainting + CenterPoint 范式)

    实际部署替换为:
      - VoxelNet / SECOND (LiDAR backbone)
      - DETR3D / BEVFormer (Camera backbone)
      - TransFusion (融合 backbone)
      加载 ONNX / TensorRT 模型推理

    这里提供完整的推理框架结构
    """

    CLASS_MAP = {
        0: ObjectClass.CAR,
        1: ObjectClass.TRUCK,
        2: ObjectClass.BUS,
        3: ObjectClass.PEDESTRIAN,
        4: ObjectClass.CYCLIST,
        5: ObjectClass.MOTORCYCLE,
        6: ObjectClass.TRAFFIC_CONE,
    }

    def __init__(self, confidence_threshold: float = 0.65,
                 nms_iou_threshold: float = 0.01,  # 3D NMS 使用距离阈值
                 max_detections: int = 200):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

    def _point_painting_fusion(self, lidar_points: NDArray,
                               camera_bev: NDArray) -> NDArray:
        """
        PointPainting: 将相机语义信息投影到 LiDAR 点云上增强特征
        每个 LiDAR 点附加 BEV 语义通道
        """
        if len(lidar_points) == 0:
            return lidar_points

        # 将 LiDAR 点投影到 BEV 图像坐标系
        bev_h, bev_w = camera_bev.shape[:2] if camera_bev.ndim == 3 else (512, 512)
        resolution = 0.1  # m/pixel

        # LiDAR 坐标 → BEV 像素坐标
        # BEV 中心对应车辆坐标系原点
        pixel_u = (lidar_points[:, 0] / resolution + bev_w / 2).astype(int)
        pixel_v = (-lidar_points[:, 1] / resolution + bev_h / 2).astype(int)

        valid = (pixel_u >= 0) & (pixel_u < bev_w) & \
                (pixel_v >= 0) & (pixel_v < bev_h)

        # 附加 BEV 语义特征
        if camera_bev.ndim == 3:
            num_features = lidar_points.shape[-1] + 3  # + RGB
        else:
            num_features = lidar_points.shape[-1] + 1

        augmented_points = np.zeros((len(lidar_points), num_features))
        augmented_points[:, :lidar_points.shape[-1]] = lidar_points

        if valid.any():
            if camera_bev.ndim == 3:
                augmented_points[valid, lidar_points.shape[-1]:] = \
                    camera_bev[pixel_v[valid], pixel_u[valid]]
            else:
                augmented_points[valid, lidar_points.shape[-1]] = \
                    camera_bev[pixel_v[valid], pixel_u[valid]]

        return augmented_points

    def _generate_proposals_from_clusters(self, lidar_clusters: List) -> List[Detection3D]:
        """
        将 LiDAR 聚类结果转换为初步 3D 检测框
        实际部署中替换为模型推理
        """
        proposals = []

        for cluster in lidar_clusters:
            # 尺寸分类启发式
            l, w, h = cluster.dimensions
            if l > 8 or w > 3:
                obj_class = ObjectClass.TRUCK
            elif l > 5 or w > 2.2:
                obj_class = ObjectClass.BUS
            elif l > 3:
                obj_class = ObjectClass.CAR
            elif h > 1.5 and l < 1.5:
                obj_class = ObjectClass.PEDESTRIAN
            elif l < 2 and w < 1.2:
                obj_class = ObjectClass.CYCLIST
            else:
                obj_class = ObjectClass.UNKNOWN

            proposals.append(Detection3D(
                object_class=obj_class,
                confidence=cluster.confidence,
                center=cluster.centroid,
                dimensions=cluster.dimensions,
                yaw=cluster.yaw,
            ))

        return proposals

    def _classify_detections(self, proposals: List[Detection3D],
                             camera_features: Dict) -> List[Detection3D]:
        """
        融合相机特征对检测框进行精细分类
        实际部署中替换为 LiDAR-Camera 融合分类头
        """
        classified = []

        for det in proposals:
            # 将目标类别未知的用启发式规则重新分类
            if det.object_class == ObjectClass.UNKNOWN:
                l, w, h = det.dimensions
                if h < 0.8 and max(l, w) < 0.5:
                    det.object_class = ObjectClass.TRAFFIC_CONE
                    det.confidence *= 0.7
                else:
                    det.confidence *= 0.5  # 降低未知目标置信度

            if det.confidence >= self.confidence_threshold:
                classified.append(det)

        return classified

    def _nms_3d(self, detections: List[Detection3D]) -> List[Detection3D]:
        """
        3D 非极大值抑制 (基于BEV中心距离)
        """
        if len(detections) <= 1:
            return detections

        # 按置信度降序排列
        sorted_idx = np.argsort([-d.confidence for d in detections])
        keep = []

        for i, idx_i in enumerate(sorted_idx):
            if detections[idx_i] is None:
                continue

            keep.append(detections[idx_i])

            for j_idx, idx_j in enumerate(sorted_idx):
                if i == j_idx or detections[idx_j] is None:
                    continue

                # BEV 中心距离
                ci = detections[idx_i].center[:2]
                cj = detections[idx_j].center[:2]
                dist = np.linalg.norm(ci - cj)

                # IoU 近似: 如果同一类别且距离 < 阈值则抑制
                if (detections[idx_i].object_class == detections[idx_j].object_class and
                        dist < 2.0):  # 2m 阈值
                    detections[idx_j] = None

        return keep[:self.max_detections]

    def detect(self, lidar_pointcloud: NDArray,
               lidar_clusters: List,
               camera_bev: NDArray,
               camera_features: Dict,
               radar_tracks: List) -> List[Detection3D]:
        """
        多传感器融合目标检测主流程
        """
        # Step 1: PointPainting 融合
        augmented_pc = self._point_painting_fusion(lidar_pointcloud, camera_bev)

        # Step 2: 基于 LiDAR 聚类的提案生成
        proposals = self._generate_proposals_from_clusters(lidar_clusters)

        # Step 3: 融合雷达跟踪补充远距目标
        for track in radar_tracks:
            # 检查是否已有 LiDAR 目标覆盖此区域
            covered = False
            for p in proposals:
                dist = np.linalg.norm(p.center[:2] - np.array([track.x, track.y]))
                if dist < 2.5:
                    covered = True
                    # 融合雷达速度
                    p.velocity = np.array([track.vx, track.vy])
                    break

            if not covered:
                # 添加纯雷达目标 (远距离, LiDAR 可能漏检)
                proposals.append(Detection3D(
                    object_class=ObjectClass.CAR,
                    confidence=0.55,  # 纯雷达置信度较低
                    center=np.array([track.x, track.y, 0.0]),
                    dimensions=np.array([4.5, 1.8, 1.5]),
                    yaw=np.arctan2(track.vy, track.vx) if abs(track.vx) > 0.1 else 0.0,
                    velocity=np.array([track.vx, track.vy]),
                ))

        # Step 4: 精细分类
        detections = self._classify_detections(proposals, camera_features)

        # Step 5: 3D NMS
        detections = self._nms_3d(detections)

        return detections
