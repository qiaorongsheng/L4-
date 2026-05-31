"""
LiDAR 点云处理模块 —— 地面分割、聚类、3D目标检测
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from numpy.typing import NDArray


@dataclass
class LidarPoint:
    """单点 LiDAR 数据"""
    x: float
    y: float
    z: float
    intensity: float = 0.0
    ring: int = 0              # 激光线号


@dataclass
class LidarCluster:
    """点云聚类结果"""
    cluster_id: int
    centroid: NDArray         # (3,) 质心
    dimensions: NDArray       # (3,) 长宽高
    yaw: float                # 朝向角
    points: NDArray           # (N, 3) 点集
    confidence: float = 1.0


class LidarProcessor:
    """
    LiDAR 点云处理管线:
      - ROI 裁剪 & 离群点去除
      - 地面分割 (RANSAC / 射线法)
      - 非地面点欧式聚类 (DBSCAN)
      - 3D BBox 拟合 (PCA + L-Shape)
    """

    def __init__(self, sensor_height: float = 1.75,
                 ground_threshold: float = 0.2,
                 cluster_eps: float = 0.8,
                 cluster_min_points: int = 15):
        self.sensor_height = sensor_height
        self.ground_threshold = ground_threshold
        self.cluster_eps = cluster_eps
        self.cluster_min_points = cluster_min_points

    def preprocess(self, pointcloud: NDArray) -> NDArray:
        """
        预处理: ROI 裁剪 + 统计离群点过滤
        pointcloud: (N, 4) [x, y, z, intensity]
        """
        # ROI 范围 (车体坐标系, 前向为 x)
        roi_mask = (
            (pointcloud[:, 0] > -5) & (pointcloud[:, 0] < 150) &   # 纵向
            (pointcloud[:, 1] > -30) & (pointcloud[:, 1] < 30) &   # 横向
            (pointcloud[:, 2] > -3) & (pointcloud[:, 2] < 5)       # 高度
        )
        pc = pointcloud[roi_mask]

        # 统计离群点去除
        if len(pc) > 0:
            mean = pc[:, :3].mean(axis=0)
            std = pc[:, :3].std(axis=0)
            inlier_mask = np.all(np.abs(pc[:, :3] - mean) < 3 * std, axis=1)
            pc = pc[inlier_mask]

        return pc

    def segment_ground_ransac(self, pointcloud: NDArray) -> Tuple[NDArray, NDArray]:
        """
        基于 RANSAC 的地面分割
        返回: (ground_points, non_ground_points)
        """
        xyz = pointcloud[:, :3]
        n = len(xyz)

        if n < 3:
            return (np.array([]).reshape(0, 4),
                    pointcloud)

        best_inliers = []
        best_plane = None
        max_inliers = 0
        max_iterations = min(200, n // 3)

        for _ in range(max_iterations):
            # 随机采样 3 个点
            samples = xyz[np.random.choice(n, 3, replace=False)]
            p1, p2, p3 = samples
            normal = np.cross(p2 - p1, p3 - p1)
            if np.linalg.norm(normal) < 1e-6:
                continue
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p1)

            # 计算所有点到平面的距离
            distances = np.abs(xyz @ normal + d)
            inliers = np.where(distances < self.ground_threshold)[0]

            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_inliers = inliers
                best_plane = (normal, d)

        if best_plane is None:
            return (np.array([]).reshape(0, 4),
                    pointcloud)

        # 进一步约束: 地平面法向量应接近垂直
        normal, d = best_plane
        if np.abs(normal[2]) < 0.7:  # 法向量与竖轴夹角过大
            return (np.array([]).reshape(0, 4),
                    pointcloud)

        ground = pointcloud[best_inliers]
        non_ground_mask = np.ones(n, dtype=bool)
        non_ground_mask[best_inliers] = False
        non_ground = pointcloud[non_ground_mask]

        return ground, non_ground

    def cluster_objects(self, non_ground: NDArray) -> List[LidarCluster]:
        """
        对非地面点进行 DBSCAN 聚类
        """
        if len(non_ground) < self.cluster_min_points:
            return []

        xyz = non_ground[:, :3]

        # DBSCAN 聚类
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_points,
            metric='euclidean',
            n_jobs=-1
        ).fit(xyz)

        labels = clustering.labels_
        clusters = []

        for label in set(labels):
            if label == -1:  # 噪声点
                continue

            cluster_mask = labels == label
            cluster_pts = xyz[cluster_mask]

            if len(cluster_pts) < self.cluster_min_points:
                continue

            # PCA 估计朝向角 (yaw)
            centroid = cluster_pts.mean(axis=0)
            centered = cluster_pts[:, :2] - centroid[:2]  # 只用 x,y
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # 主方向 = 最大特征值对应的特征向量
            principal_axis = eigenvectors[:, -1]
            yaw = np.arctan2(principal_axis[1], principal_axis[0])

            # 3D 尺寸估计
            # 将点投影到主方向和垂直方向
            R = np.array([[np.cos(yaw), np.sin(yaw)],
                          [-np.sin(yaw), np.cos(yaw)]])
            projected = centered @ R.T
            dim_xy = projected.max(axis=0) - projected.min(axis=0)
            dim_z = cluster_pts[:, 2].max() - cluster_pts[:, 2].min()
            dimensions = np.array([max(dim_xy[0], 0.3),
                                   max(dim_xy[1], 0.3),
                                   max(dim_z, 0.3)])

            clusters.append(LidarCluster(
                cluster_id=label,
                centroid=centroid,
                dimensions=dimensions,
                yaw=yaw,
                points=cluster_pts,
                confidence=min(1.0, len(cluster_pts) / 100),
            ))

        return clusters

    def process(self, pointcloud: NDArray) -> Dict:
        """
        完整 LiDAR 处理管线
        pointcloud: (N, 4) [x, y, z, intensity]
        """
        # Step 1: 预处理
        pc_filtered = self.preprocess(pointcloud)

        # Step 2: 地面分割
        ground, non_ground = self.segment_ground_ransac(pc_filtered)

        # Step 3: 聚类
        clusters = self.cluster_objects(non_ground)

        return {
            "filtered_pointcloud": pc_filtered,
            "ground_points": ground,
            "non_ground_points": non_ground,
            "clusters": clusters,
            "num_clusters": len(clusters),
        }
