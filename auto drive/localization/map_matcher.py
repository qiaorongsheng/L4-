"""
高精地图匹配模块 —— 激光雷达点云 ↔ HD Map 配准 (NDT / ICP)
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from numpy.typing import NDArray


@dataclass
class MapMatchResult:
    """地图匹配结果"""
    x: float; y: float; z: float          # 匹配位置
    yaw: float; pitch: float; roll: float  # 匹配姿态
    match_score: float                     # 匹配分数 (0-1)
    num_correspondences: int                # 对应点对数量
    convergence: bool = True
    iterations: int = 0


class MapMatcher:
    """
    高精地图匹配:
      - NDT (Normal Distributions Transform) 配准
      - ICP (Iterative Closest Point) 精配准
      - 多分辨率配准 (粗→细)
      - 配准质量评估
    """

    def __init__(self, ndt_resolution: float = 1.0,
                 icp_max_iterations: int = 50,
                 icp_tolerance: float = 1e-4,
                 max_correspondence_distance: float = 5.0):
        self.ndt_resolution = ndt_resolution
        self.icp_max_iterations = icp_max_iterations
        self.icp_tolerance = icp_tolerance
        self.max_correspondence_distance = max_correspondence_distance

        # 预加载的地图数据
        self._map_pointcloud: Optional[NDArray] = None
        self._map_kdtree: Optional[KDTree] = None
        self._map_voxel_grid: Dict = {}

    def load_map(self, map_points: NDArray):
        """
        加载 HD 地图点云
        map_points: (N, 3+) [x, y, z, ...]
        """
        self._map_pointcloud = map_points
        self._map_kdtree = KDTree(map_points[:, :3])

        # 构建体素网格 (用于 NDT)
        self._build_voxel_grid(map_points)

    def _build_voxel_grid(self, points: NDArray):
        """构建体素网格, 每个体素计算高斯分布"""
        self._map_voxel_grid = {}

        if len(points) == 0:
            return

        min_coords = points[:, :3].min(axis=0)
        max_coords = points[:, :3].max(axis=0)

        grid_dims = np.ceil(
            (max_coords - min_coords) / self.ndt_resolution
        ).astype(int)

        for i, pt in enumerate(points):
            voxel_idx = tuple(
                np.floor((pt[:3] - min_coords) / self.ndt_resolution).astype(int)
            )
            if voxel_idx not in self._map_voxel_grid:
                self._map_voxel_grid[voxel_idx] = []
            self._map_voxel_grid[voxel_idx].append(i)

        # 为每个体素计算均值和协方差
        self._voxel_params = {}
        for voxel_idx, point_indices in self._map_voxel_grid.items():
            if len(point_indices) < 5:
                continue
            voxel_points = points[point_indices, :3]
            mean = voxel_points.mean(axis=0)
            cov = np.cov(voxel_points.T)
            # 正则化防止奇异
            cov += np.eye(3) * 1e-4
            self._voxel_params[voxel_idx] = (mean, cov)

    def ndt_score(self, transform: NDArray, scan_points: NDArray) -> float:
        """
        计算 NDT 匹配分数
        transform: (4, 4) 变换矩阵 (scan → map)
        """
        if not self._voxel_params:
            return float('inf')

        R = transform[:3, :3]
        t = transform[:3, 3]

        score = 0.0
        count = 0

        for pt in scan_points:
            # 变换到地图坐标系
            pt_transformed = R @ pt + t

            # 查找所属体素
            voxel_idx = tuple(np.floor(
                (pt_transformed - self._map_pointcloud[:, :3].min(axis=0)) /
                self.ndt_resolution
            ).astype(int))

            if voxel_idx in self._voxel_params:
                mean, cov = self._voxel_params[voxel_idx]
                d = pt_transformed - mean
                try:
                    inv_cov = np.linalg.inv(cov)
                    score += d @ inv_cov @ d
                    count += 1
                except np.linalg.LinAlgError:
                    continue

        return score / max(count, 1)

    def icp_align(self, source_points: NDArray,
                  initial_transform: Optional[NDArray] = None) -> MapMatchResult:
        """
        Iterative Closest Point (ICP) 精配准
        source_points: (N, 2) 或 (N, 3)
        initial_transform: 初始变换矩阵
        """
        if self._map_kdtree is None:
            raise ValueError("Map not loaded. Call load_map() first.")

        if initial_transform is None:
            initial_transform = np.eye(source_points.shape[1] + 1)

        transform = initial_transform.copy()
        prev_error = float('inf')

        for iteration in range(self.icp_max_iterations):
            # Step 1: 变换源点云
            if source_points.shape[1] == 2:
                pts_homo = np.hstack([source_points, np.ones((len(source_points), 1))])
                pts_homo = np.hstack([source_points,
                                      np.zeros((len(source_points), 1)),
                                      np.ones((len(source_points), 1))])
                transformed = (transform @ pts_homo[:, [0, 1, 2, 3]].T).T[:, :2]
                map_pts = self._map_pointcloud[:, :2]
            else:
                pts_homo = np.hstack([source_points, np.ones((len(source_points), 1))])
                transformed = (transform @ pts_homo.T).T[:, :3]
                map_pts = self._map_pointcloud[:, :3]

            # Step 2: 最近邻搜索
            distances, indices = self._map_kdtree.query(
                transformed,
                distance_upper_bound=self.max_correspondence_distance
            )
            valid = np.isfinite(distances)
            correspondences = indices[valid]
            source_corr = source_points[valid]
            target_corr = map_pts[correspondences]

            if len(correspondences) < 10:
                break

            # Step 3: SVD 求解最优变换
            src_centroid = source_corr.mean(axis=0)
            tgt_centroid = target_corr.mean(axis=0)

            # 转换为齐次坐标计算
            src_demean = source_corr - src_centroid
            tgt_demean = target_corr - tgt_centroid

            if source_points.shape[1] == 2:
                H_mat = src_demean.T @ tgt_demean
                U, _, Vt = np.linalg.svd(H_mat)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[-1] *= -1
                    R = Vt.T @ U.T
                t_vec = tgt_centroid - R @ src_centroid

                transform[:2, :2] = R
                transform[:2, 2] = t_vec
            else:
                H_mat = src_demean.T @ tgt_demean
                U, _, Vt = np.linalg.svd(H_mat)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[-1] *= -1
                    R = Vt.T @ U.T
                t_vec = tgt_centroid - R @ src_centroid

                transform[:3, :3] = R
                transform[:3, 3] = t_vec

            # 收敛判断
            mean_error = distances[valid].mean()
            if abs(prev_error - mean_error) < self.icp_tolerance:
                break
            prev_error = mean_error

        # 计算最终匹配分数
        final_distances, _ = self._map_kdtree.query(
            (transform[:2, :2] @ source_points[:, :2].T).T + transform[:2, 2][:2],
            distance_upper_bound=self.max_correspondence_distance
        )
        match_score = 1.0 / (1.0 + np.mean(final_distances[np.isfinite(final_distances)]))

        return MapMatchResult(
            x=float(transform[0, -1]), y=float(transform[1, -1]),
            z=float(transform[2, -1]) if transform.shape[0] > 2 else 0.0,
            yaw=np.arctan2(transform[1, 0], transform[0, 0]),
            pitch=0.0, roll=0.0,
            match_score=float(match_score),
            num_correspondences=len(correspondences)
        )

    def ndt_align(self, source_points: NDArray,
                  initial_guess: Optional[NDArray] = None) -> MapMatchResult:
        """
        NDT 匹配 (用于粗配准)
        """
        if not self._voxel_params:
            raise ValueError("Map not loaded. Call load_map() first.")

        if initial_guess is not None:
            x0 = np.array([
                initial_guess[0, 3], initial_guess[1, 3],
                np.arctan2(initial_guess[1, 0], initial_guess[0, 0])
            ])
        else:
            x0 = np.array([0.0, 0.0, 0.0])

        def objective(params):
            x, y, yaw = params
            c, s = np.cos(yaw), np.sin(yaw)
            transform = np.array([[c, -s, 0, x],
                                  [s, c, 0, y],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
            return [self.ndt_score(transform, source_points)]

        result = least_squares(objective, x0, method='LM')

        return MapMatchResult(
            x=float(result.x[0]), y=float(result.x[1]), z=0.0,
            yaw=float(result.x[2]), pitch=0.0, roll=0.0,
            match_score=1.0 / (1.0 + float(result.cost)),
            num_correspondences=0,
            convergence=result.success,
            iterations=result.nfev,
        )

    def multi_resolution_align(self, source_points: NDArray,
                               initial_guess: Optional[NDArray] = None
                               ) -> MapMatchResult:
        """
        多分辨率配准: NDT 粗配准 → ICP 精配准
        """
        # Step 1: 降采样 → NDT 粗配准
        downsample_rate = max(1, len(source_points) // 500)
        coarse_points = source_points[::downsample_rate]

        ndt_result = self.ndt_align(coarse_points, initial_guess)

        if not ndt_result.convergence:
            return ndt_result

        # 构建 NDT 结果的变换矩阵
        c = np.cos(ndt_result.yaw)
        s = np.sin(ndt_result.yaw)
        ndt_transform = np.array([
            [c, -s, 0, ndt_result.x],
            [s, c, 0, ndt_result.y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Step 2: ICP 精配准
        icp_result = self.icp_align(source_points, ndt_transform)

        return icp_result
