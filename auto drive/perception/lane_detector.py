"""
车道线检测模块 —— 基于视觉 + LiDAR 融合的车道线拟合
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class LaneLine:
    """车道线参数化表示"""
    lane_id: int
    polynomial: NDArray           # 三次多项式系数 [c0, c1, c2, c3] (y = Σ c_i x^i)
    lane_type: str                # 'solid', 'dashed', 'double', 'curb'
    color: str                    # 'white', 'yellow', 'blue'
    confidence: float
    start_x: float
    end_x: float
    width: float = 0.15           # 线宽 [m]


@dataclass
class LaneBoundary:
    """车道边界 —— 左右两侧车道线组成"""
    boundary_id: int
    left_line: LaneLine
    right_line: LaneLine
    lane_index: int               # 从本车道向左偏移 (0=本车道, 1=左1, -1=右1)
    is_ego_lane: bool = False


class LaneDetector:
    """
    多源车道线检测与拟合:
      - 视觉车道线特征提取 (BEV 空间)
      - LiDAR 反射强度车道线提取
      - 三次多项式拟合 (RANSAC + 最小二乘)
      - 车道拓扑构建
    """

    def __init__(self, num_lanes: int = 4,
                 lane_width_default: float = 3.5,
                 polynomial_order: int = 3):
        self.num_lanes = num_lanes
        self.lane_width_default = lane_width_default
        self.polynomial_order = polynomial_order

    def extract_lane_points_from_bev(self, lane_mask: NDArray,
                                     bev_resolution: float = 0.1,
                                     bev_center: Tuple[float, float] = (0, 0)
                                     ) -> NDArray:
        """
        从 BEV 车道线掩码中提取车道线点集
        返回: (N, 2) 点集 (x, y in vehicle frame)
        """
        if lane_mask.sum() == 0:
            return np.array([]).reshape(0, 2)

        # 提取非零像素坐标
        rows, cols = np.where(lane_mask > 0)

        # 像素坐标 → 车辆坐标系
        # BEV: x 前向, y 左向
        x = (cols - lane_mask.shape[1] / 2) * bev_resolution
        y = (lane_mask.shape[0] / 2 - rows) * bev_resolution

        return np.column_stack([x, y])

    def extract_lane_points_from_lidar(self,
                                       ground_points: NDArray,
                                       intensity_threshold: float = 0.3
                                       ) -> NDArray:
        """
        从 LiDAR 地面点中提取高反射率车道线标记点
        ground_points: (N, 4) [x, y, z, intensity]
        """
        if len(ground_points) == 0:
            return np.array([]).reshape(0, 2)

        # 高反射率 = 车道线标记
        lane_mask = ground_points[:, 3] > intensity_threshold
        lane_pts = ground_points[lane_mask]

        return lane_pts[:, :2]  # (N, 2)

    def _ransac_polynomial_fit(self, points: NDArray,
                               order: int = 3,
                               n_iterations: int = 100,
                               inlier_threshold: float = 0.3
                               ) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """
        RANSAC 鲁棒多项式拟合
        返回: (coeffs, inlier_mask) 或 (None, None)
        """
        if len(points) < order + 2:
            return None, None

        x, y = points[:, 0], points[:, 1]
        best_inliers = []
        best_coeffs = None
        max_inliers = 0

        for _ in range(n_iterations):
            # 随机采样
            sample_idx = np.random.choice(len(points),
                                          min(order + 2, len(points)),
                                          replace=False)
            samples = points[sample_idx]

            try:
                # 最小二乘拟合
                coeffs = np.polyfit(samples[:, 0], samples[:, 1], order)

                # 计算所有点到拟合曲线的距离
                y_pred = np.polyval(coeffs, x)
                residuals = np.abs(y - y_pred)
                inliers = np.where(residuals < inlier_threshold)[0]

                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_inliers = inliers
                    best_coeffs = coeffs
            except np.linalg.LinAlgError:
                continue

        # 使用所有内点重新拟合
        if best_coeffs is not None and len(best_inliers) > order + 1:
            best_coeffs = np.polyfit(
                x[best_inliers], y[best_inliers], order
            )
            inlier_mask = np.zeros(len(points), dtype=bool)
            inlier_mask[best_inliers] = True
            return best_coeffs, inlier_mask

        return best_coeffs, None

    def _cluster_lines_left_right(self, points: NDArray,
                                  lane_centers: List[float]) -> List[NDArray]:
        """
        将点集按 y 坐标聚类为左右车道线
        lane_centers: 期望的车道中心 y 坐标列表
        """
        if len(points) == 0:
            return []

        y = points[:, 1]
        clusters = []

        for center in lane_centers:
            # 窗口内提取点
            window = 1.0  # [m]
            mask = (y > center - window) & (y < center + window)
            cluster_pts = points[mask]

            if len(cluster_pts) > 10:
                clusters.append(cluster_pts)

        return clusters

    def detect_lanes(self, camera_lane_points: NDArray,
                     lidar_lane_points: NDArray,
                     ego_x: float = 0.0) -> List[LaneBoundary]:
        """
        主车道线检测: 融合视觉 + LiDAR 车道线点, 拟合车道边界
        """
        # Step 1: 融合点集
        all_points = []
        if len(camera_lane_points) > 0:
            all_points.append(camera_lane_points)
        if len(lidar_lane_points) > 0:
            all_points.append(lidar_lane_points)

        if not all_points:
            return self._default_lane_boundaries()

        fused_points = np.vstack(all_points)

        # Step 2: 按 y 坐标分层聚类
        # 车道中心 y 位置: 以本车为中心, 左右各 N 条
        lane_offsets = []
        for i in range(-self.num_lanes // 2, self.num_lanes // 2 + 1):
            lane_offsets.append(i * self.lane_width_default)

        point_clusters = self._cluster_lines_left_right(fused_points, lane_offsets)

        # Step 3: 每个聚类拟合多项式
        lane_lines = []
        lane_id_counter = 0

        for i, cluster in enumerate(point_clusters):
            coeffs, _ = self._ransac_polynomial_fit(
                cluster, order=self.polynomial_order
            )
            if coeffs is not None:
                x_min, x_max = cluster[:, 0].min(), cluster[:, 0].max()
                # 判断线型 (基于点的连续性)
                x_sorted = np.sort(cluster[:, 0])
                gaps = np.diff(x_sorted)
                median_gap = np.median(gaps)
                line_type = 'dashed' if median_gap > 0.5 else 'solid'

                lane_lines.append(LaneLine(
                    lane_id=lane_id_counter,
                    polynomial=coeffs,
                    lane_type=line_type,
                    color='white',
                    confidence=min(1.0, len(cluster) / 200),
                    start_x=float(x_min),
                    end_x=float(x_max),
                ))
                lane_id_counter += 1

        # Step 4: 配对左右车道线 → 车道边界
        boundaries = self._pair_lane_boundaries(lane_lines)

        return boundaries if boundaries else self._default_lane_boundaries()

    def _pair_lane_boundaries(self, lane_lines: List[LaneLine]) -> List[LaneBoundary]:
        """将车道线配对成车道边界"""
        # 按 y 偏移量排序 (在 x=0 处评估)
        lane_lines = sorted(lane_lines,
                            key=lambda l: np.polyval(l.polynomial, 0))

        boundaries = []
        boundary_id = 0

        for i in range(len(lane_lines) - 1):
            left = lane_lines[i]
            right = lane_lines[i + 1]

            # 两条线在 x=0 处的间距
            left_y0 = np.polyval(left.polynomial, 0)
            right_y0 = np.polyval(right.polynomial, 0)
            spacing = abs(right_y0 - left_y0)

            # 合法车道宽度范围
            if 2.5 < spacing < 5.5:
                # 判断是否为本车道 (y≈0 在本车中心)
                is_ego = (left_y0 < 0 < right_y0)
                # 车道索引
                mid_y = (left_y0 + right_y0) / 2
                lane_idx = round(mid_y / self.lane_width_default)

                boundaries.append(LaneBoundary(
                    boundary_id=boundary_id,
                    left_line=left,
                    right_line=right,
                    lane_index=lane_idx,
                    is_ego_lane=is_ego,
                ))
                boundary_id += 1

        return boundaries

    def _default_lane_boundaries(self) -> List[LaneBoundary]:
        """检测失败时的默认车道 (基于标准车道宽度)"""
        boundaries = []
        for i in range(-1, 2):  # 左1, 本, 右1
            center_y = i * self.lane_width_default
            left_coeffs = np.array([center_y - self.lane_width_default / 2,
                                    0, 0, 0])
            right_coeffs = np.array([center_y + self.lane_width_default / 2,
                                     0, 0, 0])

            left = LaneLine(
                lane_id=i * 2, polynomial=left_coeffs,
                lane_type='solid', color='white', confidence=0.3,
                start_x=0, end_x=200,
            )
            right = LaneLine(
                lane_id=i * 2 + 1, polynomial=right_coeffs,
                lane_type='solid', color='white', confidence=0.3,
                start_x=0, end_x=200,
            )
            boundaries.append(LaneBoundary(
                boundary_id=i + 1,
                left_line=left, right_line=right,
                lane_index=i, is_ego_lane=(i == 0),
            ))

        return boundaries

    def get_ego_lane_center(self, boundaries: List[LaneBoundary],
                            lookahead_x: float) -> Tuple[float, float]:
        """
        获取本车道中心线在指定纵向距离处的坐标
        返回: (center_y, heading)
        """
        ego_lane = next((b for b in boundaries if b.is_ego_lane), None)

        if ego_lane is None:
            return (0.0, 0.0)

        left_y = np.polyval(ego_lane.left_line.polynomial, lookahead_x)
        right_y = np.polyval(ego_lane.right_line.polynomial, lookahead_x)
        center_y = (left_y + right_y) / 2

        # 车道朝向 (导数)
        left_deriv = np.polyval(
            np.polyder(ego_lane.left_line.polynomial), lookahead_x
        )
        right_deriv = np.polyval(
            np.polyder(ego_lane.right_line.polynomial), lookahead_x
        )
        heading = np.arctan((left_deriv + right_deriv) / 2)

        return (center_y, heading)
