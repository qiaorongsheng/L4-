"""
冗余检查模块 —— 多传感器/多源交叉验证
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class RedundancyCheck:
    """冗余检查结果"""
    check_name: str
    sources: List[str]
    agreement: float           # 0=完全不一致, 1=完全一致
    discrepancy: float         # 差异度量
    passed: bool
    detail: str


class RedundancyChecker:
    """
    L4 冗余检查器:
      - 多传感器交叉验证 (Camera vs LiDAR vs Radar)
      - 定位冗余 (GNSS vs Map Matching vs Wheel Odometry)
      - 感知冗余 (多模型/多帧一致性)
      - 硬件冗余 (主 ECU vs 备份 ECU)
    """

    def __init__(self, position_tolerance: float = 1.0,
                 velocity_tolerance: float = 2.0,
                 object_match_iou: float = 0.5):
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.object_match_iou = object_match_iou

    def check_localization_consistency(self,
                                       gnss_position: NDArray,
                                       map_match_position: NDArray,
                                       wheel_odom_position: NDArray
                                       ) -> RedundancyCheck:
        """
        多源定位一致性检查
        """
        # GNSS ↔ Map Match
        gnss_map_dist = np.linalg.norm(gnss_position[:2] -
                                       map_match_position[:2])

        # GNSS ↔ Wheel Odometry
        gnss_wheel_dist = np.linalg.norm(gnss_position[:2] -
                                         wheel_odom_position[:2])

        # Map Match ↔ Wheel Odometry
        map_wheel_dist = np.linalg.norm(map_match_position[:2] -
                                        wheel_odom_position[:2])

        max_discrepancy = max(gnss_map_dist, gnss_wheel_dist, map_wheel_dist)
        agreement = max(0, 1.0 - max_discrepancy / self.position_tolerance)
        passed = max_discrepancy < self.position_tolerance

        return RedundancyCheck(
            check_name='localization_consistency',
            sources=['gnss', 'map_match', 'wheel_odom'],
            agreement=float(agreement),
            discrepancy=float(max_discrepancy),
            passed=passed,
            detail=f'Max pos diff: {max_discrepancy:.2f}m '
                   f'(tol={self.position_tolerance}m)',
        )

    def check_velocity_consistency(self,
                                   gnss_velocity: NDArray,
                                   wheel_speed: float,
                                   radar_velocity: NDArray
                                   ) -> RedundancyCheck:
        """
        多源速度一致性检查
        """
        # 各来源的速度模
        gnss_speed = np.linalg.norm(gnss_velocity[:2])
        radar_speed = np.linalg.norm(radar_velocity[:2])

        speeds = [gnss_speed, wheel_speed, radar_speed]
        max_diff = max(speeds) - min(speeds)

        agreement = max(0, 1.0 - max_diff / self.velocity_tolerance)
        passed = max_diff < self.velocity_tolerance

        return RedundancyCheck(
            check_name='velocity_consistency',
            sources=['gnss', 'wheel', 'radar'],
            agreement=float(agreement),
            discrepancy=float(max_diff),
            passed=passed,
            detail=f'Speed spread: {max_diff:.2f} m/s',
        )

    def check_perception_consistency(self,
                                     lidar_objects: List,
                                     camera_objects: List,
                                     radar_objects: List
                                     ) -> RedundancyCheck:
        """
        感知一致性: LiDAR ↔ Camera ↔ Radar 交叉验证
        """
        def match_objects(src, tgt, threshold=3.0):
            """计算两个传感器目标列表的匹配率"""
            if not src or not tgt:
                return 0.0

            matched = 0
            for s in src:
                src_pos = np.array([
                    getattr(s, 'x', s.center[0]),
                    getattr(s, 'y', s.center[1])
                ])
                for t in tgt:
                    tgt_pos = np.array([
                        getattr(t, 'x', t.center[0] if hasattr(t, 'center') else t[0]),
                        getattr(t, 'y', t.center[1] if hasattr(t, 'center') else t[1])
                    ])
                    if np.linalg.norm(src_pos - tgt_pos) < threshold:
                        matched += 1
                        break

            return matched / len(src)

        # 三组匹配率
        lidar_cam_match = match_objects(lidar_objects, camera_objects)
        lidar_radar_match = match_objects(lidar_objects, radar_objects)
        cam_radar_match = match_objects(camera_objects, radar_objects)

        avg_agreement = np.mean([lidar_cam_match, lidar_radar_match,
                                 cam_radar_match])
        passed = avg_agreement > 0.6

        return RedundancyCheck(
            check_name='perception_consistency',
            sources=['lidar', 'camera', 'radar'],
            agreement=float(avg_agreement),
            discrepancy=float(1.0 - avg_agreement),
            passed=passed,
            detail=f'Match rates: L-C={lidar_cam_match:.2f}, '
                   f'L-R={lidar_radar_match:.2f}, C-R={cam_radar_match:.2f}',
        )

    def check_planning_consistency(self,
                                   primary_trajectory: NDArray,
                                   backup_trajectory: NDArray
                                   ) -> RedundancyCheck:
        """
        主/备规划一致性 (安全规划必须一致)
        """
        if primary_trajectory is None or backup_trajectory is None:
            return RedundancyCheck(
                check_name='planning_consistency',
                sources=['primary_planner', 'backup_planner'],
                agreement=0.0,
                discrepancy=float('inf'),
                passed=False,
                detail='One or both planners failed',
            )

        # 轨迹终点偏差
        end_primary = primary_trajectory[:2, -1]
        end_backup = backup_trajectory[:2, -1]
        end_diff = np.linalg.norm(end_primary - end_backup)

        # 整条轨迹的 Frechet 距离近似
        traj_diff = np.mean(np.linalg.norm(
            primary_trajectory[:2, :] - backup_trajectory[:2, :], axis=0
        ))

        agreement = max(0, 1.0 - traj_diff / 5.0)
        passed = end_diff < 5.0 and traj_diff < 3.0

        return RedundancyCheck(
            check_name='planning_consistency',
            sources=['primary_planner', 'backup_planner'],
            agreement=float(agreement),
            discrepancy=float(end_diff),
            passed=passed,
            detail=f'End point diff: {end_diff:.2f}m, '
                   f'Mean traj diff: {traj_diff:.2f}m',
        )

    def run_all_checks(self, **kwargs) -> List[RedundancyCheck]:
        """
        运行所有冗余检查
        """
        checks = []

        if all(k in kwargs for k in ['gnss_position', 'map_match_position',
                                      'wheel_odom_position']):
            checks.append(self.check_localization_consistency(
                kwargs['gnss_position'],
                kwargs['map_match_position'],
                kwargs['wheel_odom_position'],
            ))

        if all(k in kwargs for k in ['gnss_velocity', 'wheel_speed',
                                      'radar_velocity']):
            checks.append(self.check_velocity_consistency(
                kwargs['gnss_velocity'],
                kwargs['wheel_speed'],
                kwargs['radar_velocity'],
            ))

        if all(k in kwargs for k in ['lidar_objects', 'camera_objects',
                                      'radar_objects']):
            checks.append(self.check_perception_consistency(
                kwargs['lidar_objects'],
                kwargs['camera_objects'],
                kwargs['radar_objects'],
            ))

        return checks
