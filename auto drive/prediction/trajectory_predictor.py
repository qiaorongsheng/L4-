"""
轨迹预测模块 —— 多模态轨迹预测 (基于多项式 + 车道约束)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from numpy.typing import NDArray


@dataclass
class PredictedTrajectory:
    """单条预测轨迹"""
    object_id: int
    trajectory: NDArray           # (T, 4) [x, y, vx, vy] 预测轨迹
    probability: float            # 该模态的概率
    modality: str                 # 模态标签 ('keep', 'left', 'right', 'stop', ...)
    timestamps: Optional[NDArray] = None  # (T,) 时间戳
    uncertainty: Optional[NDArray] = None # (T, 2) 标准差


@dataclass
class TrajectoryPrediction:
    """完整多模态轨迹预测"""
    object_id: int
    modalities: List[PredictedTrajectory]
    prediction_horizon: float     # 预测时域 [s]
    prediction_dt: float          # 时间步长 [s]
    # 最可能的轨迹
    most_likely: Optional[PredictedTrajectory] = None


class TrajectoryPredictor:
    """
    多模态轨迹预测:
      - 恒速/恒加速度外推 (baseline)
      - 车道中心线约束投影
      - 多项式轨迹生成
      - 多模态采样 (keep lane, left change, right change, stop)
      - 实际部署替换为: VectorNet / LaneGCN / Wayformer
    """

    def __init__(self, prediction_horizon: float = 8.0,
                 prediction_dt: float = 0.2,
                 num_modalities: int = 4):
        self.prediction_horizon = prediction_horizon
        self.prediction_dt = prediction_dt
        self.num_modalities = num_modalities

    def _constant_velocity_extrapolation(self, state: NDArray,
                                         num_steps: int) -> NDArray:
        """
        恒速模型外推
        state: [x, y, vx, vy, yaw]
        返回: (T, 4) [x, y, vx, vy]
        """
        x, y, vx, vy, yaw = state

        trajectory = np.zeros((num_steps, 4))

        for i in range(num_steps):
            t = (i + 1) * self.prediction_dt
            trajectory[i, 0] = x + vx * t
            trajectory[i, 1] = y + vy * t
            trajectory[i, 2] = vx
            trajectory[i, 3] = vy

        return trajectory

    def _constant_acceleration_extrapolation(self, state: NDArray,
                                             num_steps: int) -> NDArray:
        """
        恒加速度模型外推
        state: [x, y, vx, vy, ax, ay, yaw]
        """
        x, y, vx, vy, ax, ay, yaw = state

        trajectory = np.zeros((num_steps, 4))

        for i in range(num_steps):
            t = (i + 1) * self.prediction_dt
            trajectory[i, 0] = x + vx * t + 0.5 * ax * t ** 2
            trajectory[i, 1] = y + vy * t + 0.5 * ay * t ** 2
            trajectory[i, 2] = vx + ax * t
            trajectory[i, 3] = vy + ay * t

        return trajectory

    def _lane_constrained_trajectory(self, state: NDArray,
                                     lane_center_coeffs: NDArray,
                                     num_steps: int,
                                     target_lateral_offset: float = 0.0
                                     ) -> NDArray:
        """
        车道中心线约束的轨迹生成
        目标: 在预测时域内平滑过渡到目标横向位置
        """
        x, y, vx, vy, yaw = state
        v = np.sqrt(vx ** 2 + vy ** 2)
        if v < 0.5:
            v = 0.5

        trajectory = np.zeros((num_steps, 4))

        for i in range(num_steps):
            t = (i + 1) * self.prediction_dt

            # 纵向: 恒速
            s = v * t
            traj_x = x + s * np.cos(yaw)
            traj_y = y + s * np.sin(yaw)

            # 横向: 一阶过渡到目标偏移
            alpha = min(1.0, t / 2.0)  # 2秒过渡
            target_y = np.polyval(lane_center_coeffs, traj_x) + target_lateral_offset
            traj_y = y + (target_y - y) * alpha

            trajectory[i, 0] = traj_x
            trajectory[i, 1] = traj_y
            trajectory[i, 2] = v * np.cos(yaw)
            trajectory[i, 3] = v * np.sin(yaw)

        return trajectory

    def _stopping_trajectory(self, state: NDArray,
                             num_steps: int,
                             decel: float = -3.0) -> NDArray:
        """
        停车轨迹 (恒定减速度)
        """
        x, y, vx, vy, yaw = state
        v = np.sqrt(vx ** 2 + vy ** 2)

        trajectory = np.zeros((num_steps, 4))
        heading = np.arctan2(vy, vx) if v > 0.1 else yaw

        for i in range(num_steps):
            t = (i + 1) * self.prediction_dt
            vt = max(0, v + decel * t)
            avg_v = (v + vt) / 2
            s = avg_v * t

            trajectory[i, 0] = x + s * np.cos(heading)
            trajectory[i, 1] = y + s * np.sin(heading)
            trajectory[i, 2] = vt * np.cos(heading)
            trajectory[i, 3] = vt * np.sin(heading)

        return trajectory

    def _sigmoid_lane_change_trajectory(self, state: NDArray,
                                        lane_width: float,
                                        direction: int,  # +1 = left, -1 = right
                                        num_steps: int) -> NDArray:
        """
        Sigmoid 变道轨迹
        y(x) = lane_width / (1 + exp(-k*(x - x0)))
        """
        x, y, vx, vy, yaw = state
        v = max(np.sqrt(vx ** 2 + vy ** 2), 1.0)

        target_y_offset = direction * lane_width
        x0 = 30.0  # 变道中心点距离 [m]
        k = 0.15   # 曲率参数

        trajectory = np.zeros((num_steps, 4))

        for i in range(num_steps):
            t = (i + 1) * self.prediction_dt
            dx = v * t
            traj_x = x + dx * np.cos(yaw)
            traj_y = y + target_y_offset / (1 + np.exp(-k * (dx - x0)))
            v_remaining = v

            trajectory[i, 0] = traj_x
            trajectory[i, 1] = traj_y
            trajectory[i, 2] = v_remaining * np.cos(yaw)
            trajectory[i, 3] = v_remaining * np.sin(yaw)

        return trajectory

    def predict(self, tracked_objects: List,
                behavior_predictions: List,
                lane_boundaries,
                lane_width: float = 3.5) -> List[TrajectoryPrediction]:
        """
        多目标多模态轨迹预测主入口
        """
        num_steps = int(self.prediction_horizon / self.prediction_dt)
        all_predictions = []

        for obj in tracked_objects:
            # 提取当前状态
            state = np.array([
                obj.mean[0], obj.mean[1],    # x, y
                obj.mean[7], obj.mean[8],    # vx, vy
                obj.mean[3],                  # yaw
            ])

            # 获取对应行为预测
            behavior = next((b for b in behavior_predictions
                             if b.object_id == obj.track_id), None)

            modalities = []

            # ---- 模态 1: 保持车道 ----
            lane_traj = self._lane_constrained_trajectory(
                state, np.array([0, 0, 0, 0]), num_steps, 0.0
            )
            modalities.append(PredictedTrajectory(
                object_id=obj.track_id,
                trajectory=lane_traj,
                probability=behavior.intent_probabilities.get(
                    'lane_keep', 0.5) if behavior else 0.5,
                modality='lane_keep',
            ))

            # ---- 模态 2: 左变道 ----
            left_traj = self._sigmoid_lane_change_trajectory(
                state, lane_width, +1, num_steps
            )
            modalities.append(PredictedTrajectory(
                object_id=obj.track_id,
                trajectory=left_traj,
                probability=behavior.intent_probabilities.get(
                    'lane_change_left', 0.1) if behavior else 0.1,
                modality='lane_change_left',
            ))

            # ---- 模态 3: 右变道 ----
            right_traj = self._sigmoid_lane_change_trajectory(
                state, lane_width, -1, num_steps
            )
            modalities.append(PredictedTrajectory(
                object_id=obj.track_id,
                trajectory=right_traj,
                probability=behavior.intent_probabilities.get(
                    'lane_change_right', 0.1) if behavior else 0.1,
                modality='lane_change_right',
            ))

            # ---- 模态 4: 停车 ----
            if behavior and behavior.intent == 'stop':
                stop_traj = self._stopping_trajectory(state, num_steps)
                modalities.append(PredictedTrajectory(
                    object_id=obj.track_id,
                    trajectory=stop_traj,
                    probability=behavior.intent_probabilities.get('stop', 0.1),
                    modality='stop',
                ))

            # 归一化概率
            total_prob = sum(m.probability for m in modalities)
            if total_prob > 0:
                for m in modalities:
                    m.probability /= total_prob

            # 最可能轨迹
            most_likely = max(modalities, key=lambda m: m.probability)

            all_predictions.append(TrajectoryPrediction(
                object_id=obj.track_id,
                modalities=modalities,
                prediction_horizon=self.prediction_horizon,
                prediction_dt=self.prediction_dt,
                most_likely=most_likely,
            ))

        return all_predictions
