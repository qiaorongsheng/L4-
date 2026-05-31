"""
多目标跟踪模块 —— 3D 卡尔曼滤波 + 匈牙利匹配
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from numpy.typing import NDArray


@dataclass
class TrackState:
    """单目标跟踪状态 (3D 卡尔曼滤波)"""
    track_id: int
    # 状态向量 [x, y, z, yaw, l, w, h, vx, vy, vz, vyaw]
    mean: NDArray          # (11,)
    covariance: NDArray    # (11, 11)
    # 跟踪属性
    age: int = 0           # 总存活帧数
    hits: int = 0          # 连续检测命中次数
    time_since_update: int = 0  # 自上次更新以来的帧数
    object_class: str = "unknown"
    confidence: float = 0.0
    # 历史轨迹
    history: List[NDArray] = field(default_factory=list)
    max_history: int = 50


class ObjectTracker:
    """
    3D 多目标跟踪器:
      - 恒定转弯速率和速度 (CTRV) 运动模型
      - 扩展卡尔曼滤波 (EKF) 状态估计
      - 匈牙利算法数据关联 (基于马氏距离 + 外观特征)
    """

    def __init__(self, max_age: int = 10, min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 mahalanobis_gate: float = 9.4877):  # chi2 0.95, df=4
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.mahalanobis_gate = mahalanobis_gate

        self._tracks: Dict[int, TrackState] = {}
        self._next_id = 0

    def _predict_track(self, track: TrackState, dt: float) -> TrackState:
        """
        CTRV 运动模型 EKF 预测
        状态: [x, y, z, yaw, l, w, h, vx, vy, vz, vyaw]
        """
        x = track.mean.copy()
        P = track.covariance.copy()

        # ---- 状态预测 (CTRV 非线性) ----
        yaw = x[3]
        vx = x[7]
        vy = x[8]
        vyaw = x[10]

        if abs(vyaw) > 1e-6:
            # 转弯运动
            ratio = vyaw
            x_new = np.array([
                x[0] + vx * np.sin(ratio * dt) / ratio,
                x[1] + vy * np.cos(ratio * dt) / ratio,
                x[2] + x[9] * dt,
                yaw + vyaw * dt,
                x[4], x[5], x[6],  # 尺寸不变
                vx, vy, x[9], vyaw,
            ])
        else:
            # 直线运动
            x_new = np.array([
                x[0] + vx * dt * np.cos(yaw),
                x[1] + vx * dt * np.sin(yaw),
                x[2] + x[9] * dt,
                yaw,
                x[4], x[5], x[6],
                vx, vy, x[9], vyaw,
            ])

        # ---- 雅可比矩阵 (状态转移) ----
        F = np.eye(11)

        if abs(vyaw) > 1e-6:
            ratio = vyaw
            d_x_wrt_yaw = vx * (np.cos(ratio * dt) - 1) / ratio
            d_y_wrt_yaw = vy * (np.cos(ratio * dt) - 1) / ratio
            F[0, 3] = d_x_wrt_yaw
            F[1, 3] = d_y_wrt_yaw
            F[0, 7] = np.sin(ratio * dt) / ratio
            F[1, 8] = np.cos(ratio * dt) / ratio
            F[0, 10] = vx * np.cos(ratio * dt) * dt / ratio
            F[1, 10] = vy * np.cos(ratio * dt) * dt / ratio
        else:
            F[0, 3] = -vx * np.sin(yaw) * dt
            F[1, 3] = vx * np.cos(yaw) * dt
            F[0, 7] = dt * np.cos(yaw)
            F[1, 7] = dt * np.sin(yaw)

        F[2, 9] = dt
        F[3, 10] = dt

        # 过程噪声
        Q = np.eye(11) * 0.01
        Q[7, 7] = 1.0   # vx 噪声
        Q[8, 8] = 1.0   # vy 噪声
        Q[10, 10] = 0.5 # vyaw 噪声

        P_new = F @ P @ F.T + Q

        return TrackState(
            track_id=track.track_id,
            mean=x_new, covariance=P_new,
            age=track.age + 1,
            hits=track.hits,
            time_since_update=track.time_since_update + 1,
            object_class=track.object_class,
            confidence=track.confidence,
            history=track.history,
        )

    def _update_track(self, track: TrackState, detection, dt: float) -> TrackState:
        """
        EKF 更新步骤
        观测: [x, y, z, yaw] (可观测部分)
        """
        # 观测矩阵 (我们观测位置和朝向)
        H = np.zeros((4, 11))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z
        H[3, 3] = 1  # yaw

        # 观测噪声
        R = np.diag([0.5, 0.5, 0.3, 0.1]) ** 2  # 根据传感器精度调整

        # 卡尔曼增益
        S = H @ track.covariance @ H.T + R
        K = track.covariance @ H.T @ np.linalg.inv(S)

        # 观测残差
        z = np.array([detection.center[0], detection.center[1],
                      detection.center[2], detection.yaw])
        y = z - H @ track.mean
        y[3] = np.arctan2(np.sin(y[3]), np.cos(y[3]))  # 角度归一化

        # 状态更新
        x_new = track.mean + K @ y
        P_new = (np.eye(11) - K @ H) @ track.covariance

        # 速度更新
        if track.hits > 0:
            dt_safe = max(dt, 0.01)
            alpha_v = 0.6
            x_new[7] = alpha_v * (x_new[0] - track.mean[0]) / dt_safe + \
                       (1 - alpha_v) * x_new[7]
            x_new[8] = alpha_v * (x_new[1] - track.mean[1]) / dt_safe + \
                       (1 - alpha_v) * x_new[8]

        # 融合检测中已有的速度估计
        if np.linalg.norm(detection.velocity) > 0.1:
            x_new[7] = 0.5 * x_new[7] + 0.5 * detection.velocity[0]
            x_new[8] = 0.5 * x_new[8] + 0.5 * detection.velocity[1]

        return TrackState(
            track_id=track.track_id,
            mean=x_new, covariance=P_new,
            age=track.age,
            hits=track.hits + 1,
            time_since_update=0,
            object_class=detection.object_class.value,
            confidence=detection.confidence,
            history=track.history + [x_new.copy()],
        )

    def _compute_cost_matrix(self, tracks: List[TrackState],
                             detections: List) -> NDArray:
        """计算关联代价矩阵 (马氏距离 + BEV IoU)"""
        n_tracks = len(tracks)
        n_dets = len(detections)

        if n_tracks == 0 or n_dets == 0:
            return np.array([]).reshape(n_tracks, n_dets)

        cost = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # BEV 中心马氏距离
                dz = det.center[:2] - track.mean[:2]
                cov_xy = track.covariance[:2, :2]
                try:
                    mah_dist = dz @ np.linalg.inv(cov_xy) @ dz
                except np.linalg.LinAlgError:
                    mah_dist = 1e6

                if mah_dist > self.mahalanobis_gate:
                    cost[i, j] = 1e6
                    continue

                # BEV IoU 项
                iou = self._compute_bev_iou(track, det)
                cost[i, j] = mah_dist - 5.0 * iou  # IoU 权重更高

        return cost

    def _compute_bev_iou(self, track: TrackState, det) -> float:
        """计算 BEV 投影下的旋转 IoU"""
        # 简化: 用中心距离 + 尺寸相似度近似
        center_dist = np.linalg.norm(det.center[:2] - track.mean[:2])

        tl, tw = track.mean[4], track.mean[5]
        dl, dw = det.dimensions[0], det.dimensions[1]
        size_sim = 1.0 - abs(tl - dl) / max(tl, dl, 1.0) * 0.5 - \
                   abs(tw - dw) / max(tw, dw, 1.0) * 0.5

        iou = max(0, 1.0 - center_dist / 3.0) * max(0, size_sim)
        return float(iou)

    def update(self, detections: List, dt: float) -> List[TrackState]:
        """
        单帧跟踪更新
        """
        # Step 1: 预测所有航迹
        predicted_tracks = [self._predict_track(t, dt)
                            for t in self._tracks.values()]

        # Step 2: 计算关联代价矩阵
        if len(predicted_tracks) > 0 and len(detections) > 0:
            cost = self._compute_cost_matrix(predicted_tracks, detections)

            if cost.size > 0:
                row_ind, col_ind = linear_sum_assignment(cost)

                matched_tracks = set()
                matched_dets = set()

                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < 50:  # 关联门限
                        matched_tracks.add(r)
                        matched_dets.add(c)

                        # 更新航迹
                        updated = self._update_track(
                            predicted_tracks[r], detections[c], dt
                        )
                        self._tracks[updated.track_id] = updated
            else:
                matched_tracks = set()
                matched_dets = set()
        else:
            matched_tracks = set()
            matched_dets = set()

        # Step 3: 处理未匹配的检测 → 新航迹
        for j, det in enumerate(detections):
            if len(matched_dets) == 0 or j not in matched_dets:
                if det.confidence > 0.6:
                    new_track = TrackState(
                        track_id=self._next_id,
                        mean=np.array([
                            det.center[0], det.center[1], det.center[2],
                            det.yaw,
                            det.dimensions[0], det.dimensions[1],
                            det.dimensions[2],
                            0, 0, 0, 0,  # vx, vy, vz, vyaw = 0
                        ]),
                        covariance=np.diag([1, 1, 0.5, 0.2, 0.2, 0.2, 0.2,
                                            2, 2, 1, 1]),
                        object_class=det.object_class.value,
                        confidence=det.confidence,
                    )
                    self._tracks[self._next_id] = new_track
                    self._next_id += 1

        # Step 4: 标记未匹配的航迹
        for i, track in enumerate(predicted_tracks):
            if i not in matched_tracks:
                track.time_since_update += 1
                self._tracks[track.track_id] = track

        # Step 5: 删除老/丢失航迹
        for tid in list(self._tracks.keys()):
            t = self._tracks[tid]
            if t.time_since_update > self.max_age:
                del self._tracks[tid]

        # Step 6: 返回确认的航迹
        confirmed = [t for t in self._tracks.values()
                     if t.hits >= self.min_hits]

        return confirmed
