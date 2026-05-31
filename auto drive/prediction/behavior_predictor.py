"""
行为预测模块 —— 周围交通参与者的意图识别与行为分类
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from numpy.typing import NDArray


class BehaviorIntent(Enum):
    """行为意图"""
    LANE_KEEP = "lane_keep"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
    YIELD = "yield"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    PARKING = "parking"
    UNKNOWN = "unknown"


@dataclass
class BehaviorPrediction:
    """行为预测结果"""
    object_id: int
    intent: BehaviorIntent
    intent_probabilities: Dict[BehaviorIntent, float]
    # 时间信息
    prediction_time: float
    time_to_lane_change: Optional[float] = None  # 预计变道时间 [s]
    time_to_stop: Optional[float] = None         # 预计停车时间 [s]
    # 交互标签
    yields_to_ego: bool = False
    cooperative: bool = False                    # 是否合作型驾驶员


class BehaviorPredictor:
    """
    交通参与者行为意图预测:
      - 基于规则的意图推理 (车道关系 + 运动趋势)
      - 隐马尔可夫模型 (HMM) 驾驶行为序列建模
      - 社会交互特征提取 (谁先让, 谁抢行)
      - 实际部署替换为基于学习的模型 (LSTM / Transformer)
    """

    # HMM 状态转移先验
    BEHAVIOR_TRANSITIONS = {
        BehaviorIntent.LANE_KEEP: {
            BehaviorIntent.LANE_KEEP: 0.85,
            BehaviorIntent.LANE_CHANGE_LEFT: 0.05,
            BehaviorIntent.LANE_CHANGE_RIGHT: 0.05,
            BehaviorIntent.DECELERATE: 0.03,
            BehaviorIntent.ACCELERATE: 0.02,
        },
        BehaviorIntent.LANE_CHANGE_LEFT: {
            BehaviorIntent.LANE_CHANGE_LEFT: 0.70,
            BehaviorIntent.LANE_KEEP: 0.25,
            BehaviorIntent.ACCELERATE: 0.05,
        },
        BehaviorIntent.LANE_CHANGE_RIGHT: {
            BehaviorIntent.LANE_CHANGE_RIGHT: 0.70,
            BehaviorIntent.LANE_KEEP: 0.25,
            BehaviorIntent.ACCELERATE: 0.05,
        },
    }

    def __init__(self, lane_width: float = 3.5,
                 lane_change_lateral_threshold: float = 1.5):
        self.lane_width = lane_width
        self.lane_change_lateral_threshold = lane_change_lateral_threshold

        # 历史行为
        self._behavior_history: Dict[int, List[BehaviorIntent]] = {}

    def _compute_lateral_offset(self, obj_pos: NDArray,
                                lane_center_y: float) -> float:
        """计算目标相对于车道中心的横向偏移"""
        return obj_pos[1] - lane_center_y

    def _compute_lateral_velocity_trend(self, tracked_obj,
                                        lookback_frames: int = 10) -> float:
        """
        计算横向运动趋势 (正=向左移动)
        """
        if len(tracked_obj.history) < lookback_frames:
            return 0.0

        recent = tracked_obj.history[-lookback_frames:]
        y_positions = [s[1] for s in recent]
        times = np.arange(len(y_positions))

        if len(set(times)) < 2:
            return 0.0

        # 线性回归拟合横向位移趋势
        A = np.vstack([times, np.ones_like(times)]).T
        slope, _ = np.linalg.lstsq(A, y_positions, rcond=None)[0]

        return float(slope)

    def _compute_ttc(self, ego_state, obj_state,
                     ego_v: float, obj_v: float) -> float:
        """
        计算碰撞时间 (TTC) 基于纵向距离
        """
        rel_x = obj_state[0] - ego_state[0]  # 正面相对距离
        rel_v = ego_v - obj_v                # 相对速度 (正向车)

        if abs(rel_v) < 0.1:
            return float('inf')

        ttc = rel_x / max(rel_v, 1e-3)
        return ttc

    def _infer_lane_change_intent(self, obj, lateral_offset: float,
                                  lateral_trend: float,
                                  lane_boundaries) -> float:
        """
        推断变道意图概率 (规则 + 运动趋势)
        """
        # 归一化横向偏移
        normalized_offset = lateral_offset / self.lane_width

        # 横向速度阈值
        vl_threshold = 0.3  # m/s

        prob_left = 0.0
        prob_right = 0.0
        prob_keep = 0.0

        # 如果正在穿越车道边界
        if abs(normalized_offset) > 0.4:  # 偏离车道中心超过 40% 车道宽
            if lateral_trend > vl_threshold:
                prob_left = 0.6 + 0.2 * abs(normalized_offset)
            elif lateral_trend < -vl_threshold:
                prob_right = 0.6 + 0.2 * abs(normalized_offset)

        # 靠近车道边界
        if abs(normalized_offset) > 0.35:
            if lateral_offset > 0 and lateral_trend > 0.1:
                prob_left = 0.7
            elif lateral_offset < 0 and lateral_trend < -0.1:
                prob_right = 0.7

        # 默认保持车道
        if prob_left + prob_right < 0.3:
            prob_keep = 0.7

        # 归一化
        total = prob_left + prob_right + prob_keep
        if total > 0:
            prob_left /= total
            prob_right /= total
            prob_keep /= total
        else:
            prob_keep = 1.0

        return prob_left, prob_right, prob_keep

    def predict_behavior(self, tracked_objects: List,
                         ego_state, lane_boundaries,
                         map_context: Dict = None) -> List[BehaviorPrediction]:
        """
        多目标行为预测主函数
        """
        predictions = []

        ego_pos = np.array([ego_state.x, ego_state.y])
        ego_v = ego_state.vx

        for obj in tracked_objects:
            obj_pos = np.array([obj.mean[0], obj.mean[1]])
            obj_v = np.hypot(obj.mean[7], obj.mean[8])

            # Step 1: 横向偏移
            # 获取目标所在车道的中心线
            lane_y = 0.0  # 默认本车道中心
            # TODO: 根据目标位置匹配最近车道

            lateral_offset = self._compute_lateral_offset(obj_pos, lane_y)

            # Step 2: 横向运动趋势
            lateral_trend = self._compute_lateral_velocity_trend(obj)

            # Step 3: 变道意图推断
            p_left, p_right, p_keep = self._infer_lane_change_intent(
                obj, lateral_offset, lateral_trend, lane_boundaries
            )

            # Step 4: 纵向意图推断
            ttc = self._compute_ttc(ego_pos, obj_pos, ego_v, obj_v)

            # 停止意图
            is_decelerating = obj_v < 2.0 and obj.mean[7] * obj.mean[7] + \
                              obj.mean[8] * obj.mean[8] < 1.0

            # 构建意图概率分布
            intent_probs = {
                BehaviorIntent.LANE_KEEP: p_keep,
                BehaviorIntent.LANE_CHANGE_LEFT: p_left,
                BehaviorIntent.LANE_CHANGE_RIGHT: p_right,
                BehaviorIntent.DECELERATE: 0.1 if is_decelerating else 0.01,
                BehaviorIntent.STOP: 0.3 if obj_v < 0.5 else 0.1,
                BehaviorIntent.ACCELERATE: 0.1 if obj_v < 5.0 and not is_decelerating else 0.01,
            }

            # 主意图
            main_intent = max(intent_probs, key=intent_probs.get)

            # 交互预测
            yields_to_ego = False
            if ttc < 5.0 and ttc > 0:
                # 如果对方减速且 TTC 增大 → 让行
                if obj_v < ego_v and lateral_offset < 1.0:
                    yields_to_ego = True

            # 时间估计
            tlc = None
            if main_intent in [BehaviorIntent.LANE_CHANGE_LEFT,
                               BehaviorIntent.LANE_CHANGE_RIGHT]:
                remaining_distance = self.lane_width - abs(lateral_offset)
                lateral_speed = abs(lateral_trend) if abs(lateral_trend) > 0.1 else 0.5
                tlc = remaining_distance / lateral_speed

            tts = None
            if main_intent == BehaviorIntent.STOP and obj_v > 0.1:
                decel = max(abs(obj.mean[7]), 1.0)  # 估计减速度
                tts = obj_v / decel

            # 更新历史
            if obj.track_id not in self._behavior_history:
                self._behavior_history[obj.track_id] = []
            self._behavior_history[obj.track_id] = \
                (self._behavior_history[obj.track_id] + [main_intent])[-20:]

            predictions.append(BehaviorPrediction(
                object_id=obj.track_id,
                intent=main_intent,
                intent_probabilities=intent_probs,
                prediction_time=0.0,
                time_to_lane_change=tlc,
                time_to_stop=tts,
                yields_to_ego=yields_to_ego,
                cooperative=not main_intent.value.startswith('lane_change'),
            ))

        return predictions
