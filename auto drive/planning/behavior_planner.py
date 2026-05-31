"""
行为决策模块 —— 有限状态机 (FSM) + 基于规则的驾驶行为决策
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto


class DrivingState(Enum):
    """驾驶状态机状态"""
    INIT = auto()
    STANDBY = auto()
    LANE_KEEPING = auto()
    LANE_CHANGE_PREPARE = auto()
    LANE_CHANGE_LEFT = auto()
    LANE_CHANGE_RIGHT = auto()
    CAR_FOLLOWING = auto()
    CRUISING = auto()
    YIELDING = auto()
    STOPPING = auto()
    EMERGENCY_STOP = auto()
    TURNING_LEFT = auto()
    TURNING_RIGHT = auto()
    INTERSECTION_CROSSING = auto()
    ROUNDABOUT_ENTERING = auto()
    OVERTAKING = auto()
    PARKING = auto()
    GOAL_REACHED = auto()


@dataclass
class BehaviorDecision:
    """行为决策输出"""
    state: DrivingState
    # 决策参数
    target_speed: float                     # 目标速度 [m/s]
    target_lane_id: Optional[int] = None    # 目标车道
    target_gap: Optional[float] = None      # 目标间隙 [m]
    # 时空约束
    time_to_execute: float = 0.0            # 执行时间窗口 [s]
    # 安全约束
    safe_to_proceed: bool = True
    min_safety_margin: float = 5.0          # 最小安全裕度 [m]
    # 决策置信度
    confidence: float = 1.0


class BehaviorPlanner:
    """
    行为决策规划器:
      - 层次有限状态机 (HFSM)
      - 基于规则 + 代价函数的决策
      - 场景分类与策略选择
      - 安全性门控 (safety gate)
    """

    # 状态转移规则
    STATE_TRANSITIONS = {
        DrivingState.STANDBY: [DrivingState.LANE_KEEPING, DrivingState.STOPPING],
        DrivingState.LANE_KEEPING: [
            DrivingState.CRUISING,
            DrivingState.CAR_FOLLOWING,
            DrivingState.LANE_CHANGE_PREPARE,
            DrivingState.YIELDING,
            DrivingState.STOPPING,
            DrivingState.EMERGENCY_STOP,
            DrivingState.INTERSECTION_CROSSING,
        ],
        DrivingState.CRUISING: [
            DrivingState.LANE_KEEPING,
            DrivingState.CAR_FOLLOWING,
            DrivingState.LANE_CHANGE_PREPARE,
            DrivingState.OVERTAKING,
        ],
        DrivingState.CAR_FOLLOWING: [
            DrivingState.LANE_KEEPING,
            DrivingState.CRUISING,
            DrivingState.LANE_CHANGE_PREPARE,
            DrivingState.OVERTAKING,
        ],
        DrivingState.LANE_CHANGE_PREPARE: [
            DrivingState.LANE_CHANGE_LEFT,
            DrivingState.LANE_CHANGE_RIGHT,
            DrivingState.LANE_KEEPING,  # 放弃变道
        ],
        DrivingState.LANE_CHANGE_LEFT: [
            DrivingState.LANE_KEEPING,
            DrivingState.CRUISING,
            DrivingState.CAR_FOLLOWING,
        ],
        DrivingState.LANE_CHANGE_RIGHT: [
            DrivingState.LANE_KEEPING,
            DrivingState.CRUISING,
            DrivingState.CAR_FOLLOWING,
        ],
        DrivingState.OVERTAKING: [
            DrivingState.LANE_KEEPING,
            DrivingState.CRUISING,
        ],
        DrivingState.YIELDING: [
            DrivingState.LANE_KEEPING,
            DrivingState.STOPPING,
            DrivingState.INTERSECTION_CROSSING,
        ],
        DrivingState.STOPPING: [
            DrivingState.STANDBY,
            DrivingState.LANE_KEEPING,
            DrivingState.YIELDING,
        ],
        DrivingState.EMERGENCY_STOP: [DrivingState.STOPPING],
        DrivingState.INTERSECTION_CROSSING: [
            DrivingState.LANE_KEEPING,
            DrivingState.TURNING_LEFT,
            DrivingState.TURNING_RIGHT,
        ],
        DrivingState.GOAL_REACHED: [DrivingState.STOPPING],
    }

    def __init__(self, safe_following_time: float = 2.0,
                 lane_change_min_gap: float = 25.0,
                 min_speed: float = 0.0,
                 max_speed: float = 33.3):
        self.safe_following_time = safe_following_time
        self.lane_change_min_gap = lane_change_min_gap
        self.min_speed = min_speed
        self.max_speed = max_speed

        # 当前状态
        self._current_state = DrivingState.INIT
        self._state_history: List[DrivingState] = []
        self._state_duration: Dict[DrivingState, float] = {}

    def _evaluate_lane_keeping(self, ego_state, fused_objects,
                              lane_boundaries, route_maneuvers) -> BehaviorDecision:
        """
        评估车道保持状态 → 决定子状态
        """
        # 查找前方最近车辆
        lead_vehicle = self._find_lead_vehicle(ego_state, fused_objects,
                                               lateral_threshold=2.0)

        if lead_vehicle is None:
            # 无前车 → 巡航
            return BehaviorDecision(
                state=DrivingState.CRUISING,
                target_speed=self.max_speed,
            )

        # 计算与前车的安全距离
        ego_v = ego_state.vx
        lead_v = lead_vehicle.vx
        lead_dist = abs(lead_vehicle.x - ego_state.x)

        safe_distance = ego_v * self.safe_following_time + 5.0  # +5m 最小间隙

        if lead_dist < safe_distance * 0.8:
            # 需要跟车
            target_speed = min(lead_v, self.max_speed)
            return BehaviorDecision(
                state=DrivingState.CAR_FOLLOWING,
                target_speed=target_speed,
                target_gap=safe_distance,
            )
        elif lead_v < ego_v * 0.7 and lead_dist < safe_distance * 2.0:
            # 前车明显较慢 → 考虑变道
            return BehaviorDecision(
                state=DrivingState.LANE_CHANGE_PREPARE,
                target_speed=ego_v,
                target_gap=lead_dist,
            )

        # 巡航跟随（适当调整速度）
        return BehaviorDecision(
            state=DrivingState.CRUISING,
            target_speed=min(lead_v, self.max_speed),
        )

    def _evaluate_lane_change(self, ego_state, fused_objects,
                              lane_boundaries, interaction_graph) -> BehaviorDecision:
        """
        评估变道可行性
        """
        # 检查目标车道是否有足够间隙
        target_lane_id = self._select_target_lane(ego_state, lane_boundaries)

        # 检查目标车道的安全间隙
        gaps = self._find_lane_gaps(fused_objects, target_lane_id, ego_state)

        if not gaps:
            return BehaviorDecision(
                state=DrivingState.LANE_KEEPING,
                target_speed=ego_state.vx,
                safe_to_proceed=False,
            )

        best_gap = max(gaps, key=lambda g: g[1] - g[0])  # 选最大间隙
        gap_size = best_gap[1] - best_gap[0]

        if gap_size >= self.lane_change_min_gap:
            # 安全变道
            direction = 'left' if target_lane_id == 1 else 'right'
            return BehaviorDecision(
                state=(DrivingState.LANE_CHANGE_LEFT
                       if direction == 'left'
                       else DrivingState.LANE_CHANGE_RIGHT),
                target_speed=ego_state.vx,
                target_lane_id=target_lane_id,
                time_to_execute=5.0,  # 5秒内完成
                min_safety_margin=self.lane_change_min_gap,
            )
        else:
            # 间隙不足, 等待
            return BehaviorDecision(
                state=DrivingState.LANE_KEEPING,
                target_speed=min(ego_state.vx, 15.0),  # 减速等待
                safe_to_proceed=False,
            )

    def _evaluate_intersection(self, ego_state, fused_objects,
                               interaction_graph, traffic_light_state
                               ) -> BehaviorDecision:
        """
        路口通行决策
        """
        # 红绿灯状态
        if traffic_light_state == 'red':
            return BehaviorDecision(
                state=DrivingState.STOPPING,
                target_speed=0.0,
                safe_to_proceed=False,
            )
        elif traffic_light_state == 'yellow':
            # 决策: 停还是过
            dist_to_stop_line = ego_state.x  # simplified
            stopping_distance = ego_state.vx ** 2 / (2 * 3.0)  # decel 3 m/s²

            if dist_to_stop_line < stopping_distance * 0.8:
                # 安全通过
                return BehaviorDecision(
                    state=DrivingState.INTERSECTION_CROSSING,
                    target_speed=ego_state.vx,
                    safe_to_proceed=True,
                )
            else:
                return BehaviorDecision(
                    state=DrivingState.STOPPING,
                    target_speed=0.0,
                    safe_to_proceed=False,
                )

        # 绿灯 → 检查冲突
        conflicts = interaction_graph.get('all_conflicts', [])
        if conflicts:
            most_dangerous = min(conflicts, key=lambda c: c.ttc)
            if most_dangerous.ttc < 3.0:
                return BehaviorDecision(
                    state=DrivingState.YIELDING,
                    target_speed=min(ego_state.vx, 10.0),
                    safe_to_proceed=False,
                )

        return BehaviorDecision(
            state=DrivingState.INTERSECTION_CROSSING,
            target_speed=min(ego_state.vx, 15.0),  # 限速通过
            safe_to_proceed=True,
        )

    def decide(self, ego_state, fused_objects: List,
               lane_boundaries, route_maneuvers: List[Dict],
               interaction_graph: Dict = None,
               traffic_light_state: str = 'green',
               dt: float = 0.1) -> BehaviorDecision:
        """
        行为决策主入口
        """
        # 状态持续时间累积
        self._state_history.append(self._current_state)
        self._state_duration[self._current_state] = \
            self._state_duration.get(self._current_state, 0) + dt

        # 最近的路线动作
        next_maneuver = route_maneuvers[0] if route_maneuvers else \
            {'maneuver': 'follow_lane'}

        decision = None

        # ---- 基于当前状态的决策 ----
        if self._current_state in [DrivingState.INIT, DrivingState.STANDBY]:
            if ego_state.vx < 0.1:
                decision = BehaviorDecision(
                    state=DrivingState.STANDBY,
                    target_speed=0.0,
                )
            else:
                decision = BehaviorDecision(
                    state=DrivingState.LANE_KEEPING,
                    target_speed=ego_state.vx,
                )

        elif self._current_state in [DrivingState.LANE_KEEPING,
                                     DrivingState.CRUISING,
                                     DrivingState.CAR_FOLLOWING]:

            # 路线动作触发
            if next_maneuver['maneuver'].startswith('lane_change'):
                decision = self._evaluate_lane_change(
                    ego_state, fused_objects, lane_boundaries,
                    interaction_graph
                )
            elif next_maneuver['maneuver'] in ['turn_left', 'turn_right']:
                decision = self._evaluate_intersection(
                    ego_state, fused_objects, interaction_graph,
                    traffic_light_state
                )
            else:
                decision = self._evaluate_lane_keeping(
                    ego_state, fused_objects, lane_boundaries,
                    route_maneuvers
                )

        elif self._current_state == DrivingState.LANE_CHANGE_PREPARE:
            decision = self._evaluate_lane_change(
                ego_state, fused_objects, lane_boundaries,
                interaction_graph
            )

        elif self._current_state in [DrivingState.LANE_CHANGE_LEFT,
                                     DrivingState.LANE_CHANGE_RIGHT]:
            # 变道执行中 → 监测完成
            lateral_offset = ego_state.y  # 简化
            if abs(lateral_offset) > 3.0:  # 完成变道
                decision = BehaviorDecision(
                    state=DrivingState.LANE_KEEPING,
                    target_speed=ego_state.vx,
                )
            else:
                decision = BehaviorDecision(
                    state=self._current_state,
                    target_speed=ego_state.vx,
                    target_lane_id=self._current_state ==
                    DrivingState.LANE_CHANGE_LEFT,
                )

        elif self._current_state == DrivingState.YIELDING:
            # 持续让行
            if interaction_graph and not interaction_graph.get('all_conflicts'):
                decision = BehaviorDecision(
                    state=DrivingState.LANE_KEEPING,
                    target_speed=min(ego_state.vx + 1.0, self.max_speed),
                )
            else:
                decision = BehaviorDecision(
                    state=DrivingState.YIELDING,
                    target_speed=min(ego_state.vx, 8.0),
                )

        elif self._current_state == DrivingState.STOPPING:
            if ego_state.vx < 0.1:
                decision = BehaviorDecision(
                    state=DrivingState.STANDBY,
                    target_speed=0.0,
                )
            else:
                decision = BehaviorDecision(
                    state=DrivingState.STOPPING,
                    target_speed=0.0,
                )

        elif self._current_state == DrivingState.EMERGENCY_STOP:
            # 紧急制动 → 不可逆, 直到完全停止
            decision = BehaviorDecision(
                state=DrivingState.EMERGENCY_STOP,
                target_speed=0.0,
                safe_to_proceed=False,
                confidence=0.99,
            )

        # ---- 安全检查门控 ----
        if decision is None:
            decision = BehaviorDecision(
                state=DrivingState.LANE_KEEPING,
                target_speed=ego_state.vx,
            )

        # 紧急制动门控 (最高优先级)
        if interaction_graph:
            min_ttc = interaction_graph.get('min_ttc', float('inf'))
            if min_ttc < 1.5:  # TTC < 1.5s → 紧急制动
                decision = BehaviorDecision(
                    state=DrivingState.EMERGENCY_STOP,
                    target_speed=0.0,
                    safe_to_proceed=False,
                    confidence=0.99,
                )

        # 状态转移合法性检查
        if decision.state != self._current_state:
            allowed = self.STATE_TRANSITIONS.get(self._current_state, [])
            if decision.state not in allowed:
                # 非法转移 → 保持当前状态
                decision = BehaviorDecision(
                    state=self._current_state,
                    target_speed=decision.target_speed,
                    confidence=decision.confidence * 0.5,
                )

        self._current_state = decision.state
        return decision

    def _find_lead_vehicle(self, ego_state, fused_objects,
                           lateral_threshold: float = 2.0):
        """找本车道前方最近车辆"""
        lead = None
        min_dist = float('inf')

        for obj in fused_objects:
            # 同向且在前方
            if obj.x > ego_state.x:
                lateral_dist = abs(obj.y - ego_state.y)
                if lateral_dist < lateral_threshold:
                    dist = obj.x - ego_state.x
                    if dist < min_dist:
                        min_dist = dist
                        lead = obj

        return lead

    def _select_target_lane(self, ego_state, lane_boundaries) -> int:
        """选择变道目标车道"""
        # 简化: 默认左变道 (超越慢车)
        return 1  # 0=本车道, 1=左车道, -1=右车道

    def _find_lane_gaps(self, fused_objects, target_lane_id, ego_state) -> List:
        """在目标车道中找可插入的安全间隙"""
        # 返回: [(gap_start_x, gap_end_x), ...]
        lane_objects = [obj for obj in fused_objects
                        if abs(obj.y - target_lane_id * 3.5) < 1.5]

        if not lane_objects:
            return [(ego_state.x - 100, ego_state.x + 200)]

        # 按纵向位置排序
        lane_objects.sort(key=lambda o: o.x)

        gaps = []
        prev_x = ego_state.x - 50

        for obj in lane_objects:
            gap_start = prev_x
            gap_end = obj.x - 5  # 5m 缓冲
            if gap_end - gap_start > self.lane_change_min_gap:
                gaps.append((gap_start, gap_end))
            prev_x = obj.x + 5 + obj.length

        # 最后一段
        gaps.append((prev_x, ego_state.x + 200))

        return gaps
