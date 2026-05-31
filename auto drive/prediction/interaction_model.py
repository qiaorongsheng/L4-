"""
交互模型 —— 多智能体交互建模 (社会注意力 + 博弈论)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from numpy.typing import NDArray


@dataclass
class InteractionEdge:
    """交互边: 描述两个交通参与者之间的交互关系"""
    agent_a: int
    agent_b: int
    # 交互特征
    relative_distance: float
    relative_speed: float
    ttc: float                        # 碰撞时间
    drac: float                       # 避免碰撞所需减速度
    # 交互类型
    is_following: bool
    is_yielding: bool
    is_cutting_in: bool
    is_conflicting: bool              # 是否有潜在冲突
    interaction_strength: float       # 交互强度 [0, 1]
    # 通行优先级
    priority_a_over_b: float          # A 在 B 之前通行的概率


@dataclass
class InteractionGraph:
    """交互图: 场景级多智能体交互关系"""
    nodes: List[int]                  # 参与者 ID 列表
    edges: List[InteractionEdge]
    adjacency_matrix: NDArray         # (N, N) 邻接矩阵
    ego_id: int                       # 本车 ID


class InteractionModel:
    """
    交通参与者交互建模:
      - 交互图构建 (距离/冲突/注意力)
      - 博弈论决策 (非合作博弈 / Stackelberg)
      - 社会注意力机制 (谁在关注谁)
      - 群体行为分析
    """

    def __init__(self, interaction_radius: float = 50.0,
                 ttc_threshold: float = 5.0,
                 min_interaction_distance: float = 3.0):
        self.interaction_radius = interaction_radius
        self.ttc_threshold = ttc_threshold
        self.min_interaction_distance = min_interaction_distance

    def build_interaction_graph(self, ego_state, tracked_objects: List,
                                behavior_predictions: List = None) -> InteractionGraph:
        """
        构建场景交互图
        """
        all_objects = [ego_state] + list(tracked_objects)
        n = len(all_objects)
        ids = [getattr(obj, 'track_id', 0) for obj in all_objects]
        ids[0] = 0  # ego id

        edges = []
        adj_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                obj_i = all_objects[i]
                obj_j = all_objects[j]

                # 提取位置和速度
                if i == 0:
                    pi = np.array([obj_i.x, obj_i.y])
                    vi = np.array([obj_i.vx, obj_i.vy])
                else:
                    pi = np.array([obj_i.mean[0], obj_i.mean[1]])
                    vi = np.array([obj_i.mean[7], obj_i.mean[8]])

                if j == 0:
                    pj = np.array([obj_j.x, obj_j.y])
                    vj = np.array([obj_j.vx, obj_j.vy])
                else:
                    pj = np.array([obj_j.mean[0], obj_j.mean[1]])
                    vj = np.array([obj_j.mean[7], obj_j.mean[8]])

                # 相对距离
                rel_dist = np.linalg.norm(pi - pj)

                # 超过交互半径, 跳过
                if rel_dist > self.interaction_radius:
                    continue

                # 相对速度
                rel_vel = np.linalg.norm(vi - vj)

                # TTC (纵向)
                ttc = self._compute_ttc(pi, pj, vi, vj)

                # DRAC (避免碰撞所需减速度)
                drac = self._compute_drac(pi, pj, vi, vj)

                # 交互强度 (基于距离和 TTC)
                dist_factor = max(0, 1 - rel_dist / self.interaction_radius)
                ttc_factor = max(0, 1 - ttc / self.ttc_threshold) if ttc < self.ttc_threshold else 0.1
                strength = 0.6 * dist_factor + 0.4 * ttc_factor

                # 跟车关系
                is_following = self._is_following(pi, pj, vi, vj)

                # 让行关系
                is_yielding = False
                if ttc < 4.0 and ttc > 0:
                    # 减速的一方在让行
                    vi_mag = np.linalg.norm(vi)
                    vj_mag = np.linalg.norm(vj)
                    is_yielding = abs(vi_mag - vj_mag) > 2.0

                # 切入检测
                is_cutting = False
                if rel_dist < 15 and is_following:
                    # 横向移动趋势
                    lateral_speed_i = abs(vi[1]) if i > 0 else 0.0
                    lateral_speed_j = abs(vj[1]) if j > 0 else 0.0
                    is_cutting = lateral_speed_i > 0.3 or lateral_speed_j > 0.3

                # 冲突检测
                is_conflicting = ttc < 3.0 and ttc > 0 and rel_dist < 20

                # 优先级 (基于规则: 直行 > 转弯, 右侧来车优先)
                priority_ij = 0.5
                if is_conflicting:
                    # 根据到达冲突点的时间判断优先级
                    time_i = rel_dist / max(np.linalg.norm(vi), 0.1)
                    time_j = rel_dist / max(np.linalg.norm(vj), 0.1)
                    priority_ij = 1.0 / (1 + np.exp(-(time_j - time_i)))

                edge = InteractionEdge(
                    agent_a=ids[i],
                    agent_b=ids[j],
                    relative_distance=rel_dist,
                    relative_speed=rel_vel,
                    ttc=ttc,
                    drac=drac,
                    is_following=is_following,
                    is_yielding=is_yielding,
                    is_cutting_in=is_cutting,
                    is_conflicting=is_conflicting,
                    interaction_strength=min(strength, 1.0),
                    priority_a_over_b=priority_ij,
                )

                edges.append(edge)
                adj_matrix[i, j] = strength
                adj_matrix[j, i] = strength

        return InteractionGraph(
            nodes=ids,
            edges=edges,
            adjacency_matrix=adj_matrix,
            ego_id=0,
        )

    def _compute_ttc(self, p1, p2, v1, v2) -> float:
        """计算碰撞时间"""
        rel_pos = p2 - p1
        rel_vel = v1 - v2

        dist = np.linalg.norm(rel_pos)
        closing_speed = np.dot(rel_pos, rel_vel) / max(dist, 0.01)

        if closing_speed > 0:
            return dist / closing_speed
        return float('inf')

    def _compute_drac(self, p1, p2, v1, v2) -> float:
        """计算避免碰撞所需减速度"""
        rel_pos = p2 - p1
        rel_vel = v1 - v2
        dist = np.linalg.norm(rel_pos)
        closing_speed = np.dot(rel_pos, rel_vel) / max(dist, 0.01)

        if closing_speed > 0 and dist > 0:
            return closing_speed ** 2 / (2 * dist)
        return 0.0

    def _is_following(self, p1, p2, v1, v2) -> bool:
        """判断是否在跟车"""
        # 距离在合理范围内
        dist = np.linalg.norm(p2 - p1)
        if dist > 50 or dist < 2:
            return False

        # 方向相似
        v1_norm = v1 / max(np.linalg.norm(v1), 0.01)
        v2_norm = v2 / max(np.linalg.norm(v2), 0.01)
        angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1))

        return abs(angle) < np.radians(20)

    def analyze_ego_interactions(self, graph: InteractionGraph) -> Dict:
        """
        分析与本车有关的关键交互
        """
        ego_edges = [e for e in graph.edges
                     if e.agent_a == 0 or e.agent_b == 0]

        conflicts = [e for e in ego_edges if e.is_conflicting]
        followers = [e for e in ego_edges if e.is_following]
        cut_ins = [e for e in ego_edges if e.is_cutting_in]

        # 找出最危险的交互
        most_dangerous = None
        min_ttc = float('inf')
        for e in conflicts:
            if e.ttc < min_ttc:
                min_ttc = e.ttc
                most_dangerous = e

        return {
            "num_conflicts": len(conflicts),
            "num_followers": len(followers),
            "num_cut_ins": len(cut_ins),
            "most_dangerous_interaction": most_dangerous,
            "min_ttc": min_ttc,
            "all_conflicts": conflicts,
            "interaction_graph": graph,
        }

    def solve_stackelberg_game(self, ego_state, conflicting_agent,
                               ego_options: List[Dict],
                               agent_options: List[Dict]) -> Dict:
        """
        Stackelberg 博弈求解: 本车为领导者, 冲突方为跟随者
        用于冲突场景下的决策
        """
        best_ego_action = None
        best_utility = -float('inf')

        for ego_opt in ego_options:
            # 假设对方最优响应
            best_response_utility = -float('inf')
            best_response = None

            for agent_opt in agent_options:
                # 联合效用
                utility = self._compute_joint_utility(
                    ego_opt, agent_opt, ego_state, conflicting_agent
                )

                if utility > best_response_utility:
                    best_response_utility = utility
                    best_response = agent_opt

            if best_response_utility > best_utility:
                best_utility = best_response_utility
                best_ego_action = ego_opt

        return {
            "ego_action": best_ego_action,
            "expected_response": best_response,
            "expected_utility": best_utility,
        }

    def _compute_joint_utility(self, ego_action: Dict,
                               agent_action: Dict,
                               ego_state, agent_state) -> float:
        """
        博弈联合效用函数:
          - 安全性 (TTC, 距离)
          - 效率 (速度保持)
          - 舒适性 (加加速度)
        """
        utility = 0.0

        # 安全项
        safety = min(1.0, ego_action.get('safety_margin', 0) / 10.0)
        utility += 2.0 * safety

        # 效率项
        efficiency = ego_action.get('speed_penalty', 0)
        utility -= 0.5 * efficiency

        # 舒适性
        comfort = ego_action.get('jerk', 0)
        utility -= 0.3 * comfort

        # 对方效率 (利他项, 避免过度保守)
        agent_efficiency = agent_action.get('speed_penalty', 0)
        utility -= 0.1 * agent_efficiency

        return float(utility)
