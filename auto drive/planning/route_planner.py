"""
全局路由规划模块 —— A* / Hybrid A* 在 HD Map Lane Graph 上的路由搜索
"""
import numpy as np
import heapq
import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class RoadClass(Enum):
    """道路等级"""
    HIGHWAY = 0
    URBAN_ARTERIAL = 1
    URBAN_LOCAL = 2
    RURAL = 3


@dataclass
class LaneNode:
    """车道图节点"""
    node_id: int
    x: float; y: float; z: float
    lane_id: int
    segment_id: int
    speed_limit: float          # [m/s]
    road_class: RoadClass
    lane_width: float = 3.5
    # 拓扑连接
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    left_neighbor: Optional[int] = None
    right_neighbor: Optional[int] = None


@dataclass
class RouteResult:
    """路由规划结果"""
    node_sequence: List[LaneNode]
    total_distance: float
    estimated_time: float
    maneuvers: List[Dict]        # 每个路段的驾驶动作
    alternative_routes: List[List[LaneNode]] = field(default_factory=list)


class RoutePlanner:
    """
    全局路由规划器:
      - A* 在车道图上的启发式搜索
      - 多目标优化 (距离 + 时间 + 高速偏好)
      - 车道级路由 (lane-level routing)
      - K-最短路径备选
    """

    def __init__(self, lane_graph=None):
        self._graph: Optional[nx.DiGraph] = None
        self._nodes: Dict[int, LaneNode] = {}
        self._node_positions: Dict[int, Tuple[float, float, float]] = {}

        if lane_graph is not None:
            self.load_lane_graph(lane_graph)

    def load_lane_graph(self, lane_graph):
        """加载 HD Map 车道图"""
        self._graph = nx.DiGraph()

        for node in lane_graph:
            self._nodes[node.node_id] = node
            self._node_positions[node.node_id] = (node.x, node.y, node.z)

            self._graph.add_node(
                node.node_id,
                x=node.x, y=node.y, z=node.z,
                lane_id=node.lane_id,
                speed_limit=node.speed_limit,
            )

            for succ_id in node.successors:
                # 边权重 = 路段长度 / 限速 (通行时间)
                if succ_id in self._nodes:
                    succ = self._nodes[succ_id]
                    dist = np.sqrt((succ.x - node.x) ** 2 +
                                   (succ.y - node.y) ** 2)
                    time_cost = dist / max(node.speed_limit, 1.0)
                    self._graph.add_edge(
                        node.node_id, succ_id,
                        weight=time_cost,
                        distance=dist,
                    )

    def _heuristic(self, node_id: int, goal_id: int) -> float:
        """A* 启发函数: 欧氏距离 / 最大限速"""
        if node_id not in self._node_positions or \
           goal_id not in self._node_positions:
            return 0.0

        p1 = self._node_positions[node_id][:2]
        p2 = self._node_positions[goal_id][:2]
        dist = np.linalg.norm(np.array(p1) - np.array(p2))

        max_speed = 33.3  # 120 km/h
        return dist / max_speed

    def _a_star_search(self, start_id: int, goal_id: int) -> Optional[List[int]]:
        """
        A* 搜索最优路径
        """
        if self._graph is None or start_id not in self._graph or \
           goal_id not in self._graph:
            return None

        open_set = [(0, start_id)]
        came_from = {}
        g_score = {start_id: 0}
        f_score = {start_id: self._heuristic(start_id, goal_id)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_id:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_id)
                return path[::-1]

            for neighbor in self._graph.successors(current):
                edge_data = self._graph[current][neighbor]
                tentative_g = g_score[current] + edge_data['weight']

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + \
                        self._heuristic(neighbor, goal_id)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _yens_k_shortest_paths(self, start_id: int, goal_id: int,
                               k: int = 3) -> List[List[int]]:
        """
        Yen's K-最短路径算法
        """
        if k <= 1:
            path = self._a_star_search(start_id, goal_id)
            return [path] if path else []

        shortest = self._a_star_search(start_id, goal_id)
        if shortest is None:
            return []

        A = [shortest]  # K 最短路径集
        B = []          # 候选路径 (堆)

        for k_idx in range(1, k):
            for i in range(len(A[-1]) - 1):
                spur_node = A[-1][i]
                root_path = A[-1][:i + 1]

                # 暂时移除使用过的边
                removed_edges = []
                for path in A:
                    if len(path) > i and path[:i + 1] == root_path:
                        u, v = path[i], path[i + 1]
                        if self._graph.has_edge(u, v):
                            removed_edges.append(
                                (u, v, self._graph[u][v].copy())
                            )
                            self._graph.remove_edge(u, v)

                # 从 spur_node 到目标的路径
                spur_path = self._a_star_search(spur_node, goal_id)

                # 恢复边
                for u, v, data in removed_edges:
                    self._graph.add_edge(u, v, **data)

                if spur_path and spur_path[0] == spur_node:
                    total_path = root_path[:-1] + spur_path

                    # 计算路径代价
                    cost = sum(
                        self._graph[u][v]['weight']
                        for u, v in zip(total_path[:-1], total_path[1:])
                        if self._graph.has_edge(u, v)
                    )

                    heapq.heappush(B, (cost, total_path))

            if not B:
                break

            _, next_path = heapq.heappop(B)
            A.append(next_path)

        return A

    def _extract_maneuvers(self, node_sequence: List[LaneNode]) -> List[Dict]:
        """从车道级路径中提取驾驶动作"""
        maneuvers = []

        for i in range(len(node_sequence) - 1):
            current = node_sequence[i]
            next_node = node_sequence[i + 1]

            # 判断动作类型
            if next_node.lane_id != current.lane_id:
                # 变道
                if next_node.left_neighbor == current.node_id:
                    maneuver = 'lane_change_right'
                elif next_node.right_neighbor == current.node_id:
                    maneuver = 'lane_change_left'
                elif next_node.node_id in current.successors:
                    maneuver = 'follow_lane'
                else:
                    maneuver = 'lane_change'
            else:
                maneuver = 'follow_lane'

            # 弯道检测 (基于相邻段方向变化)
            dx1 = node_sequence[min(i + 1, len(node_sequence) - 1)].x - \
                  node_sequence[i].x
            dy1 = node_sequence[min(i + 1, len(node_sequence) - 1)].y - \
                  node_sequence[i].y
            heading1 = np.arctan2(dy1, dx1)

            if i + 2 < len(node_sequence):
                dx2 = node_sequence[i + 2].x - node_sequence[i + 1].x
                dy2 = node_sequence[i + 2].y - node_sequence[i + 1].y
                heading2 = np.arctan2(dy2, dx2)
                heading_change = np.arctan2(
                    np.sin(heading2 - heading1),
                    np.cos(heading2 - heading1)
                )

                if abs(heading_change) > np.radians(15):
                    maneuver = 'turn_left' if heading_change > 0 else 'turn_right'

            maneuvers.append({
                'step': i,
                'maneuver': maneuver,
                'from_node': current.node_id,
                'to_node': next_node.node_id,
                'speed_limit': next_node.speed_limit,
                'distance': np.sqrt(
                    (next_node.x - current.x) ** 2 +
                    (next_node.y - current.y) ** 2
                ),
            })

        return maneuvers

    def plan_route(self, start_pose: Tuple[float, float, float],
                   goal_pose: Tuple[float, float, float],
                   current_lane_id: Optional[int] = None,
                   max_alternatives: int = 3) -> Optional[RouteResult]:
        """
        全局路由规划主入口
        """
        if self._graph is None:
            return None

        # 找最近的图中节点
        start_node = self._find_nearest_node(start_pose)
        goal_node = self._find_nearest_node(goal_pose)

        if start_node is None or goal_node is None:
            return None

        # K-最短路径搜索
        paths = self._yens_k_shortest_paths(
            start_node.node_id, goal_node.node_id, k=max_alternatives
        )

        if not paths:
            return None

        # 主路径
        main_path_ids = paths[0]
        main_path_nodes = [self._nodes[nid] for nid in main_path_ids]

        # 备选路径
        alt_paths = []
        for alt in paths[1:]:
            alt_paths.append([self._nodes[nid] for nid in alt])

        # 计算总距离和时间
        total_distance = 0.0
        total_time = 0.0
        for i in range(len(main_path_ids) - 1):
            u, v = main_path_ids[i], main_path_ids[i + 1]
            if self._graph.has_edge(u, v):
                total_distance += self._graph[u][v]['distance']
                total_time += self._graph[u][v]['weight']

        # 提取驾驶动作
        maneuvers = self._extract_maneuvers(main_path_nodes)

        return RouteResult(
            node_sequence=main_path_nodes,
            total_distance=total_distance,
            estimated_time=total_time,
            maneuvers=maneuvers,
            alternative_routes=alt_paths,
        )

    def _find_nearest_node(self, pose: Tuple[float, float, float]) -> Optional[LaneNode]:
        """找最近的图节点"""
        x, y, z = pose
        best_node = None
        best_dist = float('inf')

        for nid, pos in self._node_positions.items():
            dist = np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_node = self._nodes[nid]

        return best_node
