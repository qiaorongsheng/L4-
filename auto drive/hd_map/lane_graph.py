"""
车道拓扑图 —— 有向图表示的车道级路网
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class LaneGraphNode:
    """车道图节点"""
    node_id: int
    segment_id: int
    lane_id: int
    x: float
    y: float
    z: float
    speed_limit: float
    lane_type: str


class LaneGraph:
    """
    车道级拓扑图:
      - 有向图 (nx.DiGraph)
      - 车道连接关系
      - 变道关系
      - 路径搜索
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self._nodes: Dict[int, LaneGraphNode] = {}
        self._lane_to_nodes: Dict[int, List[int]] = {}  # lane_id → [node_ids]

    def build_from_segments(self, lane_segments: Dict):
        """
        从车道段构建拓扑图
        """
        for seg_id, seg in lane_segments.items():
            # 为每个段创建节点 (取中心线中点作为节点位置)
            cl = seg.centerline
            mid_idx = len(cl) // 2
            x, y, z = cl[mid_idx]

            node = LaneGraphNode(
                node_id=seg_id,
                segment_id=seg_id,
                lane_id=seg.lane_id,
                x=float(x), y=float(y), z=float(z),
                speed_limit=seg.speed_limit,
                lane_type=seg.lane_type,
            )

            self._nodes[seg_id] = node
            self._graph.add_node(seg_id, **node.__dict__)

            if seg.lane_id not in self._lane_to_nodes:
                self._lane_to_nodes[seg.lane_id] = []
            self._lane_to_nodes[seg.lane_id].append(seg_id)

        # 添加边 (纵向连接 + 横向变道)
        for seg_id, seg in lane_segments.items():
            # 纵向连接 (successors)
            for succ_id in seg.successor_ids:
                if succ_id in self._nodes:
                    dist = self._compute_node_distance(seg_id, succ_id)
                    self._graph.add_edge(
                        seg_id, succ_id,
                        edge_type='longitudinal',
                        weight=dist / self._nodes[succ_id].speed_limit,
                        distance=dist,
                    )

            # 横向连接 (变道)
            if seg.left_neighbor_id and seg.left_neighbor_id in self._nodes:
                self._graph.add_edge(
                    seg_id, seg.left_neighbor_id,
                    edge_type='lane_change_left',
                    weight=3.0,  # 变道代价 ~3秒
                    distance=0.0,
                )

            if seg.right_neighbor_id and seg.right_neighbor_id in self._nodes:
                self._graph.add_edge(
                    seg_id, seg.right_neighbor_id,
                    edge_type='lane_change_right',
                    weight=3.0,
                    distance=0.0,
                )

    def _compute_node_distance(self, node_a: int, node_b: int) -> float:
        """计算两个节点间的欧氏距离"""
        na = self._nodes[node_a]
        nb = self._nodes[node_b]
        return np.sqrt((na.x - nb.x) ** 2 + (na.y - nb.y) ** 2)

    def find_path(self, start_id: int, goal_id: int) -> List[int]:
        """
        A* 最短路径搜索
        """
        if start_id not in self._graph or goal_id not in self._graph:
            return []

        try:
            path = nx.astar_path(
                self._graph, start_id, goal_id,
                heuristic=lambda u, v: self._compute_node_distance(u, v),
                weight='weight',
            )
            return path
        except nx.NetworkXNoPath:
            return []

    def get_successors(self, node_id: int) -> List[int]:
        """获取节点的后继"""
        if node_id in self._graph:
            return list(self._graph.successors(node_id))
        return []

    def get_predecessors(self, node_id: int) -> List[int]:
        """获取节点的前驱"""
        if node_id in self._graph:
            return list(self._graph.predecessors(node_id))
        return []

    def get_neighbors(self, node_id: int) -> Dict[str, Optional[int]]:
        """获取节点的邻居车道"""
        node = self._nodes.get(node_id)
        if node is None:
            return {}

        neighbors = {'left': None, 'right': None}

        for neighbor_id in self._graph.neighbors(node_id):
            edge = self._graph[node_id][neighbor_id]
            if edge.get('edge_type') == 'lane_change_left':
                neighbors['left'] = neighbor_id
            elif edge.get('edge_type') == 'lane_change_right':
                neighbors['right'] = neighbor_id

        return neighbors

    def query_topology(self, segment_id: int) -> Dict:
        """查询拓扑关系"""
        return {
            'segment_id': segment_id,
            'predecessors': self.get_predecessors(segment_id),
            'successors': self.get_successors(segment_id),
            'neighbors': self.get_neighbors(segment_id),
        }

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
