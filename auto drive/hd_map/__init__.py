"""
HD Map 管理模块 —— 高精地图加载、查询、局部地图维护
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from numpy.typing import NDArray
import json
import pickle


@dataclass
class LaneSegment:
    """车道段 (HD Map 基本单元)"""
    segment_id: int
    lane_id: int
    # 中心线 (采样点)
    centerline: NDArray         # (N, 3) [x, y, z]
    # 左右边界
    left_boundary: NDArray      # (N, 3)
    right_boundary: NDArray     # (N, 3)
    # 属性
    speed_limit: float          # [m/s]
    lane_type: str              # 'driving', 'shoulder', 'parking', ...
    direction: int              # 1=顺向, -1=逆向
    # 拓扑
    predecessor_ids: List[int] = field(default_factory=list)
    successor_ids: List[int] = field(default_factory=list)
    left_neighbor_id: Optional[int] = None
    right_neighbor_id: Optional[int] = None
    # 交通元素
    stop_line: Optional[NDArray] = None    # (2, 3) 停止线
    crosswalk: Optional[NDArray] = None    # (N, 3) 人行横道
    traffic_light_id: Optional[int] = None


@dataclass
class MapElement:
    """地图元素 (交通标志、信号灯等)"""
    element_id: int
    element_type: str          # 'traffic_light', 'stop_sign', 'yield_sign', ...
    position: NDArray          # (3,) [x, y, z]
    orientation: float         # [rad]
    associated_lane_ids: List[int] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)


class HDMapManager:
    """
    高精地图管理器:
      - 地图加载 (OpenDRIVE / Lanelet2 / 自定义格式)
      - 空间索引查询 (R-Tree)
      - 局部地图提取 (ego-centric window)
      - 地图更新与热加载
      - 车道级拓扑查询
    """

    def __init__(self, map_path: Optional[str] = None):
        # 地图数据
        self._lane_segments: Dict[int, LaneSegment] = {}
        self._map_elements: Dict[int, MapElement] = {}
        self._lane_graph: Optional[LaneGraph] = None

        # 空间索引
        self._rtree = None
        self._segment_bboxes: Dict[int, Tuple[float, float, float, float]] = {}

        # 参考点
        self._origin_lat: float = 0.0
        self._origin_lon: float = 0.0
        self._origin_alt: float = 0.0

        if map_path:
            self.load_map(map_path)

    def load_map(self, map_path: str):
        """加载 HD Map"""
        if map_path.endswith('.json'):
            with open(map_path, 'r') as f:
                data = json.load(f)
            self._parse_map_data(data)
        elif map_path.endswith('.pkl'):
            with open(map_path, 'rb') as f:
                data = pickle.load(f)
            self._parse_map_data(data)
        else:
            raise ValueError(f"Unsupported map format: {map_path}")

    def _parse_map_data(self, data: Dict):
        """解析地图数据"""
        # 解析车道段
        for seg_data in data.get('lane_segments', []):
            seg = LaneSegment(
                segment_id=seg_data['segment_id'],
                lane_id=seg_data['lane_id'],
                centerline=np.array(seg_data['centerline']),
                left_boundary=np.array(seg_data['left_boundary']),
                right_boundary=np.array(seg_data['right_boundary']),
                speed_limit=seg_data.get('speed_limit', 16.67),
                lane_type=seg_data.get('lane_type', 'driving'),
                direction=seg_data.get('direction', 1),
                predecessor_ids=seg_data.get('predecessors', []),
                successor_ids=seg_data.get('successors', []),
                left_neighbor_id=seg_data.get('left_neighbor'),
                right_neighbor_id=seg_data.get('right_neighbor'),
            )
            self._lane_segments[seg.segment_id] = seg

            # 计算 bbox
            cl = seg.centerline
            self._segment_bboxes[seg.segment_id] = (
                float(cl[:, 0].min()), float(cl[:, 1].min()),
                float(cl[:, 0].max()), float(cl[:, 1].max()),
            )

        # 解析地图元素
        for elem_data in data.get('map_elements', []):
            elem = MapElement(
                element_id=elem_data['element_id'],
                element_type=elem_data['element_type'],
                position=np.array(elem_data['position']),
                orientation=elem_data.get('orientation', 0.0),
                associated_lane_ids=elem_data.get('associated_lanes', []),
                attributes=elem_data.get('attributes', {}),
            )
            self._map_elements[elem.element_id] = elem

        # 构建空间索引
        self._build_spatial_index()

        # 构建车道图
        self._build_lane_graph()

    def _build_spatial_index(self):
        """构建 R-Tree 空间索引"""
        try:
            from rtree import index

            self._rtree = index.Index()

            for seg_id, bbox in self._segment_bboxes.items():
                self._rtree.insert(seg_id, bbox)
        except ImportError:
            self._rtree = None

    def _build_lane_graph(self):
        """构建车道拓扑图"""
        from .lane_graph import LaneGraph
        self._lane_graph = LaneGraph()
        self._lane_graph.build_from_segments(self._lane_segments)

    def query_nearby_segments(self, position: Tuple[float, float],
                              radius: float = 100.0) -> List[LaneSegment]:
        """
        查询给定位置周围的车道段
        """
        x, y = position[:2]

        if self._rtree is not None:
            bbox = (x - radius, y - radius, x + radius, y + radius)
            nearby_ids = list(self._rtree.intersection(bbox))

            return [self._lane_sements[sid] for sid in nearby_ids
                    if sid in self._lane_segments]
        else:
            # Fallback: 线性搜索
            nearby = []
            for seg_id, seg in self._lane_segments.items():
                cl = seg.centerline
                if cl[:, 0].min() > x + radius or cl[:, 0].max() < x - radius:
                    continue
                if cl[:, 1].min() > y + radius or cl[:, 1].max() < y - radius:
                    continue
                nearby.append(seg)
            return nearby

    def get_local_map(self, ego_position: Tuple[float, float, float],
                      window_size: Tuple[float, float] = (200, 100)
                      ) -> Dict:
        """
        获取以自车为中心的局部地图窗口
        window_size: (longitudinal_range, lateral_range) [m]
        """
        x, y, z = ego_position
        long_range, lat_range = window_size

        bbox = (x - long_range / 2, y - lat_range / 2,
                x + long_range / 2, y + lat_range / 2)

        # 查询局部车道段
        local_segments = []
        for seg_id, bbox_seg in self._segment_bboxes.items():
            # bbox 相交检测
            if (bbox_seg[2] >= bbox[0] and bbox_seg[0] <= bbox[2] and
                    bbox_seg[3] >= bbox[1] and bbox_seg[1] <= bbox[3]):
                local_segments.append(self._lane_segments[seg_id])

        # 查询局部地图元素
        local_elements = []
        for elem in self._map_elements.values():
            dist = np.linalg.norm(elem.position[:2] - np.array([x, y]))
            if dist < max(long_range, lat_range):
                local_elements.append(elem)

        # 构建参考线 (本车道中心线)
        ego_lane = self._find_ego_lane(ego_position)
        reference_line = ego_lane.centerline if ego_lane else None

        return {
            'ego_lane': ego_lane,
            'local_segments': local_segments,
            'local_elements': local_elements,
            'reference_line': reference_line,
            'window_center': (x, y),
            'window_size': window_size,
        }

    def _find_ego_lane(self, ego_position: Tuple[float, float, float]
                       ) -> Optional[LaneSegment]:
        """找到本车所在车道"""
        x, y, z = ego_position

        candidates = self.query_nearby_segments((x, y), radius=10.0)

        best_seg = None
        best_dist = float('inf')

        for seg in candidates:
            if seg.lane_type != 'driving':
                continue

            # 到车道中心线的最小距离
            cl = seg.centerline
            dists = np.linalg.norm(cl[:, :2] - np.array([x, y]), axis=1)
            min_dist = dists.min()

            if min_dist < best_dist and min_dist < 5.0:
                best_dist = min_dist
                best_seg = seg

        return best_seg

    def get_speed_limit(self, position: Tuple[float, float, float]) -> float:
        """获取当前位置的限速"""
        ego_lane = self._find_ego_lane(position)
        return ego_lane.speed_limit if ego_lane else 33.3

    def get_lane_topology(self, segment_id: int) -> Dict:
        """获取车道的拓扑关系"""
        if self._lane_graph is None:
            return {}

        return self._lane_graph.query_topology(segment_id)

    def find_route_lanes(self, start_pos: Tuple[float, float, float],
                         end_pos: Tuple[float, float, float]) -> List[int]:
        """查找从起点到终点的车道序列"""
        if self._lane_graph is None:
            return []

        start_lane = self._find_ego_lane(start_pos)
        end_lane = self._find_ego_lane(end_pos)

        if start_lane is None or end_lane is None:
            return []

        return self._lane_graph.find_path(
            start_lane.segment_id, end_lane.segment_id
        )
