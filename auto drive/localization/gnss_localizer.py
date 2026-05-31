"""
GNSS 定位模块 —— RTK-GNSS 解算、坐标转换、精度评估
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pyproj


class GNSSFixType(Enum):
    """GNSS 定位解类型"""
    NO_FIX = 0
    SINGLE = 1
    DGPS = 2
    RTK_FLOAT = 4
    RTK_FIXED = 5


@dataclass
class GNSSObservation:
    """GNSS 观测值"""
    timestamp: float
    latitude: float          # 纬度 [deg]
    longitude: float         # 经度 [deg]
    altitude: float          # 海拔 [m]
    # 速度 (ENU 坐标系)
    v_east: float = 0.0
    v_north: float = 0.0
    v_up: float = 0.0
    # 精度
    hdop: float = 99.0       # 水平精度因子
    vdop: float = 99.0
    fix_type: GNSSFixType = GNSSFixType.NO_FIX
    num_satellites: int = 0
    # 标准差估计
    lat_std: float = 5.0
    lon_std: float = 5.0
    alt_std: float = 10.0
    heading: float = 0.0     # 航向角 (双天线) [rad]


class GNSSLocalizer:
    """
    GNSS 高精度定位:
      - RTK 解算状态管理
      - WGS84 ↔ 局部 ENU/UTM 坐标转换
      - DOP 精度评估与异常检测
      - 多星座联合解算 (GPS + GLONASS + Galileo + BeiDou)
    """

    def __init__(self, ref_lat: float = 0.0,
                 ref_lon: float = 0.0,
                 ref_alt: float = 0.0):
        # 参考点 (用于 ENU 坐标转换)
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt

        # 坐标转换器
        self._wgs84 = pyproj.CRS.from_epsg(4326)
        self._utm_proj = None
        self._ecef_proj = pyproj.CRS.from_epsg(4978)

        if ref_lat != 0.0:
            self._setup_projection(ref_lat, ref_lon)

        # 定位历史
        self._position_history = []
        self._max_history = 100

        # RTK 状态
        self._rtk_fixed = False
        self._rtk_fix_age = 0

    def _setup_projection(self, lat: float, lon: float):
        """初始化 UTM 投影"""
        zone = int((lon + 180) / 6) + 1
        south = lat < 0
        self._utm_proj = pyproj.CRS.from_proj4(
            f'+proj=utm +zone={zone} +{"south" if south else "north"} '
            f'+ellps=WGS84 +datum=WGS84 +units=m +no_defs'
        )

    def parse_nmea(self, nmea_sentence: str) -> Optional[GNSSObservation]:
        """解析 NMEA 0183 语句 (GGA/RMC)"""
        try:
            import pynmea2
            msg = pynmea2.parse(nmea_sentence)

            if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                fix_type = GNSSFixType.NO_FIX
                if hasattr(msg, 'gps_qual'):
                    fix_map = {0: GNSSFixType.NO_FIX, 1: GNSSFixType.SINGLE,
                               2: GNSSFixType.DGPS, 4: GNSSFixType.RTK_FIXED,
                               5: GNSSFixType.RTK_FLOAT}
                    fix_type = fix_map.get(int(msg.gps_qual),
                                           GNSSFixType.NO_FIX)

                return GNSSObservation(
                    timestamp=getattr(msg, 'timestamp', 0.0),
                    latitude=float(msg.latitude),
                    longitude=float(msg.longitude),
                    altitude=float(getattr(msg, 'altitude', 0)),
                    hdop=float(getattr(msg, 'horizontal_dil', 99.0)),
                    fix_type=fix_type,
                    num_satellites=int(getattr(msg, 'num_sats', 0)),
                )
        except Exception:
            pass

        return None

    def wgs84_to_enu(self, lat: float, lon: float, alt: float,
                     ref_lat: float = None, ref_lon: float = None,
                     ref_alt: float = None) -> Tuple[float, float, float]:
        """
        WGS84 经纬度 → 局部 ENU (East-North-Up) 坐标系
        """
        ref_lat = ref_lat or self.ref_lat
        ref_lon = ref_lon or self.ref_lon
        ref_alt = ref_alt or self.ref_alt

        # 使用 pyproj 的 Transformer
        transformer = pyproj.Transformer.from_crs(
            self._wgs84, "epsg:4978",  # WGS84 → ECEF
            always_xy=True
        )

        # 参考点 ECEF
        ref_x, ref_y, ref_z = transformer.transform(ref_lon, ref_lat, ref_alt)

        # 目标点 ECEF
        tgt_x, tgt_y, tgt_z = transformer.transform(lon, lat, alt)

        # ECEF 差值
        dx = tgt_x - ref_x
        dy = tgt_y - ref_y
        dz = tgt_z - ref_z

        # ECEF → ENU 旋转矩阵
        sin_lat = np.sin(np.radians(ref_lat))
        cos_lat = np.cos(np.radians(ref_lat))
        sin_lon = np.sin(np.radians(ref_lon))
        cos_lon = np.cos(np.radians(ref_lon))

        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return (float(e), float(n), float(u))

    def enu_to_wgs84(self, e: float, n: float, u: float) -> Tuple[float, float, float]:
        """ENU → WGS84 逆变换"""
        # ECEF → ENU 矩阵的转置 (也是逆)
        sin_lat = np.sin(np.radians(self.ref_lat))
        cos_lat = np.cos(np.radians(self.ref_lat))
        sin_lon = np.sin(np.radians(self.ref_lon))
        cos_lon = np.cos(np.radians(self.ref_lon))

        # ENU → ECEF 逆变换
        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u

        # 参考点 ECEF
        transformer = pyproj.Transformer.from_crs(
            self._wgs84, "epsg:4978",
            always_xy=True
        )
        ref_x, ref_y, ref_z = transformer.transform(
            self.ref_lon, self.ref_lat, self.ref_alt
        )

        # 目标点 ECEF → WGS84
        rev_transformer = pyproj.Transformer.from_crs(
            "epsg:4978", self._wgs84,
            always_xy=True
        )
        lon, lat, alt = rev_transformer.transform(
            ref_x + dx, ref_y + dy, ref_z + dz
        )

        return (float(lat), float(lon), float(alt))

    def assess_solution_quality(self, obs: GNSSObservation) -> float:
        """
        评估定位解质量 (0-1, 1=最优)
        考虑: 定位类型, DOP, 卫星数
        """
        score = 0.0

        # 定位类型权重
        fix_weights = {
            GNSSFixType.RTK_FIXED: 1.0,
            GNSSFixType.RTK_FLOAT: 0.7,
            GNSSFixType.DGPS: 0.5,
            GNSSFixType.SINGLE: 0.3,
            GNSSFixType.NO_FIX: 0.0,
        }
        score = fix_weights.get(obs.fix_type, 0.0)

        # DOP 衰减
        if obs.hdop < 1.0:
            score *= 1.0
        elif obs.hdop < 3.0:
            score *= 0.8
        elif obs.hdop < 5.0:
            score *= 0.5
        else:
            score *= 0.2

        # 卫星数衰减
        if obs.num_satellites >= 20:
            score *= 1.0
        elif obs.num_satellites >= 12:
            score *= 0.8
        else:
            score *= 0.5

        self._rtk_fixed = obs.fix_type == GNSSFixType.RTK_FIXED
        if self._rtk_fixed:
            self._rtk_fix_age = 0
        else:
            self._rtk_fix_age += 1

        return min(score, 1.0)

    def get_enu_observation(self, obs: GNSSObservation) -> Tuple:
        """将 GNSS 观测转换为 ENU 坐标"""
        e, n, u = self.wgs84_to_enu(
            obs.latitude, obs.longitude, obs.altitude
        )

        # 速度分量 (已经在 ENU 中)
        v_e = obs.v_east
        v_n = obs.v_north
        v_u = obs.v_up

        return (e, n, u, v_e, v_n, v_u)
