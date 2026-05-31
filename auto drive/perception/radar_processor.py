"""
毫米波雷达处理模块 —— 多目标检测、速度估计、杂波过滤
"""
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks
from numpy.typing import NDArray


@dataclass
class RadarTarget:
    """雷达检测目标"""
    target_id: int
    range: float            # 距离 [m]
    azimuth: float          # 方位角 [rad]
    elevation: float        # 俯仰角 [rad]
    range_rate: float       # 径向速度 [m/s] (多普勒)
    rcs: float              # 雷达截面积 [dBsm]
    snr: float              # 信噪比 [dB]
    existence_prob: float   # 存在概率 [0,1]


@dataclass
class RadarTrack:
    """雷达跟踪目标 (带卡尔曼滤波)"""
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float = 0.0
    ay: float = 0.0
    age: int = 0
    missed: int = 0


class RadarProcessor:
    """
    毫米波雷达信号处理与目标跟踪:
      - CFAR (恒虚警率) 目标检测
      - DOA (波达方向) 估计
      - 多普勒速度解算
      - 卡尔曼滤波跟踪
    """

    def __init__(self, num_rx: int = 4, num_tx: int = 3,
                 fc: float = 77e9, bandwidth: float = 4e9,
                 chirp_time: float = 40e-6, frame_time: float = 50e-3):
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.fc = fc                          # 载频 [Hz]
        self.bandwidth = bandwidth             # 带宽 [Hz]
        self.chirp_time = chirp_time          # chirp 时长 [s]
        self.frame_time = frame_time           # 帧时间 [s]
        self.c = 3e8                           # 光速 [m/s]

        # 分辨率
        self.range_resolution = self.c / (2 * bandwidth)
        self.velocity_resolution = self.c / (2 * fc * frame_time)
        self.max_range = self.c * bandwidth / (2 * chirp_time * 1e6)  # 近似
        self.max_velocity = self.c / (4 * fc * chirp_time)

        # 卡尔曼滤波跟踪器
        self._tracks: Dict[int, RadarTrack] = {}
        self._next_track_id = 0
        self._track_max_age = 8
        self._track_min_confirmed = 3

    def range_doppler_map(self, adc_data: NDArray) -> NDArray:
        """
        2D-FFT 生成距离-多普勒图
        adc_data: (num_chirps, num_samples_per_chirp, num_rx)
        返回: (num_range_bins, num_doppler_bins) 功率谱
        """
        # Range FFT (沿采样点维度)
        range_fft = np.fft.fft(adc_data, axis=1)
        range_fft = np.fft.fftshift(range_fft, axes=1)

        # Doppler FFT (沿 chirp 维度)
        doppler_fft = np.fft.fft(range_fft, axis=0)
        doppler_fft = np.fft.fftshift(doppler_fft, axes=0)

        # 功率谱 (多通道非相干累积)
        rd_map = np.sum(np.abs(doppler_fft) ** 2, axis=2)

        return rd_map

    def cfar_detection(self, rd_map: NDArray,
                       pfa: float = 1e-6,
                       guard_cells: Tuple[int, int] = (4, 4),
                       training_cells: Tuple[int, int] = (12, 12)
                       ) -> List[Tuple[int, int, float]]:
        """
        2D CA-CFAR 目标检测
        返回: [(range_idx, doppler_idx, snr), ...]
        """
        nr, nd = rd_map.shape
        gr, gd = guard_cells
        tr, td = training_cells

        # 训练单元总数 (排除保护单元)
        n_train = (2 * tr + 2 * gr + 1) * (2 * td + 2 * gd + 1) - \
                  (2 * gr + 1) * (2 * gd + 1)

        # CFAR 阈值因子
        alpha = n_train * (pfa ** (-1 / n_train) - 1)

        detections = []

        # 滑动窗口 CFAR
        for r in range(tr + gr, nr - tr - gr):
            for d in range(td + gd, nd - td - gd):
                # 训练区域 (排除保护区域)
                window = rd_map[
                    r - tr - gr : r + tr + gr + 1,
                    d - td - gd : d + td + gd + 1
                ].copy()
                window[tr:tr + 2 * gr + 1, td:td + 2 * gd + 1] = 0  # 排除保护格
                noise_estimate = window.sum() / n_train

                threshold = alpha * noise_estimate
                cell_value = rd_map[r, d]

                if cell_value > threshold:
                    snr = 10 * np.log10(cell_value / max(noise_estimate, 1e-10))
                    detections.append((r, d, snr))

        return detections

    def angle_estimation(self, range_doppler_cube: NDArray,
                         detections: List[Tuple[int, int, float]]
                         ) -> List[RadarTarget]:
        """
        基于 FFT 的到达角 (AOA) 估计
        """
        targets = []

        for ridx, didx, snr in detections:
            # 提取所有虚拟天线在该距离-多普勒 bin 的数据
            steering_vector = range_doppler_cube[didx, ridx, :]

            # 角度 FFT (零填充提高分辨率)
            angle_fft_size = 256
            angle_spectrum = np.fft.fft(steering_vector, n=angle_fft_size)
            angle_spectrum = np.abs(angle_spectrum)

            # 找峰值
            peaks, properties = find_peaks(
                angle_spectrum,
                height=angle_spectrum.max() * 0.3,
                distance=5
            )

            if len(peaks) == 0:
                continue

            best_peak = peaks[np.argmax(properties['peak_heights'])]
            azimuth = np.arcsin(2 * best_peak / angle_fft_size - 1)

            # 距离
            _range = ridx * self.range_resolution

            # 径向速度
            range_rate = ((didx - range_doppler_cube.shape[0] // 2) *
                          self.velocity_resolution)

            # RCS 估计
            rcs = snr + 10 * np.log10(_range ** 4)

            targets.append(RadarTarget(
                target_id=0,  # 后续分配
                range=_range,
                azimuth=azimuth,
                elevation=0.0,
                range_rate=range_rate,
                rcs=rcs,
                snr=snr,
                existence_prob=min(1.0, snr / 20.0),
            ))

        return targets

    def update_tracks(self, targets: List[RadarTarget], dt: float) -> List[RadarTrack]:
        """
        卡尔曼滤波多目标跟踪 (最近邻数据关联)
        状态: [x, y, vx, vy, ax, ay]
        """
        from filterpy.kalman import KalmanFilter

        # 预测所有现有航迹
        for tid, track in list(self._tracks.items()):
            track.age += 1
            track.missed += 1
            # 简单运动模型外推
            track.x += track.vx * dt + 0.5 * track.ax * dt ** 2
            track.y += track.vy * dt + 0.5 * track.ay * dt ** 2
            track.vx += track.ax * dt
            track.vy += track.ay * dt

        # 数据关联 (最近邻)
        matched_tracks = set()
        matched_targets = set()

        for tidx, target in enumerate(targets):
            # 极坐标 → 笛卡尔坐标 (假设 elevation=0)
            tx = target.range * np.cos(target.azimuth)
            ty = target.range * np.sin(target.azimuth)

            best_tid = None
            best_dist = 3.0  # 关联门限 [m]

            for tid, track in self._tracks.items():
                if tid in matched_tracks:
                    continue
                dist = np.sqrt((tx - track.x) ** 2 + (ty - track.y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                # 更新航迹
                track = self._tracks[best_tid]
                alpha = 0.6  # 平滑系数
                track.x = track.x * (1 - alpha) + tx * alpha
                track.y = track.y * (1 - alpha) + ty * alpha
                track.vx = track.vx * (1 - alpha) + target.range_rate * np.cos(target.azimuth) * alpha
                track.vy = track.vy * (1 - alpha) + target.range_rate * np.sin(target.azimuth) * alpha
                track.missed = 0
                matched_tracks.add(best_tid)
                matched_targets.add(tidx)
            else:
                # 新航迹初始化
                self._tracks[self._next_track_id] = RadarTrack(
                    track_id=self._next_track_id,
                    x=tx, y=ty,
                    vx=target.range_rate * np.cos(target.azimuth),
                    vy=target.range_rate * np.sin(target.azimuth),
                    age=0, missed=0,
                )
                self._next_track_id += 1

        # 删除丢失的航迹
        for tid in list(self._tracks.keys()):
            if self._tracks[tid].missed > self._track_max_age:
                del self._tracks[tid]

        # 只返回确认的航迹
        confirmed = [t for t in self._tracks.values()
                     if t.missed == 0 and t.age >= self._track_min_confirmed]

        return confirmed

    def process(self, adc_data: NDArray, dt: float = 0.05) -> Dict:
        """
        完整雷达处理管线
        """
        # Step 1: 距离-多普勒图
        rd_map = self.range_doppler_map(adc_data)

        # Step 2: CFAR 检测
        detections = self.cfar_detection(rd_map)

        # Step 3: 角度估计
        rd_cube = np.expand_dims(adc_data, axis=-1)  # 简化
        targets = self.angle_estimation(rd_cube, detections)

        # Step 4: 多目标跟踪
        tracks = self.update_tracks(targets, dt)

        return {
            "range_doppler_map": rd_map,
            "detections": detections,
            "targets": targets,
            "tracks": tracks,
            "num_tracks": len(tracks),
        }
