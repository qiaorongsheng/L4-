"""
相机处理模块 —— 多相机图像采集、畸变校正、BEV投影
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class CameraIntrinsic:
    """相机内参"""
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0


@dataclass
class CameraExtrinsic:
    """相机外参: 相机坐标系 → 车辆坐标系"""
    rotation: NDArray[np.float64]    # 3x3 旋转矩阵
    translation: NDArray[np.float64] # 3x1 平移向量


class CameraProcessor:
    """
    多相机图像处理管线:
      - Bayer→RGB 解马赛克
      - 畸变校正
      - IPM (Inverse Perspective Mapping) → BEV
      - 多相机 BEV 拼接
    """

    def __init__(self, intrinsics: List[CameraIntrinsic],
                 extrinsics: List[CameraExtrinsic],
                 image_size: Tuple[int, int] = (1920, 1208)):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.image_size = image_size
        self._num_cameras = len(intrinsics)

        # 预计算畸变校正映射表 (加速)
        self._undistort_maps: List[Tuple[NDArray, NDArray]] = []
        for i, K in enumerate(intrinsics):
            mtx = np.array([[K.fx, 0, K.cx],
                            [0, K.fy, K.cy],
                            [0, 0, 1]], dtype=np.float32)
            dist = np.array([K.k1, K.k2, K.p1, K.p2, K.k3], dtype=np.float32)
            map1, map2 = cv2.initUndistortRectifyMap(
                mtx, dist, None, mtx, image_size, cv2.CV_32FC1
            )
            self._undistort_maps.append((map1, map2))

        # IPM 单应矩阵 (每个相机的路面→BEV 映射)
        self._ipm_homographies: List[NDArray] = self._compute_ipm_homographies()

    def _compute_ipm_homographies(self) -> List[NDArray]:
        """计算每个相机的逆透视变换矩阵"""
        H_list = []
        for i, (K, E) in enumerate(zip(self.intrinsics, self.extrinsics)):
            mtx = np.array([[K.fx, 0, K.cx],
                            [0, K.fy, K.cy],
                            [0, 0, 1]], dtype=np.float32)

            # 相机→车辆的旋转矩阵
            R_cv = E.rotation  # camera → vehicle

            # 选取路面平面的法向量在相机坐标系中的表达
            # 假设地面在车辆坐标系中 z=0
            # H = K * [r1, r2, t]  其中 r1,r2 是旋转矩阵的前两列
            r1 = R_cv[:, 0]
            r2 = R_cv[:, 1]
            t = E.translation.flatten()

            H = mtx @ np.column_stack([r1, r2, t])
            H_list.append(H)

        return H_list

    def undistort(self, images: List[NDArray]) -> List[NDArray]:
        """对所有相机图像进行畸变校正"""
        undistorted = []
        for img, (map1, map2) in zip(images, self._undistort_maps):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            undistorted.append(cv2.remap(img, map1, map2, cv2.INTER_LINEAR))
        return undistorted

    def ipm_transform(self, images: List[NDArray],
                      bev_size: Tuple[int, int] = (512, 512),
                      bev_resolution: float = 0.1) -> NDArray:
        """
        多相机图像逆透视变换到 BEV 空间
        返回: (H_bev, W_bev, 3) BEV 图像
        """
        bev_h, bev_w = bev_size
        bev = np.zeros((bev_h, bev_w, 3), dtype=np.float32)

        for img, H in zip(images, self._ipm_homographies):
            warped = cv2.warpPerspective(
                img, H, (bev_w, bev_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            # 加权融合重叠区域
            mask = (warped.sum(axis=-1) > 0).astype(np.float32)
            alpha = 0.5 * mask[..., None]
            bev = bev * (1 - alpha) + warped.astype(np.float32) * alpha

        return bev.astype(np.uint8)

    def extract_features(self, images: List[NDArray]) -> Dict[str, NDArray]:
        """
        从原始图像提取视觉特征 (供下游检测/分割模型使用)
        实际部署替换为深度学习 backbone (RegNet / EfficientNet)
        """
        features = {}
        gray_images = []

        for i, img in enumerate(images):
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_images.append(gray)

            # 梯度特征 (边缘信息)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            direction = np.arctan2(grad_y, grad_x)

            features[f"cam_{i}_gradient_mag"] = magnitude
            features[f"cam_{i}_gradient_dir"] = direction

        return features

    def detect_lane_markings(self, bev_img: NDArray) -> NDArray:
        """
        在 BEV 图像上检测车道线标记 (传统视觉方法)
        返回: 二值车道线掩码
        """
        gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)

        # 自适应阈值 + 形态学操作
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )

        # 形态学闭运算填补断裂
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 边缘提取
        edges = cv2.Canny(closed, 50, 150)

        return edges

    def process(self, raw_images: List[NDArray]) -> Dict:
        """
        完整相机处理管线
        """
        # Step 1: 畸变校正
        undistorted = self.undistort(raw_images)

        # Step 2: BEV 投影
        bev = self.ipm_transform(undistorted)

        # Step 3: 特征提取
        features = self.extract_features(undistorted)

        # Step 4: 车道线检测
        lane_mask = self.detect_lane_markings(bev)

        return {
            "undistorted": undistorted,
            "bev": bev,
            "features": features,
            "lane_mask": lane_mask,
        }
