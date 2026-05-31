"""
运动规划模块 —— Frenet 坐标系下的轨迹生成 (Lattice Planner)
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class FrenetState:
    """Frenet 坐标系状态"""
    s: float        # 纵向弧长 [m]
    s_dot: float    # 纵向速度 [m/s]
    s_ddot: float   # 纵向加速度 [m/s²]
    d: float        # 横向偏移 [m]
    d_dot: float    # 横向速度 [m/s]
    d_ddot: float   # 横向加速度 [m/s²]


@dataclass
class MotionTrajectory:
    """运动规划轨迹"""
    # 轨迹点 (Cartesian)
    x: NDArray          # (N,) 纵向位置
    y: NDArray          # (N,) 横向位置
    heading: NDArray    # (N,) 航向角
    v: NDArray          # (N,) 速度
    kappa: NDArray      # (N,) 曲率
    # 时间
    t: NDArray          # (N,) 时间戳
    # 代价
    cost: float
    # 轨迹类型
    trajectory_type: str  # 'cruise', 'follow', 'stop', 'lane_change'


@dataclass
class MotionPlan:
    """运动规划结果"""
    best_trajectory: MotionTrajectory
    candidate_trajectories: List[MotionTrajectory]
    planning_time: float


class MotionPlanner:
    """
    Lattice 运动规划器:
      - Frenet 坐标系轨迹生成
      - 横纵向解耦采样 (多项式)
      - 碰撞检测 & 代价评估
      - 最优轨迹选择
    """

    def __init__(self, planning_horizon: float = 8.0,
                 planning_dt: float = 0.1,
                 num_lateral_samples: int = 7,
                 num_longitudinal_samples: int = 16,
                 max_lateral_offset: float = 5.0,
                 vehicle_width: float = 1.95,
                 vehicle_length: float = 4.85):
        self.planning_horizon = planning_horizon
        self.planning_dt = planning_dt
        self.num_lateral_samples = num_lateral_samples
        self.num_longitudinal_samples = num_longitudinal_samples
        self.max_lateral_offset = max_lateral_offset
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length

        self._num_steps = int(planning_horizon / planning_dt)

    def _sample_lateral_polynomials(self, initial_d: float,
                                    initial_d_dot: float,
                                    initial_d_ddot: float) -> List[Tuple]:
        """
        横向采样: 五阶多项式 d(s)
        边界条件: d(0), d'(0), d''(0) → d(S), d'(S), d''(S)
        """
        samples = []

        # 目标横向偏移
        d_targets = np.linspace(-self.max_lateral_offset,
                                self.max_lateral_offset,
                                self.num_lateral_samples)

        for d_T in d_targets:
            # 终端条件: d'(S)=0, d''(S)=0 (平滑到达目标位置)
            d_dot_T = 0.0
            d_ddot_T = 0.0

            S = self.planning_horizon * 15.0  # 纵向弧长 (假设平均 15m/s)

            # 五阶多项式系数求解
            # d(s) = a0 + a1*s + a2*s^2 + a3*s^3 + a4*s^4 + a5*s^5
            T_mat = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, S, S ** 2, S ** 3, S ** 4, S ** 5],
                [0, 1, 2 * S, 3 * S ** 2, 4 * S ** 3, 5 * S ** 4],
                [0, 0, 2, 6 * S, 12 * S ** 2, 20 * S ** 3],
            ])

            b = np.array([initial_d, initial_d_dot, initial_d_ddot,
                          d_T, d_dot_T, d_ddot_T])

            try:
                coeffs = np.linalg.solve(T_mat, b)
                samples.append((coeffs, d_T))
            except np.linalg.LinAlgError:
                continue

        return samples

    def _sample_longitudinal_polynomials(self, initial_s_dot: float,
                                         initial_s_ddot: float,
                                         target_speed: float) -> List[Tuple]:
        """
        纵向采样: 四阶多项式 s(t)
        边界条件: s'(0), s''(0) → s'(T), s''(T)
        """
        samples = []

        T = self.planning_horizon

        # 目标速度采样 (围绕目标速度)
        speed_variations = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        s_dot_targets = target_speed + speed_variations
        s_dot_targets = np.clip(s_dot_targets, 0.0, 33.3)

        for s_dot_T in s_dot_targets:
            s_ddot_T = 0.0  # 终端加速度为 0

            # 四阶多项式: s(t) = s0 + s't + 1/2*s''t² + 1/6*jerk0*t³ + 1/24*snap*t⁴
            # 求解 jerk0 和 snap
            s0 = 0.0
            A = np.array([
                [T ** 3 / 6, T ** 4 / 24],
                [T ** 2 / 2, T ** 3 / 6],
            ])
            b_vec = np.array([
                s_dot_T - initial_s_dot - initial_s_ddot * T,
                s_ddot_T - initial_s_ddot,
            ])

            try:
                jerk_snap = np.linalg.solve(A, b_vec)
                coeffs = np.array([s0, initial_s_dot, initial_s_ddot,
                                   jerk_snap[0], jerk_snap[1]])
                samples.append((coeffs, s_dot_T))
            except np.linalg.LinAlgError:
                continue

        return samples

    def _evaluate_longitudinal_polynomial(self, coeffs: NDArray,
                                          t: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """评估纵向多项式: 返回 (s, s_dot, s_ddot)"""
        s = coeffs[0] + coeffs[1] * t + 0.5 * coeffs[2] * t ** 2 + \
            coeffs[3] * t ** 3 / 6 + coeffs[4] * t ** 4 / 24
        s_dot = coeffs[1] + coeffs[2] * t + 0.5 * coeffs[3] * t ** 2 + \
            coeffs[4] * t ** 3 / 6
        s_ddot = coeffs[2] + coeffs[3] * t + 0.5 * coeffs[4] * t ** 2

        return s, s_dot, s_ddot

    def _evaluate_lateral_polynomial(self, coeffs: NDArray,
                                     s: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """评估横向多项式: 返回 (d, d_prime, d_double_prime)"""
        d = np.polyval(coeffs[::-1], s)
        d_prime = np.polyval(np.polyder(coeffs[::-1]), s)
        d_double_prime = np.polyval(np.polyder(np.polyder(coeffs[::-1])), s)

        return d, d_prime, d_double_prime

    def _frenet_to_cartesian(self, s: NDArray, d: NDArray,
                             reference_line: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Frenet → Cartesian 坐标转换
        reference_line: (M, 4) [x, y, heading, curvature]
        """
        num_pts = len(s)
        x = np.zeros(num_pts)
        y = np.zeros(num_pts)
        heading = np.zeros(num_pts)
        kappa = np.zeros(num_pts)

        for i in range(num_pts):
            # 找参考线上最近点
            ref_s = s[i]
            # 简化: 线性插值
            ref_x = np.interp(ref_s, reference_line[:, 0],
                              reference_line[:, 1])
            ref_y = np.interp(ref_s, reference_line[:, 0],
                              reference_line[:, 2])
            ref_heading = np.interp(ref_s, reference_line[:, 0],
                                    reference_line[:, 3])

            x[i] = ref_x - d[i] * np.sin(ref_heading)
            y[i] = ref_y + d[i] * np.cos(ref_heading)
            heading[i] = ref_heading
            kappa[i] = 0.0  # 简化

        return x, y, heading, kappa

    def _check_collision(self, trajectory: MotionTrajectory,
                         obstacles: List,
                         safety_margin: float = 1.0) -> bool:
        """
        基于圆盘的碰撞检测
        """
        # 车辆圆盘近似 (3 个圆)
        disc_offsets = np.array([-self.vehicle_length / 3, 0,
                                 self.vehicle_length / 3])
        disc_radius = np.sqrt((self.vehicle_length / 3) ** 2 +
                              (self.vehicle_width / 2) ** 2)

        for i in range(len(trajectory.x)):
            heading = trajectory.heading[i]
            c, s = np.cos(heading), np.sin(heading)

            for offset in disc_offsets:
                disc_x = trajectory.x[i] + offset * c
                disc_y = trajectory.y[i] + offset * s

                for obs in obstacles:
                    obs_x = getattr(obs, 'x', obs[0] if isinstance(obs, (list, tuple)) else 0)
                    obs_y = getattr(obs, 'y', obs[1] if isinstance(obs, (list, tuple)) else 0)
                    obs_radius = getattr(obs, 'width', 1.0) / 2 + safety_margin

                    dist = np.sqrt((disc_x - obs_x) ** 2 +
                                   (disc_y - obs_y) ** 2)

                    if dist < disc_radius + obs_radius:
                        return True  # 碰撞!

        return False

    def _compute_trajectory_cost(self, trajectory: MotionTrajectory,
                                 target_speed: float,
                                 reference_line: NDArray,
                                 behavior_decision) -> float:
        """
        轨迹代价函数:
          cost = w_safety * safety_cost
               + w_comfort * comfort_cost
               + w_efficiency * efficiency_cost
               + w_lateral * lateral_cost
               + w_obstacle * obstacle_cost
        """
        # 安全代价: 与障碍物的最小距离
        min_dist = float('inf')
        for obs_placeholder in []:  # 实际使用时遍历障碍物
            dists = np.sqrt((trajectory.x - 0) ** 2 +
                            (trajectory.y - 0) ** 2)
            min_dist = min(min_dist, dists.min())

        safety_cost = max(0, 1.0 - min_dist / 10.0) if min_dist < 10 else 0

        # 舒适代价: 加加速度 (jerk)
        jerk = np.diff(trajectory.v) / self.planning_dt
        comfort_cost = np.mean(jerk ** 2)

        # 效率代价: 速度偏差
        speed_error = target_speed - np.mean(trajectory.v)
        efficiency_cost = speed_error ** 2

        # 横向代价: 偏离参考线
        lateral_cost = np.mean(trajectory.y ** 2)

        # 障碍物距离代价 (来自预测轨迹)
        obstacle_cost = 0.0
        # 简化为安全距离的倒数

        # 加权求和
        total_cost = (5.0 * safety_cost +
                      2.0 * comfort_cost +
                      1.0 * efficiency_cost +
                      3.0 * lateral_cost +
                      4.0 * obstacle_cost)

        return float(total_cost)

    def plan(self, ego_state, reference_line: NDArray,
             fused_objects: List, predicted_trajectories: List,
             behavior_decision,
             lane_boundaries) -> MotionPlan:
        """
        运动规划主入口
        """
        # 当前 Frenet 状态 (简化: 假设在参考线起点)
        initial_d = 0.0  # ego_state.y (横向偏移)
        initial_d_dot = 0.0
        initial_d_ddot = 0.0
        initial_s_dot = ego_state.vx
        initial_s_ddot = 0.0

        target_speed = behavior_decision.target_speed

        # Step 1: 横向采样
        lateral_samples = self._sample_lateral_polynomials(
            initial_d, initial_d_dot, initial_d_ddot
        )

        # Step 2: 纵向采样
        longitudinal_samples = self._sample_longitudinal_polynomials(
            initial_s_dot, initial_s_ddot, target_speed
        )

        # Step 3: 横纵向组合 → 生成候选轨迹
        t = np.linspace(0, self.planning_horizon, self._num_steps)
        candidates = []

        for lat_coeffs, d_target in lateral_samples:
            for lon_coeffs, s_dot_target in longitudinal_samples:
                # 评估纵向
                s, s_dot, s_ddot = self._evaluate_longitudinal_polynomial(
                    lon_coeffs, t
                )

                # 评估横向
                d, d_prime, d_double_prime = self._evaluate_lateral_polynomial(
                    lat_coeffs, s
                )

                # Frenet → Cartesian
                x, y, heading, kappa = self._frenet_to_cartesian(
                    s, d, reference_line
                )

                # 速度
                v = np.sqrt(s_dot ** 2 + (d_prime * s_dot) ** 2)

                # 碰撞检测
                trajectory = MotionTrajectory(
                    x=x, y=y, heading=heading, v=v,
                    kappa=kappa, t=t,
                    cost=0.0,
                    trajectory_type='cruise',
                )

                if self._check_collision(trajectory, fused_objects):
                    continue

                # 代价计算
                cost = self._compute_trajectory_cost(
                    trajectory, target_speed, reference_line,
                    behavior_decision
                )

                trajectory.cost = cost
                candidates.append(trajectory)

        # Step 4: 选最优轨迹
        if not candidates:
            # 紧急制动轨迹
            stop_traj = self._generate_stop_trajectory(ego_state, t)
            return MotionPlan(
                best_trajectory=stop_traj,
                candidate_trajectories=[],
                planning_time=0.0,
            )

        best = min(candidates, key=lambda tr: tr.cost)

        return MotionPlan(
            best_trajectory=best,
            candidate_trajectories=sorted(candidates, key=lambda tr: tr.cost)[:10],
            planning_time=0.0,
        )

    def _generate_stop_trajectory(self, ego_state, t: NDArray) -> MotionTrajectory:
        """生成紧急制动轨迹"""
        v0 = ego_state.vx
        decel = -5.0  # 最大舒适减速度
        t_stop = v0 / abs(decel)

        v = np.maximum(0, v0 + decel * t)
        x = ego_state.x + v0 * t + 0.5 * decel * np.minimum(t, t_stop) ** 2
        y = np.full_like(t, ego_state.y)
        heading = np.full_like(t, 0.0)
        kappa = np.zeros_like(t)

        return MotionTrajectory(
            x=x, y=y, heading=heading, v=v, kappa=kappa, t=t,
            cost=float('inf'),
            trajectory_type='stop',
        )
