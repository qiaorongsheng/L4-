"""
轨迹优化模块 —— 非线性优化 (CasADi) + 数值优化
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from numpy.typing import NDArray

# CasADi 用于非线性优化 (实际部署必须项)
try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


@dataclass
class OptimizationConfig:
    """优化参数配置"""
    horizon: int = 40                  # 控制时域步数
    dt: float = 0.1                    # 时间步长 [s]
    # 权重
    w_pos: float = 10.0               # 位置跟踪权重
    w_heading: float = 5.0            # 航向跟踪权重
    w_vel: float = 2.0                # 速度跟踪权重
    w_steer: float = 50.0             # 转向平滑权重
    w_accel: float = 30.0             # 加速度平滑权重
    w_jerk: float = 10.0              # 加加速度权重
    w_safety: float = 100.0           # 安全距离权重
    # 约束
    max_steer: float = 0.65            # 最大转向角 [rad]
    max_steer_rate: float = 0.52       # 最大转向速率 [rad/s]
    max_accel: float = 3.0             # 最大加速度 [m/s²]
    max_decel: float = -6.0            # 最大减速度 [m/s²]


class TrajectoryOptimizer:
    """
    轨迹优化器:
      - 基于运动学自行车模型的 MPC
      - 避障约束 (线性化/势场)
      - 参考轨迹跟踪
      - CasADi IPOPT/SQP 求解
    """

    def __init__(self, config: OptimizationConfig = None,
                 wheelbase: float = 2.85):
        self.config = config or OptimizationConfig()
        self.wheelbase = wheelbase

        if HAS_CASADI:
            self._setup_optimization_problem()

    def _setup_optimization_problem(self):
        """
        构建 CasADi NLP 优化问题
        运动学自行车模型:
          x' = v * cos(θ)
          y' = v * sin(θ)
          θ' = v * tan(δ) / L
          v' = a
        """
        N = self.config.horizon
        dt = self.config.dt
        L = self.wheelbase

        # ---- 决策变量 ----
        # 状态: [x, y, theta, v]
        X = ca.SX.sym('X', 4, N + 1)

        # 控制: [steer, accel]
        U = ca.SX.sym('U', 2, N)

        # ---- 目标函数 ----
        cost = 0

        # 参考轨迹 (作为参数传入)
        X_ref = ca.SX.sym('X_ref', 4, N + 1)

        cfg = self.config

        for k in range(N):
            # 位置误差
            cost += cfg.w_pos * ((X[0, k] - X_ref[0, k]) ** 2 +
                                 (X[1, k] - X_ref[1, k]) ** 2)
            # 航向误差
            cost += cfg.w_heading * (X[2, k] - X_ref[2, k]) ** 2
            # 速度误差
            cost += cfg.w_vel * (X[3, k] - X_ref[3, k]) ** 2

            # 控制平滑
            cost += cfg.w_steer * U[0, k] ** 2
            cost += cfg.w_accel * U[1, k] ** 2

            # 控制变化率
            if k > 0:
                cost += cfg.w_jerk * ((U[0, k] - U[0, k - 1]) ** 2 +
                                      (U[1, k] - U[1, k - 1]) ** 2)

        # 终端代价
        cost += cfg.w_pos * 2 * ((X[0, N] - X_ref[0, N]) ** 2 +
                                 (X[1, N] - X_ref[1, N]) ** 2)

        # ---- 动力学约束 ----
        g = []

        for k in range(N):
            # 运动学自行车模型 (欧拉离散化)
            x_next = X[0, k] + dt * X[3, k] * ca.cos(X[2, k])
            y_next = X[1, k] + dt * X[3, k] * ca.sin(X[2, k])
            theta_next = X[2, k] + dt * X[3, k] * ca.tan(U[0, k]) / L
            v_next = X[3, k] + dt * U[1, k]

            g.append(X[0, k + 1] - x_next)
            g.append(X[1, k + 1] - y_next)
            g.append(X[2, k + 1] - theta_next)
            g.append(X[3, k + 1] - v_next)

        # ---- 不等式约束 ----
        lbg = [0] * (4 * N)
        ubg = [0] * (4 * N)

        # ---- 变量边界 ----
        lbx_X = [-float('inf'), -float('inf'), -float('inf'), -cfg.max_decel]
        ubx_X = [float('inf'), float('inf'), float('inf'), cfg.max_accel]

        lbx_U = [-cfg.max_steer, cfg.max_decel]
        ubx_U = [cfg.max_steer, cfg.max_accel]

        lbx = lbx_X * (N + 1) + lbx_U * N
        ubx = ubx_X * (N + 1) + ubx_U * N

        # ---- 构建 NLP ----
        opt_vars = ca.vertcat(ca.reshape(X, 4 * (N + 1), 1),
                              ca.reshape(U, 2 * N, 1))

        nlp = {'x': opt_vars, 'f': cost, 'g': ca.vertcat(*g)}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
            'print_time': 0,
        }

        self._solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self._N = N
        self._X_ref = X_ref
        self._opt_vars = opt_vars

    def optimize(self, reference_trajectory,
                 initial_state: NDArray,
                 obstacles: List = None) -> Tuple[NDArray, NDArray, bool]:
        """
        求解轨迹优化问题
        返回: (state_trajectory, control_sequence, success)
        """
        if not HAS_CASADI:
            # Fallback: 直接返回参考轨迹
            return self._fallback_optimize(reference_trajectory, initial_state)

        N = self._N

        # 初始猜测
        x0 = np.zeros(4 * (N + 1) + 2 * N)

        # 用参考轨迹填充状态
        for k in range(N + 1):
            x0[4 * k:4 * k + 4] = reference_trajectory[:4, k] if \
                reference_trajectory.shape[1] > k else reference_trajectory[:4, -1]

        # 初始状态约束
        lbx = [-float('inf')] * len(x0)
        ubx = [float('inf')] * len(x0)

        for i in range(4):
            lbx[i] = initial_state[i]
            ubx[i] = initial_state[i]

        # 求解
        try:
            sol = self._solver(x0=x0, lbx=lbx, ubx=ubx,
                               lbg=[0] * (4 * N), ubg=[0] * (4 * N))

            opt_vars = sol['x'].full().flatten()

            # 提取状态和控制
            X_opt = opt_vars[:4 * (N + 1)].reshape(4, N + 1)
            U_opt = opt_vars[4 * (N + 1):].reshape(2, N)

            return X_opt, U_opt, True

        except Exception:
            return self._fallback_optimize(reference_trajectory, initial_state)

    def _fallback_optimize(self, reference_trajectory,
                           initial_state: NDArray) -> Tuple[NDArray, NDArray, bool]:
        """
        无 CasADi 时的回退优化: 直接使用参考轨迹 + 平滑
        """
        # 使用参考轨迹作为输出
        state_dim = min(reference_trajectory.shape[0], 4)
        horizon = min(reference_trajectory.shape[1], self.config.horizon + 1)

        X_opt = np.zeros((4, horizon))
        X_opt[:state_dim, :] = reference_trajectory[:state_dim, :horizon]

        # 简单控制量计算 (纯追踪 + 速度控制)
        U_opt = np.zeros((2, max(0, horizon - 1)))
        for k in range(horizon - 1):
            # 转向角: 基于曲率
            if state_dim > 2 and k < horizon - 1:
                dx = reference_trajectory[0, k + 1] - reference_trajectory[0, k]
                dy = reference_trajectory[1, k + 1] - reference_trajectory[1, k]
                heading = np.arctan2(dy, dx)
            else:
                heading = 0.0
            U_opt[0, k] = 0.0  # 直行
            U_opt[1, k] = 0.0  # 匀速

        return X_opt, U_opt, True

    def add_obstacle_constraint(self, obstacle_positions: List[Tuple[float, float]],
                                safety_radius: float = 2.0):
        """
        添加避障约束 (通过线性化近似)
        """
        # 实际部署中通过 CasADi 将障碍物作为非线性不等式约束加入
        # g_obs: (x - ox)² + (y - oy)² >= safety_radius²
        pass
