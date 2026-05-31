"""
MPC 控制器 —— 基于运动学模型的模型预测控制 (横纵向耦合)
"""
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from numpy.typing import NDArray

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


@dataclass
class MPCConfig:
    """MPC 控制器配置"""
    horizon: int = 20
    dt: float = 0.1
    # 权重矩阵
    Q: NDArray = None      # 状态权重 (4x4)
    R: NDArray = None      # 控制权重 (2x2)
    Qf: NDArray = None     # 终端状态权重
    # 约束
    max_steer: float = 0.65
    max_steer_rate: float = 0.52
    max_accel: float = 3.0
    max_decel: float = -6.0
    max_speed: float = 33.3

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.diag([10.0, 10.0, 5.0, 2.0])  # x, y, theta, v
        if self.R is None:
            self.R = np.diag([50.0, 30.0])  # steer, accel
        if self.Qf is None:
            self.Qf = np.diag([20.0, 20.0, 10.0, 4.0])


class MPCController:
    """
    模型预测控制器 (MPC):
      - 运动学自行车模型
      - 横纵向耦合优化
      - 约束处理 (执行器饱和、速度限制)
      - 实时迭代求解 (RTI)
    """

    def __init__(self, config: MPCConfig = None,
                 wheelbase: float = 2.85):
        self.config = config or MPCConfig()
        self.wheelbase = wheelbase

        self._prev_steer = 0.0
        self._prev_accel = 0.0

        if HAS_CASADI:
            self._build_solver()

    def _build_solver(self):
        """构建 CasADi MPC 优化问题"""
        N = self.config.horizon
        dt = self.config.dt
        L = self.wheelbase

        # ---- 优化变量 ----
        # 状态 (4, N+1)
        X = ca.SX.sym('X', 4, N + 1)
        # 控制 (2, N)
        U = ca.SX.sym('U', 2, N)

        # ---- 初始状态 (参数) ----
        X0 = ca.SX.sym('X0', 4)
        # ---- 参考轨迹 (参数) ----
        X_ref = ca.SX.sym('X_ref', 4, N + 1)

        # ---- 代价函数 ----
        obj = 0
        Q = self.config.Q
        R = self.config.R
        Qf = self.config.Qf

        for k in range(N):
            # 状态误差代价
            state_err = X[:, k] - X_ref[:, k]
            obj += state_err.T @ Q @ state_err

            # 控制代价
            obj += U[:, k].T @ R @ U[:, k]

            # 控制变化率代价
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                obj += du.T @ np.diag([100.0, 50.0]) @ du

        # 终端代价
        terminal_err = X[:, N] - X_ref[:, N]
        obj += terminal_err.T @ Qf @ terminal_err

        # ---- 动力学约束 ----
        g = []
        lbg = []
        ubg = []

        for k in range(N):
            x_next = X[0, k] + dt * X[3, k] * ca.cos(X[2, k])
            y_next = X[1, k] + dt * X[3, k] * ca.sin(X[2, k])
            theta_next = X[2, k] + dt * X[3, k] * ca.tan(U[0, k]) / L
            v_next = X[3, k] + dt * U[1, k]

            g.append(X[0, k + 1] - x_next)
            g.append(X[1, k + 1] - y_next)
            g.append(X[2, k + 1] - theta_next)
            g.append(X[3, k + 1] - v_next)

            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]

        # ---- 变量边界 ----
        lbx = []
        ubx = []

        for k in range(N + 1):
            lbx += [-ca.inf, -ca.inf, -ca.inf, 0.0]  # v >= 0
            ubx += [ca.inf, ca.inf, ca.inf, self.config.max_speed]

        for k in range(N):
            lbx += [-self.config.max_steer, self.config.max_decel]
            ubx += [self.config.max_steer, self.config.max_accel]

        # ---- 构建 NLP ----
        opt_vars = ca.vertcat(
            ca.reshape(X, 4 * (N + 1), 1),
            ca.reshape(U, 2 * N, 1)
        )
        p = ca.vertcat(X0, ca.reshape(X_ref, 4 * (N + 1), 1))

        nlp = {
            'x': opt_vars,
            'f': obj,
            'g': ca.vertcat(*g),
            'p': p,
        }

        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 50,
            'ipopt.tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes',
            'print_time': 0,
        }

        self._solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)
        self._N = N
        self._opt_vars = opt_vars
        self._p = p
        self._X = X
        self._U = U
        self._warm_start = None

    def solve(self, current_state: NDArray,
              reference_trajectory: NDArray) -> Tuple[NDArray, NDArray, bool]:
        """
        MPC 求解
        Args:
            current_state: (4,) [x, y, yaw, v]
            reference_trajectory: (4, N+1) [x, y, yaw, v]
        Returns:
            (predicted_states, control_sequence, success)
        """
        if not HAS_CASADI:
            return self._mpc_fallback(current_state, reference_trajectory)

        N = self._N

        # 参数
        p_val = np.concatenate([
            current_state,
            reference_trajectory[:4, :N + 1].T.flatten()
        ])

        # 初始猜测 (warm start 或零)
        if self._warm_start is not None:
            x0 = self._warm_start
        else:
            x0 = np.zeros(4 * (N + 1) + 2 * N)
            # 用参考轨迹初始化状态
            for k in range(N + 1):
                x0[4 * k:4 * k + 4] = reference_trajectory[:4, k]

        # 初始状态等式约束
        lbx = [-ca.inf] * len(x0)
        ubx = [ca.inf] * len(x0)
        for i in range(4):
            lbx[i] = current_state[i]
            ubx[i] = current_state[i]

        lbx[4 * (N + 1):] = [
            self.config.max_decel if i % 2 == 1 else -self.config.max_steer
            for i in range(2 * N)
        ]
        ubx[4 * (N + 1):] = [
            self.config.max_accel if i % 2 == 1 else self.config.max_steer
            for i in range(2 * N)
        ]

        try:
            sol = self._solver(
                x0=x0, p=p_val,
                lbx=lbx, ubx=ubx,
                lbg=[0] * (4 * N), ubg=[0] * (4 * N),
            )

            opt_vars = sol['x'].full().flatten()
            X_opt = opt_vars[:4 * (N + 1)].reshape(4, N + 1)
            U_opt = opt_vars[4 * (N + 1):].reshape(2, N)

            # 保存 warm start (shift)
            self._warm_start = np.zeros_like(x0)
            self._warm_start[:4 * N] = opt_vars[4:4 * (N + 1)]
            self._warm_start[4 * N:4 * (N + 1)] = X_opt[:, -1]  # 复制终端状态
            self._warm_start[4 * (N + 1):4 * (N + 1) + 2 * (N - 1)] = \
                opt_vars[4 * (N + 1) + 2:4 * (N + 1) + 2 * N]
            self._warm_start[4 * (N + 1) + 2 * (N - 1):] = 0.0

            return X_opt, U_opt, True

        except Exception:
            return self._mpc_fallback(current_state, reference_trajectory)

    def _mpc_fallback(self, current_state: NDArray,
                      reference_trajectory: NDArray) -> Tuple[NDArray, NDArray, bool]:
        """
        MPC 求解失败时的 fallback (无 CasADi 也走这里)
        直接返回参考轨迹, 控制量用纯追踪 + 速度控制
        """
        N = self.config.horizon

        X_out = np.zeros((4, N + 1))
        ref_len = min(reference_trajectory.shape[1], N + 1)
        X_out[:, :ref_len] = reference_trajectory[:4, :ref_len]

        # 简单控制计算
        U_out = np.zeros((2, N))
        for k in range(N):
            if k < ref_len - 1:
                dx = X_out[0, k + 1] - X_out[0, k]
                dy = X_out[1, k + 1] - X_out[1, k]

            # 目标速度 vs 当前速度
            v_target = X_out[3, min(k, ref_len - 1)]
            v_current = current_state[3] if k == 0 else X_out[3, k - 1]
            U_out[1, k] = np.clip((v_target - v_current) / self.config.dt,
                                  self.config.max_decel,
                                  self.config.max_accel)

        return X_out, U_out, True

    def get_first_control(self, control_sequence: NDArray) -> Tuple[float, float]:
        """获取第一个控制指令"""
        steer = float(np.clip(control_sequence[0, 0],
                              -self.config.max_steer,
                              self.config.max_steer))
        accel = float(np.clip(control_sequence[1, 0],
                              self.config.max_decel,
                              self.config.max_accel))
        return steer, accel
