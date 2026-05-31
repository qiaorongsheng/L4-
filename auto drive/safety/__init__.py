"""
安全监控模块 —— 多层级安全监控与故障响应
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from numpy.typing import NDArray


class SafetyLevel(Enum):
    """安全等级"""
    NOMINAL = auto()          # 正常运行
    CAUTION = auto()          # 注意 (轻微异常)
    WARNING = auto()          # 警告 (性能下降)
    DEGRADED = auto()         # 降级运行 (限速)
    MINIMAL_RISK = auto()     # 最小风险状态 (MRM)
    EMERGENCY = auto()        # 紧急制动


class FaultType(Enum):
    """故障类型"""
    SENSOR_FAILURE = "sensor_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    LOCALIZATION_LOSS = "localization_loss"
    PERCEPTION_DEGRADATION = "perception_degradation"
    PLANNING_FAILURE = "planning_failure"
    CONTROL_ERROR = "control_error"
    COMMUNICATION_LOSS = "communication_loss"
    ODD_VIOLATION = "odd_violation"
    SOFTWARE_EXCEPTION = "software_exception"


@dataclass
class SafetyCheck:
    """单项安全检查结果"""
    check_name: str
    passed: bool
    severity: float         # 0=安全, 1=危险
    detail: str = ""
    timestamp: float = 0.0


@dataclass
class SafetyAssessment:
    """综合安全评估"""
    level: SafetyLevel
    overall_score: float           # 0=完全安全, 1=立即危险
    checks: List[SafetyCheck]
    faults: List[FaultType]
    # MRM 触发
    trigger_mrm: bool = False
    mrm_type: str = ""             # 'safe_stop', 'pull_over', 'handover'
    # 时间约束
    time_to_react: float = float('inf')  # 剩余反应时间 [s]
    assessment_time: float = 0.0


class SafetyMonitor:
    """
    L4 安全监控器:
      - 多层安全壳 (Safety Shell)
      - 实时安全检查 (TTC, 横向误差, 传感器健康...)
      - 故障树分析 (FTA)
      - 最小风险策略 (MRM) 触发
      - ISO 26262 ASIL-D 理念
    """

    # 安全检查阈值
    THRESHOLDS = {
        'ttc_critical': 1.5,          # 碰撞时间紧急阈值 [s]
        'ttc_warning': 4.0,            # 碰撞时间警告阈值 [s]
        'lateral_error_max': 0.3,      # 最大横向偏差 [m]
        'speed_limit_exceed': 1.1,     # 超速比例
        'localization_uncertainty': 0.5,  # 最大定位不确定性 [m]
        'sensor_data_age_max': 0.2,    # 传感器数据最大延迟 [s]
        'planning_failure_timeout': 0.3,  # 规划失败容忍时间 [s]
        'control_error_max': 0.1,      # 控制误差阈值
    }

    def __init__(self):
        self._current_level = SafetyLevel.NOMINAL
        self._fault_history: List[Tuple[float, FaultType]] = []
        self._safety_score_history: List[float] = []
        self._mrm_active = False

    def check_ttc_safety(self, interaction_graph: Dict) -> SafetyCheck:
        """
        TTC (Time-To-Collision) 安全检查
        """
        min_ttc = interaction_graph.get('min_ttc', float('inf'))

        if min_ttc < self.THRESHOLDS['ttc_critical']:
            return SafetyCheck(
                check_name='ttc_safety',
                passed=False,
                severity=1.0,
                detail=f'Critical TTC: {min_ttc:.2f}s',
            )
        elif min_ttc < self.THRESHOLDS['ttc_warning']:
            return SafetyCheck(
                check_name='ttc_safety',
                passed=True,
                severity=0.6,
                detail=f'Warning TTC: {min_ttc:.2f}s',
            )
        else:
            return SafetyCheck(
                check_name='ttc_safety',
                passed=True,
                severity=0.0,
                detail=f'TTC OK: {min_ttc:.2f}s',
            )

    def check_lateral_safety(self, lateral_error: float,
                             max_allowed: float = None) -> SafetyCheck:
        """
        横向偏差安全检查
        """
        max_allowed = max_allowed or self.THRESHOLDS['lateral_error_max']

        severity = min(1.0, abs(lateral_error) / max_allowed)

        return SafetyCheck(
            check_name='lateral_safety',
            passed=abs(lateral_error) <= max_allowed,
            severity=severity,
            detail=f'Lateral error: {lateral_error:.3f}m (max={max_allowed}m)',
        )

    def check_speed_safety(self, current_speed: float,
                           speed_limit: float) -> SafetyCheck:
        """
        速度安全检查
        """
        ratio = current_speed / max(speed_limit, 0.1)
        severity = max(0, (ratio - 0.9) / 0.3)  # 90% 限速以上开始警告

        return SafetyCheck(
            check_name='speed_safety',
            passed=ratio <= self.THRESHOLDS['speed_limit_exceed'],
            severity=min(severity, 1.0),
            detail=f'Speed: {current_speed:.1f}/{speed_limit:.1f} m/s',
        )

    def check_localization_health(self, ekf_covariance: NDArray,
                                  gnss_quality: float) -> SafetyCheck:
        """
        定位健康检查
        """
        # 从协方差矩阵提取位置不确定性
        pos_uncertainty = np.sqrt(np.trace(ekf_covariance[:2, :2]))

        severity = min(1.0, pos_uncertainty /
                       self.THRESHOLDS['localization_uncertainty'])

        passed = (pos_uncertainty < self.THRESHOLDS['localization_uncertainty']
                  and gnss_quality > 0.3)

        return SafetyCheck(
            check_name='localization_health',
            passed=passed,
            severity=severity,
            detail=f'Pos uncertainty: {pos_uncertainty:.3f}m, '
                   f'GNSS quality: {gnss_quality:.2f}',
        )

    def check_sensor_health(self, sensor_health: Dict[str, bool],
                            sensor_data_ages: Dict[str, float]) -> List[SafetyCheck]:
        """
        传感器健康检查
        """
        checks = []

        for sensor, healthy in sensor_health.items():
            age = sensor_data_ages.get(sensor, 0)
            age_ok = age < self.THRESHOLDS['sensor_data_age_max']

            checks.append(SafetyCheck(
                check_name=f'sensor_{sensor}',
                passed=healthy and age_ok,
                severity=0.0 if healthy else 0.8,
                detail=f'{sensor}: healthy={healthy}, '
                       f'data_age={age*1000:.1f}ms',
            ))

        return checks

    def check_planning_safety(self, motion_plan, planning_success: bool,
                              planning_latency: float) -> SafetyCheck:
        """
        规划安全检查
        """
        if not planning_success:
            return SafetyCheck(
                check_name='planning_safety',
                passed=False,
                severity=0.9,
                detail='Planning failed — no valid trajectory',
            )

        if planning_latency > self.THRESHOLDS['planning_failure_timeout']:
            return SafetyCheck(
                check_name='planning_safety',
                passed=False,
                severity=0.7,
                detail=f'Planning latency: {planning_latency*1000:.0f}ms',
            )

        # 检查规划的轨迹是否包含急转弯
        max_curvature = np.max(np.abs(motion_plan.best_trajectory.kappa)) if \
            hasattr(motion_plan.best_trajectory, 'kappa') else 0

        return SafetyCheck(
            check_name='planning_safety',
            passed=max_curvature < 0.3,  # 最大曲率约 0.3 (对应转弯半径 3.3m)
            severity=min(1.0, max_curvature / 0.5),
            detail=f'Max curvature: {max_curvature:.3f}',
        )

    def check_odd_compliance(self, environment: Dict) -> SafetyCheck:
        """
        ODD (Operational Design Domain) 合规性检查
        """
        violations = []

        # 天气检查
        rainfall = environment.get('rainfall_mmh', 0)
        if rainfall > 50:
            violations.append(f'Heavy rain: {rainfall}mm/h')

        # 能见度检查
        visibility = environment.get('visibility_m', 1000)
        if visibility < 100:
            violations.append(f'Low visibility: {visibility}m')

        # 光照检查
        ambient_light = environment.get('ambient_light_lux', 1000)
        if ambient_light < 0.5:
            violations.append(f'Low light: {ambient_light}lux')

        severity = min(1.0, len(violations) / 3.0)

        return SafetyCheck(
            check_name='odd_compliance',
            passed=len(violations) == 0,
            severity=severity,
            detail='; '.join(violations) if violations else 'ODD OK',
        )

    def assess(self, interaction_graph: Dict,
               lateral_error: float,
               current_speed: float,
               speed_limit: float,
               ekf_covariance: NDArray,
               gnss_quality: float,
               sensor_health: Dict[str, bool],
               sensor_data_ages: Dict[str, float],
               motion_plan,
               planning_success: bool,
               planning_latency: float,
               environment: Dict = None) -> SafetyAssessment:
        """
        综合安全评估
        """
        checks = []

        # 1. TTC 检查
        checks.append(self.check_ttc_safety(interaction_graph))

        # 2. 横向安全
        checks.append(self.check_lateral_safety(lateral_error))

        # 3. 速度安全
        checks.append(self.check_speed_safety(current_speed, speed_limit))

        # 4. 定位健康
        checks.append(self.check_localization_health(ekf_covariance, gnss_quality))

        # 5. 传感器健康
        checks.extend(self.check_sensor_health(sensor_health, sensor_data_ages))

        # 6. 规划安全
        checks.append(self.check_planning_safety(
            motion_plan, planning_success, planning_latency
        ))

        # 7. ODD 合规
        if environment:
            checks.append(self.check_odd_compliance(environment))

        # ---- 综合评分 ----
        failed = [c for c in checks if not c.passed]
        severity_scores = [c.severity for c in checks]

        # 加权平均 (关键检查权重更高)
        weights = [2.0 if 'ttc' in c.check_name else
                   1.5 if 'lateral' in c.check_name else
                   1.0 for c in checks]
        overall_score = np.average(severity_scores, weights=weights) if checks else 0

        # ---- 确定安全等级 ----
        if overall_score > 0.8 or any(c.severity > 0.95 for c in checks):
            level = SafetyLevel.EMERGENCY
            trigger_mrm = True
            mrm_type = 'safe_stop'
        elif overall_score > 0.5:
            level = SafetyLevel.WARNING
            trigger_mrm = False
            mrm_type = ''
        elif overall_score > 0.3:
            level = SafetyLevel.CAUTION
            trigger_mrm = False
            mrm_type = ''
        else:
            level = SafetyLevel.NOMINAL
            trigger_mrm = False
            mrm_type = ''

        # 特殊条件: 定位丢失 → 降级
        loc_check = next((c for c in checks
                          if c.check_name == 'localization_health'), None)
        if loc_check and not loc_check.passed:
            level = max(level, SafetyLevel.DEGRADED)
            trigger_mrm = True
            mrm_type = 'pull_over'

        self._current_level = level
        self._safety_score_history.append(overall_score)
        if len(self._safety_score_history) > 100:
            self._safety_score_history.pop(0)

        # 时间约束
        ttc_check = checks[0]  # TTC check
        time_to_react = float('inf')
        if 'Critical' in ttc_check.detail:
            time_to_react = 1.5  # 1.5s 内必须响应

        return SafetyAssessment(
            level=level,
            overall_score=float(overall_score),
            checks=checks,
            faults=[FaultType(c.check_name) for c in failed],
            trigger_mrm=trigger_mrm,
            mrm_type=mrm_type,
            time_to_react=time_to_react,
        )
