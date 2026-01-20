"""
配置常量
集中管理所有魔法数字和配置常量
"""
from typing import Dict, Tuple

# ==========================================
# 状态管理常量
# ==========================================

# 状态变化上限（单次变化的最大值）
STATE_CHANGE_LIMITS: Dict[str, float] = {
    "knowledge": 10.0,
    "stress": 15.0,
    "physical_health": 8.0,
    "father_relationship": 5.0,
    "mother_relationship": 5.0,
    "grandfather_relationship": 5.0,
    "grandmother_relationship": 5.0,
    "iq": 2.0
}

# 状态边界
STATE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "knowledge": (0.0, 100.0),
    "stress": (0.0, 100.0),
    "physical_health": (0.0, 100.0),
    "father_relationship": (0.0, 100.0),
    "mother_relationship": (0.0, 100.0),
    "grandfather_relationship": (0.0, 100.0),
    "grandmother_relationship": (0.0, 100.0),
    "iq": (0, 200),
    "family_savings": (0.0, float('inf'))
}

# ==========================================
# 高考评价常量
# ==========================================

# 高考分数参数
GAOKAO_BASE_SCORE: int = 450
GAOKAO_MAX_SCORE: int = 750

# 大学录取分数线
UNIVERSITY_THRESHOLDS: Dict[str, int] = {
    "TOP_985": 680,
    "TOP_211": 650,
    "ORDINARY_UNIVERSITY": 550,
    "JUNIOR_COLLEGE": 400,
}

# 高考分数计算权重（默认值）
DEFAULT_GAOKAO_WEIGHTS: Dict[str, float] = {
    "knowledge": 0.40,
    "iq": 0.20,
    "stress": -0.15,  # 负面影响
    "health": 0.10,
    "family_investment": 0.15,
}

# 综合评价权重（默认值）
DEFAULT_EVALUATION_WEIGHTS: Dict[str, float] = {
    "hard_metrics": 0.6,
    "soft_metrics": 0.4,
}

# ==========================================
# 事件系统常量
# ==========================================

# 事件链最大年龄（周）
MAX_CHAIN_AGE_WEEKS: int = 52

# 事件影响衰减因子
EVENT_DECAY_FACTOR: float = 0.1  # 每个后续事件衰减10%
EVENT_MIN_DECAY: float = 0.5  # 最低50%影响

# ==========================================
# 年龄阶段常量
# ==========================================

# 年龄阶段阈值
AGE_INFANT_MAX: float = 3.0
AGE_PRESCHOOL_MAX: float = 6.0

# IQ更新间隔（天）
IQ_UPDATE_INTERVAL_DAYS: int = 28  # 约4周（1个月），适合周级模拟

# IQ变化范围
IQ_MIN_CHANGE: float = -1.0
IQ_MAX_CHANGE: float = 1.0
IQ_CHANGE_RATE: float = 0.3  # 每次更新最多变化30%

# ==========================================
# 压力崩溃常量
# ==========================================

# 压力崩溃阈值
STRESS_BREAKDOWN_THRESHOLD: float = 90.0

# 崩溃影响
BREAKDOWN_EFFECTS: Dict[str, float] = {
    "knowledge": -5.0,
    "stress": -15.0,
    "father_relationship": -3.0,
    "mother_relationship": -3.0,
    "grandfather_relationship": -2.0,
    "grandmother_relationship": -2.0,
    "physical_health": -5.0
}

# ==========================================
# UI常量
# ==========================================

# UI刷新间隔（秒）
UI_REFRESH_INTERVAL: int = 15

# UI缓存时间（秒）
UI_CACHE_TTL: int = 60

# UI数据加载批次大小
UI_BATCH_SIZE: int = 100

# ==========================================
# 内存管理常量
# ==========================================

# 内存清理间隔（秒）
MEMORY_CLEANUP_INTERVAL: float = 3600.0  # 1小时

# 最大内存使用（MB，可选）
MEMORY_MAX_USAGE_MB: float = 2048.0  # 2GB

# ==========================================
# 并发控制常量
# ==========================================

# 默认最大并发数
DEFAULT_MAX_CONCURRENT: int = 10

# 速率限制默认值
DEFAULT_RATE_LIMIT_MAX_CALLS: int = 100
DEFAULT_RATE_LIMIT_WINDOW: float = 60.0  # 秒

# 错误隔离默认值
DEFAULT_ERROR_ISOLATION_MAX_FAILURES: int = 5
DEFAULT_ERROR_ISOLATION_WINDOW: float = 60.0  # 秒

# ==========================================
# 日志常量
# ==========================================

# 日志文件默认大小（MB）
DEFAULT_LOG_MAX_SIZE_MB: float = 100.0

# 日志备份默认数量
DEFAULT_LOG_MAX_BACKUPS: int = 10
