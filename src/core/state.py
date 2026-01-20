from pydantic import BaseModel, Field
from datetime import date, timedelta
from typing import Optional, Dict, List
from dateutil.relativedelta import relativedelta
from enum import Enum
import random


class ChildEmotionalState(str, Enum):
    """孩子的情绪状态"""
    HAPPY = "开心"
    CALM = "平静"
    SAD = "难过"
    ANXIOUS = "焦虑"
    ANGRY = "愤怒"
    CURIOUS = "好奇"
    BORED = "无聊"
    CONFIDENT = "自信"
    INSECURE = "不安"
    EXCITED = "兴奋"
    TIRED = "疲惫"


class ChildPersonality(BaseModel):
    """
    孩子的性格特质（相对稳定，会随成长缓慢变化）
    
    性格特质基于大五人格模型简化版
    """
    # 内向-外向维度 (0=内向, 1=外向)
    extroversion: float = Field(default=0.5, ge=0.0, le=1.0, description="外向程度")
    
    # 情绪稳定性 (0=敏感易波动, 1=稳定)
    emotional_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="情绪稳定性")
    
    # 好奇心/开放性 (0=保守, 1=好奇)
    curiosity: float = Field(default=0.7, ge=0.0, le=1.0, description="好奇心")
    
    # 顺从性 (0=叛逆, 1=顺从)
    agreeableness: float = Field(default=0.6, ge=0.0, le=1.0, description="顺从程度")
    
    # 自律性 (0=随性, 1=自律)
    conscientiousness: float = Field(default=0.4, ge=0.0, le=1.0, description="自律程度")
    
    # 倔强程度 (0=随和, 1=倔强)
    stubbornness: float = Field(default=0.5, ge=0.0, le=1.0, description="倔强程度")
    
    @classmethod
    def generate_random(cls) -> "ChildPersonality":
        """生成随机性格（用于初始化）"""
        return cls(
            extroversion=random.uniform(0.3, 0.7),
            emotional_stability=random.uniform(0.4, 0.6),
            curiosity=random.uniform(0.5, 0.9),
            agreeableness=random.uniform(0.4, 0.8),
            conscientiousness=random.uniform(0.2, 0.5),
            stubbornness=random.uniform(0.3, 0.7)
        )


class DevelopmentSensitivity(BaseModel):
    """
    发展敏感期（基于蒙特梭利教育理论）
    
    敏感期是孩子在特定年龄段对某些能力发展特别敏感的时期
    在敏感期进行相应的教育会事半功倍
    """
    # 语言敏感期 (0-6岁)
    language: float = Field(default=0.0, ge=0.0, le=1.0, description="语言发展敏感度")
    
    # 秩序敏感期 (1-3岁)
    order: float = Field(default=0.0, ge=0.0, le=1.0, description="秩序感敏感度")
    
    # 感官敏感期 (0-6岁)
    sensory: float = Field(default=0.0, ge=0.0, le=1.0, description="感官发展敏感度")
    
    # 动作敏感期 (0-6岁)
    movement: float = Field(default=0.0, ge=0.0, le=1.0, description="动作发展敏感度")
    
    # 社交敏感期 (2.5-6岁)
    social: float = Field(default=0.0, ge=0.0, le=1.0, description="社交能力敏感度")
    
    # 数学敏感期 (4-6岁)
    math: float = Field(default=0.0, ge=0.0, le=1.0, description="数学思维敏感度")
    
    # 阅读敏感期 (4.5-5.5岁)
    reading: float = Field(default=0.0, ge=0.0, le=1.0, description="阅读兴趣敏感度")
    
    def update_for_age(self, age: float) -> None:
        """根据年龄更新敏感期强度"""
        # 语言敏感期 (0-6岁，峰值在2-3岁)
        if age < 6:
            self.language = self._bell_curve(age, peak=2.5, width=2.0)
        else:
            self.language = max(0.0, 0.3 - (age - 6) * 0.05)
        
        # 秩序敏感期 (1-3岁，峰值在2岁)
        self.order = self._bell_curve(age, peak=2.0, width=1.0) if 0.5 < age < 4 else 0.0
        
        # 感官敏感期 (0-6岁，峰值在1.5岁)
        if age < 6:
            self.sensory = self._bell_curve(age, peak=1.5, width=2.5)
        else:
            self.sensory = 0.1
        
        # 动作敏感期 (0-6岁，峰值在1-3岁)
        if age < 6:
            self.movement = self._bell_curve(age, peak=2.0, width=2.0)
        else:
            self.movement = 0.2
        
        # 社交敏感期 (2.5-6岁)
        if 2 < age < 7:
            self.social = self._bell_curve(age, peak=4.0, width=2.0)
        else:
            self.social = 0.1 if age >= 7 else 0.0
        
        # 数学敏感期 (4-6岁)
        if 3 < age < 8:
            self.math = self._bell_curve(age, peak=5.0, width=1.5)
        else:
            self.math = 0.2 if age >= 8 else 0.0
        
        # 阅读敏感期 (4.5-7岁)
        if 4 < age < 8:
            self.reading = self._bell_curve(age, peak=5.5, width=1.5)
        else:
            self.reading = 0.3 if age >= 8 else 0.0
    
    def _bell_curve(self, x: float, peak: float, width: float) -> float:
        """钟形曲线计算敏感度"""
        import math
        return math.exp(-((x - peak) ** 2) / (2 * width ** 2))
    
    def get_active_sensitivities(self) -> Dict[str, float]:
        """获取当前活跃的敏感期（敏感度>0.5）"""
        active = {}
        if self.language > 0.5:
            active["语言"] = self.language
        if self.order > 0.5:
            active["秩序"] = self.order
        if self.sensory > 0.5:
            active["感官"] = self.sensory
        if self.movement > 0.5:
            active["动作"] = self.movement
        if self.social > 0.5:
            active["社交"] = self.social
        if self.math > 0.5:
            active["数学"] = self.math
        if self.reading > 0.5:
            active["阅读"] = self.reading
        return active


class ChildInterests(BaseModel):
    """
    孩子的兴趣爱好（会随时间和经历变化）
    """
    # 各类兴趣的强度 (0-1)
    sports: float = Field(default=0.3, ge=0.0, le=1.0, description="运动兴趣")
    music: float = Field(default=0.3, ge=0.0, le=1.0, description="音乐兴趣")
    art: float = Field(default=0.3, ge=0.0, le=1.0, description="美术兴趣")
    reading: float = Field(default=0.3, ge=0.0, le=1.0, description="阅读兴趣")
    science: float = Field(default=0.3, ge=0.0, le=1.0, description="科学兴趣")
    games: float = Field(default=0.5, ge=0.0, le=1.0, description="游戏兴趣")
    social: float = Field(default=0.4, ge=0.0, le=1.0, description="社交兴趣")
    nature: float = Field(default=0.4, ge=0.0, le=1.0, description="自然探索兴趣")
    
    def get_top_interests(self, n: int = 3) -> List[str]:
        """获取前N个最感兴趣的领域"""
        interests = {
            "运动": self.sports,
            "音乐": self.music,
            "美术": self.art,
            "阅读": self.reading,
            "科学": self.science,
            "游戏": self.games,
            "社交": self.social,
            "自然": self.nature
        }
        sorted_interests = sorted(interests.items(), key=lambda x: -x[1])
        return [name for name, _ in sorted_interests[:n]]


class ChildState(BaseModel):
    """
    孩子状态模型（增强版）
    
    包含基础属性、性格特质、情绪状态、发展敏感期和兴趣爱好
    """
    # ========== 基础属性 ==========
    birth_date: date = Field(default=date(2010, 1, 1))
    iq: int = Field(default=120, ge=0, le=200)
    knowledge: float = Field(default=0.0, ge=0.0, le=100.0)
    stress: float = Field(default=0.0, ge=0.0, le=100.0)
    physical_health: float = Field(default=100.0, ge=0.0, le=100.0, description="身体素质：0=极差，50=一般，100=非常好")
    
    # ========== 家庭关系 ==========
    father_relationship: float = Field(default=100.0, ge=0.0, le=100.0, description="与父亲的关系")
    mother_relationship: float = Field(default=100.0, ge=0.0, le=100.0, description="与母亲的关系")
    grandfather_relationship: float = Field(default=100.0, ge=0.0, le=100.0, description="与祖父的关系")
    grandmother_relationship: float = Field(default=100.0, ge=0.0, le=100.0, description="与祖母的关系")
    
    # ========== 拟人化属性（新增）==========
    # 性格特质
    personality: ChildPersonality = Field(default_factory=ChildPersonality.generate_random)
    
    # 当前情绪状态
    emotional_state: ChildEmotionalState = Field(default=ChildEmotionalState.CALM)
    
    # 情绪持续时间（周数）
    emotional_duration: int = Field(default=0, ge=0, description="当前情绪持续的周数")
    
    # 发展敏感期
    development_sensitivity: DevelopmentSensitivity = Field(default_factory=DevelopmentSensitivity)
    
    # 兴趣爱好
    interests: ChildInterests = Field(default_factory=ChildInterests)
    
    # ========== 额外拟人化属性 ==========
    # 安全感 (0-100)：影响孩子的情绪稳定性和探索欲
    security_feeling: float = Field(default=80.0, ge=0.0, le=100.0, description="安全感")
    
    # 自信心 (0-100)：影响学习效率和社交
    self_confidence: float = Field(default=60.0, ge=0.0, le=100.0, description="自信心")
    
    # 社交能力 (0-100)
    social_skill: float = Field(default=50.0, ge=0.0, le=100.0, description="社交能力")
    
    # 创造力 (0-100)
    creativity: float = Field(default=50.0, ge=0.0, le=100.0, description="创造力")
    
    # 专注力 (0-100)：影响学习效率
    focus: float = Field(default=40.0, ge=0.0, le=100.0, description="专注力")
    
    # 逆商/抗挫折能力 (0-100)
    resilience: float = Field(default=50.0, ge=0.0, le=100.0, description="抗挫折能力")
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # 记录初始IQ，用于动态变化计算
        if not hasattr(self, '_initial_iq'):
            self._initial_iq = self.iq
        if not hasattr(self, '_last_iq_update_date'):
            self._last_iq_update_date = self.birth_date
        if not hasattr(self, '_last_emotion_update'):
            self._last_emotion_update = self.birth_date
    
    def update_iq(self, current_date: date, config=None) -> None:
        """
        更新IQ值（动态变化机制）
        
        IQ随年龄增长（0-18岁），受教育和训练影响
        变化范围：初始IQ ± 10点
        
        参数:
            current_date: 当前日期
            config: 模拟配置对象
        """
        from src.core.config import get_default_config
        if config is None:
            config = get_default_config()
        
        age = self.calculate_age(current_date)
        
        # IQ主要在0-18岁期间增长
        if age > 18:
            age = 18  # 18岁后IQ基本稳定
        
        # 基础IQ增长：随年龄小幅增长（0-18岁增长约5-10点）
        age_factor = age / 18.0  # 0-1之间
        iq_growth_from_age = age_factor * 8.0  # 最多增长8点
        
        # 教育和训练影响（基于知识储备）
        # 知识储备高时，IQ增长更快
        knowledge_factor = self.knowledge / 100.0  # 0-1之间
        education_bonus = knowledge_factor * 2.0  # 最多额外增长2点
        
        # 计算目标IQ
        target_iq = self._initial_iq + iq_growth_from_age + education_bonus
        
        # 限制在合理范围内（初始IQ ± 10点）
        min_iq = max(80, self._initial_iq - 10)
        max_iq = min(200, self._initial_iq + 10)
        target_iq = max(min_iq, min(max_iq, target_iq))
        
        # 平滑更新（避免突然变化）
        if not hasattr(self, '_last_iq_update_date'):
            self._last_iq_update_date = self.birth_date
        
        # 每4周更新一次IQ（适合周级模拟）
        days_since_update = (current_date - self._last_iq_update_date).days
        if days_since_update >= 28:  # 约4周（1个月）
            # 逐步向目标IQ靠近（每次更新不超过1点）
            iq_diff = target_iq - self.iq
            if abs(iq_diff) > 0.5:
                change = max(-1.0, min(1.0, iq_diff * 0.3))  # 每次最多变化0.3点
                self.iq = max(min_iq, min(max_iq, int(self.iq + change)))
                self._last_iq_update_date = current_date
    
    def calculate_age(self, current_date: date) -> float:
        """
        计算孩子当前的年龄（精确到小数，使用relativedelta提高精度）
        
        参数:
            current_date: 当前日期
            
        返回:
            孩子的年龄，如1.5表示1岁6个月
        """
        try:
            # 使用relativedelta计算精确的年龄差
            delta = relativedelta(current_date, self.birth_date)
            age_years = delta.years
            age_months = delta.months
            age_days = delta.days
            
            # 转换为小数年龄（年）
            # 考虑月份和天数
            age = age_years + age_months / 12.0 + age_days / 365.25
            
            return max(0.0, age)
        except (ValueError, AttributeError, TypeError) as e:
            # 降级方案：使用简单计算（只捕获预期的异常）
            try:
                age = current_date.year - self.birth_date.year
                month_diff = current_date.month - self.birth_date.month
                day_diff = current_date.day - self.birth_date.day
                
                if month_diff < 0 or (month_diff == 0 and day_diff < 0):
                    age -= 1
                    if month_diff < 0:
                        month_diff += 12
                    if day_diff < 0:
                        prev_month = current_date.replace(day=1) - timedelta(days=1)
                        day_diff += prev_month.day
                
                return max(0.0, age + (month_diff * 30 + day_diff) / 365.25)
            except Exception:
                # 最后的降级方案：返回默认值
                return 0.0
    
    def get_age_group(self, current_date: date) -> str:
        """
        获取孩子当前的年龄阶段
        
        参数:
            current_date: 当前日期
            
        返回:
            年龄阶段字符串: "infant"(0-3岁), "preschool"(3-6岁), "primary"(6岁以上)
        """
        age = self.calculate_age(current_date)
        if age < 3.0:
            return "infant"
        elif 3.0 <= age < 6.0:
            return "preschool"
        else:
            return "primary"
    
    def update_emotional_state(self, stress_change: float = 0.0, 
                                relationship_change: float = 0.0,
                                event_type: str = None) -> None:
        """
        根据状态变化更新情绪
        
        参数:
            stress_change: 压力变化量
            relationship_change: 关系变化量（平均）
            event_type: 触发的事件类型
        """
        # 情绪转换概率受性格影响
        stability = self.personality.emotional_stability
        
        # 基于压力水平决定情绪
        if self.stress > 80:
            # 高压力状态
            if stability > 0.6:
                self.emotional_state = ChildEmotionalState.ANXIOUS
            else:
                # 不稳定的孩子更容易愤怒或难过
                self.emotional_state = random.choice([
                    ChildEmotionalState.ANGRY,
                    ChildEmotionalState.SAD,
                    ChildEmotionalState.ANXIOUS
                ])
        elif self.stress > 60:
            self.emotional_state = ChildEmotionalState.ANXIOUS
        elif stress_change < -5:
            # 压力明显减少，变得开心
            self.emotional_state = ChildEmotionalState.HAPPY
        elif relationship_change > 3:
            # 关系改善，变得开心或兴奋
            self.emotional_state = random.choice([
                ChildEmotionalState.HAPPY,
                ChildEmotionalState.EXCITED
            ])
        elif relationship_change < -3:
            # 关系恶化，变得难过或不安
            self.emotional_state = random.choice([
                ChildEmotionalState.SAD,
                ChildEmotionalState.INSECURE
            ])
        elif self.stress < 30 and self.physical_health > 70:
            # 低压力、健康状态好，可能好奇或平静
            age = 0  # 使用默认值，因为这里没有current_date
            if self.personality.curiosity > 0.6:
                self.emotional_state = ChildEmotionalState.CURIOUS
            else:
                self.emotional_state = ChildEmotionalState.CALM
        
        # 更新情绪持续时间
        self.emotional_duration += 1
    
    def update_development_sensitivity(self, current_date: date) -> None:
        """
        更新发展敏感期（根据年龄）
        
        参数:
            current_date: 当前日期
        """
        age = self.calculate_age(current_date)
        self.development_sensitivity.update_for_age(age)
    
    def update_personality_slowly(self, current_date: date, 
                                   positive_experience: bool = True) -> None:
        """
        缓慢更新性格特质（性格变化非常缓慢）
        
        参数:
            current_date: 当前日期
            positive_experience: 是否是积极经历
        """
        # 性格变化非常微小（每次约0.01）
        change = 0.01 if positive_experience else -0.01
        
        # 积极经历增加外向性和情绪稳定性
        if positive_experience:
            self.personality.extroversion = min(1.0, self.personality.extroversion + change * 0.5)
            self.personality.emotional_stability = min(1.0, self.personality.emotional_stability + change)
        else:
            # 负面经历可能增加内向和情绪不稳定
            self.personality.emotional_stability = max(0.0, self.personality.emotional_stability + change)
    
    def update_interests(self, action_type: str, success: bool = True) -> None:
        """
        根据活动类型更新兴趣
        
        参数:
            action_type: 行为类型
            success: 活动是否成功/愉快
        """
        change = 0.02 if success else -0.01
        
        # 行为类型到兴趣的映射
        action_interest_map = {
            "户外活动": "sports",
            "健康教育": "sports",
            "感官刺激": "music",
            "早期阅读": "reading",
            "启蒙教育": "science",
            "游戏互动": "games",
            "社交接触": "social",
            "创新活动": "science",
        }
        
        interest_attr = action_interest_map.get(action_type)
        if interest_attr and hasattr(self.interests, interest_attr):
            current = getattr(self.interests, interest_attr)
            new_value = max(0.0, min(1.0, current + change))
            setattr(self.interests, interest_attr, new_value)
    
    def update_secondary_attributes(self, action_type: str, 
                                      success: bool = True,
                                      relationship_improved: bool = False) -> None:
        """
        更新次要属性（安全感、自信心、社交能力等）
        
        参数:
            action_type: 行为类型
            success: 行为是否成功
            relationship_improved: 亲子关系是否改善
        """
        # 安全感更新
        if relationship_improved:
            self.security_feeling = min(100.0, self.security_feeling + 1.0)
        if self.stress > 70:
            self.security_feeling = max(0.0, self.security_feeling - 0.5)
        
        # 自信心更新
        if success:
            self.self_confidence = min(100.0, self.self_confidence + 0.5)
        else:
            self.self_confidence = max(0.0, self.self_confidence - 0.3)
        
        # 社交能力更新
        social_actions = ["社交接触", "游戏互动", "沟通"]
        if action_type in social_actions:
            self.social_skill = min(100.0, self.social_skill + 0.5)
        
        # 创造力更新
        creative_actions = ["创新活动", "游戏互动", "启蒙教育"]
        if action_type in creative_actions:
            self.creativity = min(100.0, self.creativity + 0.3)
        
        # 专注力随年龄自然增长，学习活动加速增长
        learning_actions = ["辅导", "简单辅导", "监督学习", "早期阅读"]
        if action_type in learning_actions:
            self.focus = min(100.0, self.focus + 0.4)
        
        # 抗挫折能力
        if not success:
            # 失败但坚持可以增加抗挫折能力
            self.resilience = min(100.0, self.resilience + 0.2)
    
    def get_state_summary(self) -> Dict:
        """获取状态摘要（用于显示和日志）"""
        return {
            "基础属性": {
                "智商": self.iq,
                "知识储备": f"{self.knowledge:.1f}/100",
                "压力值": f"{self.stress:.1f}/100",
                "身体健康": f"{self.physical_health:.1f}/100"
            },
            "情绪状态": self.emotional_state.value if isinstance(self.emotional_state, ChildEmotionalState) else self.emotional_state,
            "性格特点": {
                "外向程度": f"{self.personality.extroversion:.2f}",
                "情绪稳定": f"{self.personality.emotional_stability:.2f}",
                "好奇心": f"{self.personality.curiosity:.2f}"
            },
            "能力属性": {
                "安全感": f"{self.security_feeling:.1f}",
                "自信心": f"{self.self_confidence:.1f}",
                "社交能力": f"{self.social_skill:.1f}",
                "创造力": f"{self.creativity:.1f}",
                "专注力": f"{self.focus:.1f}"
            },
            "兴趣爱好": self.interests.get_top_interests(3),
            "活跃敏感期": self.development_sensitivity.get_active_sensitivities()
        }


class ParentEmotionalState(str, Enum):
    """家长的情绪状态"""
    CALM = "平静"
    HAPPY = "开心"
    ANXIOUS = "焦虑"
    STRESSED = "压力大"
    FRUSTRATED = "沮丧"
    HOPEFUL = "充满希望"
    WORRIED = "担忧"
    PROUD = "骄傲"
    ANGRY = "生气"
    TIRED = "疲惫"


class FamilyMember(BaseModel):
    """
    家庭成员基础模型（增强版）
    
    包含经济属性、性格属性和情绪属性
    """
    # 基础属性
    salary: float = Field(default=0.0, ge=0.0, description="月薪")
    influence_weight: float = Field(default=1.0, ge=0.0, le=2.0, description="影响力权重")
    education_level: str = Field(default="大专", description="教育水平")
    personality: str = Field(default="普通", description="性格特点：严厉、温和、开明、传统、急躁、沉稳")
    
    # 情绪属性（新增）
    stress_level: float = Field(default=30.0, ge=0.0, le=100.0, description="压力水平：0=无压力，100=极度压力")
    emotional_state: str = Field(default="平静", description="当前情绪状态")
    patience: float = Field(default=70.0, ge=0.0, le=100.0, description="耐心程度")
    
    # 教育信心（新增）
    education_confidence: float = Field(default=60.0, ge=0.0, le=100.0, description="对教育孩子的信心")
    
    def update_stress(self, change: float) -> None:
        """更新压力水平"""
        self.stress_level = max(0.0, min(100.0, self.stress_level + change))
        
        # 压力影响情绪
        if self.stress_level > 80:
            self.emotional_state = "压力大"
        elif self.stress_level > 60:
            self.emotional_state = "焦虑"
        elif self.stress_level < 30:
            self.emotional_state = "平静"
    
    def update_patience(self, child_stress: float, relationship: float) -> None:
        """根据孩子状态更新耐心"""
        # 孩子压力大时，家长耐心会下降
        if child_stress > 70:
            self.patience = max(0.0, self.patience - 2.0)
        # 关系好时，耐心恢复
        if relationship > 70:
            self.patience = min(100.0, self.patience + 1.0)
        # 自然恢复
        self.patience = min(100.0, self.patience + 0.5)

class FamilyState(BaseModel):
    """
    家庭状态模型，包含所有核心家庭成员
    """
    father: FamilyMember = Field(default_factory=lambda: FamilyMember(salary=6000.0, influence_weight=1.0, education_level="大专", personality="严厉"))
    mother: FamilyMember = Field(default_factory=lambda: FamilyMember(salary=4000.0, influence_weight=1.0, education_level="大专", personality="温和"))
    grandfather: FamilyMember = Field(default_factory=lambda: FamilyMember(salary=2000.0, influence_weight=0.8, education_level="高中", personality="传统"))
    grandmother: FamilyMember = Field(default_factory=lambda: FamilyMember(salary=1500.0, influence_weight=0.8, education_level="初中", personality="温和"))
    family_savings: float = Field(default=50000.0, ge=0.0)
    current_date: date = Field(default=date(2010, 1, 1))
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # 初始化工资更新年份记录
        if not hasattr(self, '_last_salary_update_year'):
            self._last_salary_update_year = self.current_date.year
        # 保存初始工资（用于计算增长）
        if not hasattr(self, '_initial_salaries'):
            self._initial_salaries = {
                'father': self.father.salary,
                'mother': self.mother.salary,
                'grandfather': self.grandfather.salary,
                'grandmother': self.grandmother.salary
            }
    
    def update_financials(self, months_passed: float = 1.0, config=None) -> None:
        """
        更新家庭财务状况
        
        参数:
            months_passed: 经过的月数，可以是小数（如0.23表示1周）
            config: 模拟配置对象，用于获取通胀率和工资增长率
        """
        from src.core.config import get_default_config
        if config is None:
            config = get_default_config()
        
        year = self.current_date.year
        years_since_2010 = year - 2010
        
        # 限制年份范围，避免溢出
        years_since_2010 = min(years_since_2010, 50)  # 最多计算50年
        
        # 计算通胀率（分段计算，添加上限保护）
        if years_since_2010 <= 10:
            # 2010-2020年使用第一个通胀率
            inflation_rate = (1 + config.inflation_rate_2010_2020) ** years_since_2010
        else:
            # 2020-2030年使用第二个通胀率
            inflation_rate_2010_2020 = (1 + config.inflation_rate_2010_2020) ** 10
            inflation_rate_2020_2030 = (1 + config.inflation_rate_2020_2030) ** (years_since_2010 - 10)
            inflation_rate = inflation_rate_2010_2020 * inflation_rate_2020_2030
        
        # 限制通胀率倍数（避免异常值）
        inflation_rate = min(inflation_rate, 5.0)  # 最多5倍
        
        # 计算工资总收入（工资已经在update_salaries中更新，这里直接使用）
        total_income = (
            self.father.salary + 
            self.mother.salary + 
            self.grandfather.salary + 
            self.grandmother.salary
        ) * months_passed
        
        # 计算基本生活开销（考虑通胀）
        monthly_expenses = config.base_monthly_expenses * inflation_rate
        total_expenses = monthly_expenses * months_passed
        
        # 更新家庭存款
        self.family_savings += (total_income - total_expenses)
        
        # 确保存款不为负
        self.family_savings = max(0.0, self.family_savings)
    
    def update_salaries(self, config=None) -> None:
        """
        更新家庭成员工资（基于初始工资计算，避免累积增长）
        
        参数:
            config: 模拟配置对象，用于获取工资增长率
        """
        from src.core.config import get_default_config
        if config is None:
            config = get_default_config()
        
        year = self.current_date.year
        years_since_2010 = year - 2010
        
        # 限制年份范围，避免溢出
        years_since_2010 = min(years_since_2010, 50)
        
        if years_since_2010 > 0:
            # 确保有初始工资记录
            if not hasattr(self, '_initial_salaries'):
                self._initial_salaries = {
                    'father': 6000.0,
                    'mother': 4000.0,
                    'grandfather': 2000.0,
                    'grandmother': 1500.0
                }
            
            # 基于初始工资计算目标工资（避免累积增长）
            salary_growth_factor = (1 + config.salary_growth_rate) ** years_since_2010
            
            # 限制工资增长倍数（避免异常值）
            salary_growth_factor = min(salary_growth_factor, 3.0)  # 最多3倍
            
            # 更新各成员工资（基于初始工资）
            self.father.salary = self._initial_salaries['father'] * salary_growth_factor
            self.mother.salary = self._initial_salaries['mother'] * salary_growth_factor
            self.grandfather.salary = self._initial_salaries['grandfather'] * salary_growth_factor
            self.grandmother.salary = self._initial_salaries['grandmother'] * salary_growth_factor
            
            # 记录最后更新年份
            self._last_salary_update_year = year
        
    def advance_date(self, weeks: int = 1, config=None) -> None:
        """
        推进时间（优化版：每周更新财务，避免累积误差）
        
        参数:
            weeks: 推进的周数，默认为1周
            config: 模拟配置对象，用于财务更新
        """
        from src.core.config import get_default_config
        if config is None:
            config = get_default_config()
        
        prev_date = self.current_date
        prev_year = prev_date.year
        
        # 推进日期
        self.current_date += timedelta(weeks=weeks)
        current_year = self.current_date.year
        
        # 跨年时更新工资
        if current_year != prev_year:
            self.update_salaries(config)
        
        # 每周都更新财务（使用周数比例）
        weeks_per_month = 52.0 / 12.0  # 约4.33周/月
        months_passed = weeks / weeks_per_month
        self.update_financials(months_passed, config)