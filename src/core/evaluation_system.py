import math
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import date

from src.core.state import ChildState, FamilyState
from src.core.constants import (
    GAOKAO_BASE_SCORE, GAOKAO_MAX_SCORE, 
    UNIVERSITY_THRESHOLDS, DEFAULT_GAOKAO_WEIGHTS,
    DEFAULT_EVALUATION_WEIGHTS
)


class UniversityTier(Enum):
    """大学等级"""
    TOP_985 = "985高校"
    TOP_211 = "211高校"
    ORDINARY_UNIVERSITY = "普通本科"
    JUNIOR_COLLEGE = "大专"
    NO_ADMISSION = "未录取"


class DegreeLevel(Enum):
    """学历等级"""
    MASTERS_PRE_ADMISSION = "硕士预录取"
    BACHELORS = "本科"
    JUNIOR_COLLEGE = "大专"
    HIGH_SCHOOL = "高中"
    BELOW_HIGH_SCHOOL = "高中以下"


class PersonalityTrait(Enum):
    """性格特质"""
    CONFIDENT = "自信"
    OUTGOING = "外向"
    INTROVERTED = "内向"
    STRESS_RESISTANT = "抗压"
    FRAGILE = "脆弱"
    OPTIMISTIC = "乐观"
    PESSIMISTIC = "悲观"
    INDEPENDENT = "独立"
    DEPENDENT = "依赖"
    CREATIVE = "创造性"
    PRACTICAL = "务实"


@dataclass
class HardMetrics:
    """硬性指标"""
    gaokao_score: int  # 高考分数
    university_tier: UniversityTier  # 录取院校等级
    final_degree: DegreeLevel  # 最终学历
    graduation_gpa: float  # 大学GPA（如果适用）


@dataclass
class SoftMetrics:
    """软指标"""
    personality_traits: List[PersonalityTrait]  # 性格特质列表
    happiness_index: float  # 幸福指数 (0-100)
    social_adaptability: float  # 社交适应度 (0-100)
    emotional_stability: float  # 情绪稳定性 (0-100)
    life_satisfaction: float  # 生活满意度 (0-100)


@dataclass
class ComprehensiveEvaluation:
    """综合评价结果"""
    hard_metrics: HardMetrics
    soft_metrics: SoftMetrics
    overall_score: float  # 综合评分 (0-100)
    evaluation_grade: str  # 评价等级 (A/B/C/D)
    key_strengths: List[str]  # 主要优势
    key_weaknesses: List[str]  # 主要不足
    recommendations: List[str]  # 改进建议


class EducationEvaluator:
    """教育评价系统（支持配置化权重）"""

    def __init__(self, config=None):
        """
        初始化评价系统
        
        参数:
            config: 模拟配置对象，用于获取权重配置
        """
        from src.core.config import get_default_config
        if config is None:
            config = get_default_config()
        
        self.config = config
        
        # 高考分数计算参数（可配置，使用常量作为默认值）
        self.gaokao_base_score = getattr(config, 'gaokao_base_score', GAOKAO_BASE_SCORE)
        self.gaokao_max_score = getattr(config, 'gaokao_max_score', GAOKAO_MAX_SCORE)

        # 大学录取分数线（可配置，使用常量作为默认值）
        default_thresholds = {
            UniversityTier.TOP_985: UNIVERSITY_THRESHOLDS["TOP_985"],
            UniversityTier.TOP_211: UNIVERSITY_THRESHOLDS["TOP_211"],
            UniversityTier.ORDINARY_UNIVERSITY: UNIVERSITY_THRESHOLDS["ORDINARY_UNIVERSITY"],
            UniversityTier.JUNIOR_COLLEGE: UNIVERSITY_THRESHOLDS["JUNIOR_COLLEGE"],
        }
        self.university_thresholds = getattr(config, 'university_thresholds', default_thresholds)

        # 高考分数计算权重（可配置，使用常量作为默认值）
        self.gaokao_weights = getattr(config, 'gaokao_weights', None) or DEFAULT_GAOKAO_WEIGHTS.copy()

        # 性格特质权重（可配置）
        self.trait_weights = getattr(config, 'trait_weights', {
            "stress_impact": 0.3,
            "relationship_impact": 0.4,
            "health_impact": 0.2,
            "knowledge_impact": 0.1,
        })
        
        # 综合评价权重（可配置，使用常量作为默认值）
        self.evaluation_weights = getattr(config, 'evaluation_weights', None) or DEFAULT_EVALUATION_WEIGHTS.copy()

    def evaluate_child(
        self,
        child_state: ChildState,
        family_state: FamilyState,
        simulation_weeks: int
    ) -> ComprehensiveEvaluation:
        """
        综合评价孩子培养结果

        参数:
            child_state: 孩子最终状态
            family_state: 家庭最终状态
            simulation_weeks: 模拟总周数

        返回:
            综合评价结果
        """
        # 计算硬性指标
        hard_metrics = self._calculate_hard_metrics(child_state, family_state, simulation_weeks)

        # 计算软指标
        soft_metrics = self._calculate_soft_metrics(child_state, family_state, simulation_weeks)

        # 计算综合评分
        overall_score = self._calculate_overall_score(hard_metrics, soft_metrics)

        # 生成评价等级
        evaluation_grade = self._calculate_grade(overall_score)

        # 分析优势和不足
        key_strengths, key_weaknesses = self._analyze_strengths_weaknesses(hard_metrics, soft_metrics)

        # 生成建议
        recommendations = self._generate_recommendations(hard_metrics, soft_metrics)

        return ComprehensiveEvaluation(
            hard_metrics=hard_metrics,
            soft_metrics=soft_metrics,
            overall_score=overall_score,
            evaluation_grade=evaluation_grade,
            key_strengths=key_strengths,
            key_weaknesses=key_weaknesses,
            recommendations=recommendations
        )

    def _calculate_hard_metrics(
        self,
        child_state: ChildState,
        family_state: FamilyState,
        simulation_weeks: int
    ) -> HardMetrics:
        """计算硬性指标"""

        # 1. 计算高考分数
        gaokao_score = self._calculate_gaokao_score(child_state, family_state)

        # 2. 确定录取院校
        university_tier = self._determine_university_tier(gaokao_score)

        # 3. 确定最终学历
        final_degree = self._determine_final_degree(child_state, university_tier, simulation_weeks)

        # 4. 计算毕业GPA（如果适用）
        graduation_gpa = self._calculate_graduation_gpa(child_state, university_tier) if university_tier != UniversityTier.NO_ADMISSION else 0.0

        return HardMetrics(
            gaokao_score=gaokao_score,
            university_tier=university_tier,
            final_degree=final_degree,
            graduation_gpa=graduation_gpa
        )

    def _calculate_gaokao_score(self, child_state: ChildState, family_state: FamilyState) -> int:
        """
        计算高考分数（优化版：使用配置化权重，更精确的计算）

        影响因素（可配置权重）：
        - 知识储备 (默认40%)
        - IQ智商 (默认20%)
        - 压力水平 (默认负面影响15%)
        - 身体健康 (默认10%)
        - 家庭教育投入 (默认15%)
        """
        # 基础分数
        score = float(self.gaokao_base_score)

        # 知识储备贡献 (0-300分，归一化到0-1)
        knowledge_normalized = child_state.knowledge / 100.0
        knowledge_contribution = knowledge_normalized * 300.0  # 最高300分
        score += knowledge_contribution * abs(self.gaokao_weights.get("knowledge", 0.40))

        # IQ贡献 (80-200范围，归一化)
        iq_normalized = max(0.0, min(1.0, (child_state.iq - 80) / 120.0))  # 80-200映射到0-1
        iq_contribution = iq_normalized * 100.0  # 最高100分
        score += iq_contribution * abs(self.gaokao_weights.get("iq", 0.20))

        # 压力负面影响 (0-100，归一化)
        stress_normalized = child_state.stress / 100.0
        stress_penalty = stress_normalized * 30.0  # 最高30分扣除
        stress_weight = self.gaokao_weights.get("stress", -0.15)
        score += stress_penalty * stress_weight  # 负数权重，所以是减法

        # 健康贡献 (0-100，归一化)
        health_normalized = child_state.physical_health / 100.0
        health_contribution = health_normalized * 20.0  # 最高20分
        score += health_contribution * abs(self.gaokao_weights.get("health", 0.10))

        # 家庭教育投入贡献 (基于存款变化和家长学历)
        family_investment = self._calculate_family_investment_score(family_state)
        family_investment_normalized = family_investment / 100.0  # 归一化到0-1
        family_contribution = family_investment_normalized * 50.0  # 最高50分
        score += family_contribution * abs(self.gaokao_weights.get("family_investment", 0.15))

        # 限制分数范围
        final_score = max(self.gaokao_base_score, min(self.gaokao_max_score, int(round(score))))

        return final_score

    def _calculate_family_investment_score(self, family_state: FamilyState) -> float:
        """计算家庭教育投入分数"""
        # 基于家长学历和家庭存款
        education_levels = {
            "研究生": 100,
            "本科": 80,
            "大专": 60,
            "高中": 40,
            "初中": 20,
            "小学": 10,
        }

        # 计算家长平均学历分数
        parent_scores = []
        for parent in [family_state.father, family_state.mother]:
            score = education_levels.get(parent.education_level, 50)
            parent_scores.append(score)

        avg_parent_score = sum(parent_scores) / len(parent_scores)

        # 家庭存款贡献 (0-50分)
        savings_score = min(50, family_state.family_savings / 10000)  # 每10万存款加1分

        return (avg_parent_score * 0.7 + savings_score * 0.3)

    def _determine_university_tier(self, gaokao_score: int) -> UniversityTier:
        """根据高考分数确定录取院校等级"""
        if gaokao_score >= self.university_thresholds[UniversityTier.TOP_985]:
            return UniversityTier.TOP_985
        elif gaokao_score >= self.university_thresholds[UniversityTier.TOP_211]:
            return UniversityTier.TOP_211
        elif gaokao_score >= self.university_thresholds[UniversityTier.ORDINARY_UNIVERSITY]:
            return UniversityTier.ORDINARY_UNIVERSITY
        elif gaokao_score >= self.university_thresholds[UniversityTier.JUNIOR_COLLEGE]:
            return UniversityTier.JUNIOR_COLLEGE
        else:
            return UniversityTier.NO_ADMISSION

    def _determine_final_degree(
        self,
        child_state: ChildState,
        university_tier: UniversityTier,
        simulation_weeks: int
    ) -> DegreeLevel:
        """确定最终学历"""

        # 如果未录取任何大学
        if university_tier == UniversityTier.NO_ADMISSION:
            return DegreeLevel.HIGH_SCHOOL

        # 如果只上了大专
        if university_tier == UniversityTier.JUNIOR_COLLEGE:
            return DegreeLevel.JUNIOR_COLLEGE

        # 本科及以上院校，考虑是否能读研
        base_degree = DegreeLevel.BACHELORS

        # 硕士预录取概率计算
        masters_probability = self._calculate_masters_probability(child_state, university_tier)

        if random.random() < masters_probability:
            return DegreeLevel.MASTERS_PRE_ADMISSION

        return base_degree

    def _calculate_masters_probability(self, child_state: ChildState, university_tier: UniversityTier) -> float:
        """计算硕士预录取概率"""
        # 基础概率
        base_prob = 0.3

        # 985高校加成
        if university_tier == UniversityTier.TOP_985:
            base_prob += 0.3
        elif university_tier == UniversityTier.TOP_211:
            base_prob += 0.2

        # 知识储备加成
        knowledge_bonus = child_state.knowledge / 100 * 0.2
        base_prob += knowledge_bonus

        # IQ加成
        iq_bonus = (child_state.iq - 100) / 50 * 0.1
        base_prob += max(0, iq_bonus)

        # 压力惩罚
        stress_penalty = child_state.stress / 100 * 0.15
        base_prob -= stress_penalty

        return max(0.05, min(0.8, base_prob))

    def _calculate_graduation_gpa(self, child_state: ChildState, university_tier: UniversityTier) -> float:
        """计算大学毕业GPA"""
        # 基础GPA
        base_gpa = 2.5

        # 知识储备贡献
        knowledge_bonus = (child_state.knowledge - 50) / 50 * 0.5
        base_gpa += knowledge_bonus

        # IQ贡献
        iq_bonus = (child_state.iq - 100) / 50 * 0.3
        base_gpa += iq_bonus

        # 院校等级加成
        tier_bonus = {
            UniversityTier.TOP_985: 0.3,
            UniversityTier.TOP_211: 0.2,
            UniversityTier.ORDINARY_UNIVERSITY: 0.1,
            UniversityTier.JUNIOR_COLLEGE: 0.0,
        }
        base_gpa += tier_bonus.get(university_tier, 0)

        # 限制GPA范围 (0-4.0)
        return max(0.0, min(4.0, base_gpa))

    def _calculate_soft_metrics(
        self,
        child_state: ChildState,
        family_state: FamilyState,
        simulation_weeks: int
    ) -> SoftMetrics:
        """计算软指标"""

        # 1. 分析性格特质
        personality_traits = self._analyze_personality_traits(child_state, family_state)

        # 2. 计算幸福指数
        happiness_index = self._calculate_happiness_index(child_state, family_state)

        # 3. 计算社交适应度
        social_adaptability = self._calculate_social_adaptability(child_state)

        # 4. 计算情绪稳定性
        emotional_stability = self._calculate_emotional_stability(child_state)

        # 5. 计算生活满意度
        life_satisfaction = self._calculate_life_satisfaction(child_state, family_state, happiness_index)

        return SoftMetrics(
            personality_traits=personality_traits,
            happiness_index=happiness_index,
            social_adaptability=social_adaptability,
            emotional_stability=emotional_stability,
            life_satisfaction=life_satisfaction
        )

    def _analyze_personality_traits(self, child_state: ChildState, family_state: FamilyState) -> List[PersonalityTrait]:
        """
        分析性格特质（优化版：引入概率模型）
        
        使用概率模型，根据状态值计算每个特质的概率，然后随机选择
        """
        traits = []
        avg_relationship = (child_state.father_relationship + child_state.mother_relationship +
                          child_state.grandfather_relationship + child_state.grandmother_relationship) / 4

        # 自信/自卑分析（概率模型）
        confidence_prob = self._calculate_trait_probability(
            positive_factors={
                "knowledge": child_state.knowledge / 100.0,
                "iq": (child_state.iq - 80) / 120.0,
            },
            negative_factors={
                "stress": child_state.stress / 100.0,
            },
            positive_threshold=0.6,
            negative_threshold=0.5
        )
        if random.random() < confidence_prob:
            traits.append(PersonalityTrait.CONFIDENT)
        elif child_state.stress > 70 and random.random() < 0.7:
            traits.append(PersonalityTrait.PESSIMISTIC)

        # 内向/外向分析（概率模型）
        outgoing_prob = self._calculate_trait_probability(
            positive_factors={
                "relationship": avg_relationship / 100.0,
                "health": child_state.physical_health / 100.0,
            },
            negative_factors={
                "stress": child_state.stress / 100.0,
            },
            positive_threshold=0.65,
            negative_threshold=0.4
        )
        if random.random() < outgoing_prob:
            traits.append(PersonalityTrait.OUTGOING)
        elif avg_relationship < 40 and random.random() < 0.6:
            traits.append(PersonalityTrait.INTROVERTED)

        # 抗压/脆弱分析（概率模型）
        stress_resistant_prob = self._calculate_trait_probability(
            positive_factors={
                "low_stress": (100 - child_state.stress) / 100.0,
                "health": child_state.physical_health / 100.0,
            },
            negative_factors={},
            positive_threshold=0.6,
            negative_threshold=0.0
        )
        if random.random() < stress_resistant_prob:
            traits.append(PersonalityTrait.STRESS_RESISTANT)
        elif child_state.stress > 70 and random.random() < 0.65:
            traits.append(PersonalityTrait.FRAGILE)

        # 乐观/悲观分析（概率模型）
        optimistic_prob = self._calculate_trait_probability(
            positive_factors={
                "health": child_state.physical_health / 100.0,
                "relationship": avg_relationship / 100.0,
                "knowledge": child_state.knowledge / 100.0,
            },
            negative_factors={
                "stress": child_state.stress / 100.0,
            },
            positive_threshold=0.6,
            negative_threshold=0.5
        )
        if random.random() < optimistic_prob:
            traits.append(PersonalityTrait.OPTIMISTIC)
        elif (child_state.stress > 60 or child_state.physical_health < 50) and random.random() < 0.6:
            traits.append(PersonalityTrait.PESSIMISTIC)

        # 独立/依赖分析（概率模型）
        independence_score = (child_state.knowledge + child_state.iq) / 2
        independent_prob = self._calculate_trait_probability(
            positive_factors={
                "independence": independence_score / 100.0,
                "knowledge": child_state.knowledge / 100.0,
            },
            negative_factors={},
            positive_threshold=0.65,
            negative_threshold=0.0
        )
        if random.random() < independent_prob:
            traits.append(PersonalityTrait.INDEPENDENT)
        elif independence_score < 50 and random.random() < 0.55:
            traits.append(PersonalityTrait.DEPENDENT)

        # 创造性/务实分析（概率模型）
        creativity_score = child_state.iq - child_state.knowledge
        creativity_normalized = (creativity_score + 50) / 100.0  # 归一化到0-1
        creative_prob = max(0.0, min(1.0, creativity_normalized))
        if random.random() < creative_prob:
            traits.append(PersonalityTrait.CREATIVE)
        elif creativity_score < -20 and random.random() < 0.6:
            traits.append(PersonalityTrait.PRACTICAL)

        return traits
    
    def _calculate_trait_probability(
        self,
        positive_factors: Dict[str, float],
        negative_factors: Dict[str, float],
        positive_threshold: float = 0.6,
        negative_threshold: float = 0.5
    ) -> float:
        """
        计算特质概率
        
        参数:
            positive_factors: 积极因素字典 {因子名: 归一化值(0-1)}
            negative_factors: 消极因素字典 {因子名: 归一化值(0-1)}
            positive_threshold: 积极因素阈值
            negative_threshold: 消极因素阈值
            
        返回:
            特质概率 (0-1)
        """
        # 计算积极因素加权平均
        if positive_factors:
            positive_score = sum(positive_factors.values()) / len(positive_factors)
        else:
            positive_score = 0.0
        
        # 计算消极因素加权平均
        if negative_factors:
            negative_score = sum(negative_factors.values()) / len(negative_factors)
        else:
            negative_score = 0.0
        
        # 基础概率
        base_prob = 0.3
        
        # 积极因素加成
        if positive_score > positive_threshold:
            positive_bonus = (positive_score - positive_threshold) * 0.5
            base_prob += positive_bonus
        
        # 消极因素惩罚
        if negative_score > negative_threshold:
            negative_penalty = (negative_score - negative_threshold) * 0.4
            base_prob -= negative_penalty
        
        # 限制在合理范围
        return max(0.1, min(0.9, base_prob))

    def _calculate_happiness_index(self, child_state: ChildState, family_state: FamilyState) -> float:
        """计算幸福指数"""
        # 各项指标的权重
        weights = {
            'relationship': 0.4,  # 人际关系
            'health': 0.25,       # 健康状况
            'achievement': 0.2,   # 成就感
            'stress': 0.15,       # 压力水平（负权重）
        }

        # 计算各项得分
        avg_relationship = (child_state.father_relationship + child_state.mother_relationship +
                          child_state.grandfather_relationship + child_state.grandmother_relationship) / 4

        relationship_score = avg_relationship
        health_score = child_state.physical_health
        achievement_score = child_state.knowledge
        stress_penalty = child_state.stress

        # 加权计算
        happiness = (
            relationship_score * weights['relationship'] +
            health_score * weights['health'] +
            achievement_score * weights['achievement'] -
            stress_penalty * weights['stress']
        )

        return max(0.0, min(100.0, happiness))

    def _calculate_social_adaptability(self, child_state: ChildState) -> float:
        """计算社交适应度"""
        # 基于关系和压力的综合评估
        avg_relationship = (child_state.father_relationship + child_state.mother_relationship +
                          child_state.grandfather_relationship + child_state.grandmother_relationship) / 4

        adaptability = avg_relationship * 0.7 + (100 - child_state.stress) * 0.3

        return max(0.0, min(100.0, adaptability))

    def _calculate_emotional_stability(self, child_state: ChildState) -> float:
        """计算情绪稳定性"""
        # 主要基于压力水平和健康状况
        stability = (100 - child_state.stress) * 0.6 + child_state.physical_health * 0.4

        return max(0.0, min(100.0, stability))

    def _calculate_life_satisfaction(self, child_state: ChildState, family_state: FamilyState, happiness_index: float) -> float:
        """计算生活满意度"""
        # 综合考虑幸福指数、成就和经济状况
        achievement_score = child_state.knowledge
        economic_factor = min(100, family_state.family_savings / 1000)  # 每10万存款对应10分

        satisfaction = happiness_index * 0.5 + achievement_score * 0.3 + economic_factor * 0.2

        return max(0.0, min(100.0, satisfaction))

    def _calculate_overall_score(self, hard_metrics: HardMetrics, soft_metrics: SoftMetrics) -> float:
        """计算综合评分"""
        # 硬指标权重
        hard_score = self._calculate_hard_score(hard_metrics)

        # 软指标权重
        soft_score = (
            soft_metrics.happiness_index * 0.4 +
            soft_metrics.social_adaptability * 0.3 +
            soft_metrics.emotional_stability * 0.2 +
            soft_metrics.life_satisfaction * 0.1
        )

        # 综合评分：使用配置化权重
        hard_weight = self.evaluation_weights.get("hard_metrics", 0.6)
        soft_weight = self.evaluation_weights.get("soft_metrics", 0.4)
        # 归一化权重
        total_weight = hard_weight + soft_weight
        if total_weight > 0:
            hard_weight = hard_weight / total_weight
            soft_weight = soft_weight / total_weight
        overall_score = hard_score * hard_weight + soft_score * soft_weight

        return max(0.0, min(100.0, overall_score))

    def _calculate_hard_score(self, hard_metrics: HardMetrics) -> float:
        """计算硬指标得分"""
        score = 0

        # 高考分数贡献 (0-40分)
        gaokao_percentage = (hard_metrics.gaokao_score - self.gaokao_base_score) / (self.gaokao_max_score - self.gaokao_base_score)
        score += gaokao_percentage * 40

        # 大学等级贡献 (0-35分)
        university_scores = {
            UniversityTier.TOP_985: 35,
            UniversityTier.TOP_211: 28,
            UniversityTier.ORDINARY_UNIVERSITY: 21,
            UniversityTier.JUNIOR_COLLEGE: 14,
            UniversityTier.NO_ADMISSION: 0,
        }
        score += university_scores.get(hard_metrics.university_tier, 0)

        # 学历贡献 (0-25分)
        degree_scores = {
            DegreeLevel.MASTERS_PRE_ADMISSION: 25,
            DegreeLevel.BACHELORS: 20,
            DegreeLevel.JUNIOR_COLLEGE: 15,
            DegreeLevel.HIGH_SCHOOL: 10,
            DegreeLevel.BELOW_HIGH_SCHOOL: 0,
        }
        score += degree_scores.get(hard_metrics.final_degree, 0)

        return score

    def _calculate_grade(self, overall_score: float) -> str:
        """计算评价等级"""
        if overall_score >= 90:
            return "A"
        elif overall_score >= 80:
            return "B"
        elif overall_score >= 70:
            return "C"
        elif overall_score >= 60:
            return "D"
        else:
            return "F"

    def _analyze_strengths_weaknesses(self, hard_metrics: HardMetrics, soft_metrics: SoftMetrics) -> Tuple[List[str], List[str]]:
        """分析优势和不足"""
        strengths = []
        weaknesses = []

        # 硬指标分析
        if hard_metrics.gaokao_score > 650:
            strengths.append("高考成绩优异")
        elif hard_metrics.gaokao_score < 500:
            weaknesses.append("高考成绩需提升")

        if hard_metrics.university_tier in [UniversityTier.TOP_985, UniversityTier.TOP_211]:
            strengths.append("进入重点高校")
        elif hard_metrics.university_tier == UniversityTier.NO_ADMISSION:
            weaknesses.append("未能进入理想大学")

        if hard_metrics.final_degree == DegreeLevel.MASTERS_PRE_ADMISSION:
            strengths.append("具备深造潜力")
        elif hard_metrics.final_degree == DegreeLevel.HIGH_SCHOOL:
            weaknesses.append("学历层次有待提高")

        # 软指标分析
        if soft_metrics.happiness_index > 70:
            strengths.append("心理状态健康")
        elif soft_metrics.happiness_index < 50:
            weaknesses.append("幸福指数偏低")

        if soft_metrics.social_adaptability > 70:
            strengths.append("社交能力良好")
        elif soft_metrics.social_adaptability < 50:
            weaknesses.append("社交适应能力需加强")

        if soft_metrics.emotional_stability > 70:
            strengths.append("情绪稳定")
        elif soft_metrics.emotional_stability < 50:
            weaknesses.append("情绪稳定性不足")

        if PersonalityTrait.CONFIDENT in soft_metrics.personality_traits:
            strengths.append("性格自信")
        elif PersonalityTrait.PESSIMISTIC in soft_metrics.personality_traits:
            weaknesses.append("性格较为悲观")

        return strengths, weaknesses

    def _generate_recommendations(self, hard_metrics: HardMetrics, soft_metrics: SoftMetrics) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于硬指标的建议
        if hard_metrics.gaokao_score < 600:
            recommendations.append("加强学习基础，提高应试能力")

        if hard_metrics.university_tier == UniversityTier.NO_ADMISSION:
            recommendations.append("考虑职业教育或成人教育途径")

        # 基于软指标的建议
        if soft_metrics.happiness_index < 60:
            recommendations.append("关注心理健康，培养积极心态")

        if soft_metrics.social_adaptability < 60:
            recommendations.append("加强社交技能训练，提高人际交往能力")

        if soft_metrics.emotional_stability < 60:
            recommendations.append("学习情绪管理技巧，增强抗压能力")

        if PersonalityTrait.FRAGILE in soft_metrics.personality_traits:
            recommendations.append("培养抗压能力，建立健康的生活习惯")

        if PersonalityTrait.INTROVERTED in soft_metrics.personality_traits:
            recommendations.append("适度拓展社交圈，培养外向性格")

        # 通用建议
        if not recommendations:
            recommendations.append("继续保持良好的学习和生活习惯")
            recommendations.append("注重全面发展，培养综合素质")

        return recommendations


# 创建全局评价器实例
_evaluation_system = None

def get_evaluation_system() -> EducationEvaluator:
    """获取教育评价系统实例"""
    global _evaluation_system
    if _evaluation_system is None:
        _evaluation_system = EducationEvaluator()
    return _evaluation_system