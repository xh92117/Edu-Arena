"""
情绪传染系统

实现家庭成员之间的情绪传染机制：
1. 父母压力影响孩子
2. 孩子状态影响父母情绪
3. 家庭氛围的整体影响
"""

import logging
from typing import Dict, Any, Optional
from src.core.state import ChildState, FamilyState, ChildEmotionalState

logger = logging.getLogger(__name__)


class EmotionalContagion:
    """
    情绪传染机制
    
    核心原理：
    - 家庭成员的情绪是相互影响的
    - 孩子的敏感度决定了受影响的程度
    - 关系越亲密，情绪传染越强
    """
    
    # 情绪传染系数
    PARENT_TO_CHILD_FACTOR = 0.3  # 父母对孩子的影响系数
    CHILD_TO_PARENT_FACTOR = 0.2  # 孩子对父母的影响系数
    
    @classmethod
    def propagate_family_stress(cls, child_state: ChildState, family_state: FamilyState) -> Dict[str, float]:
        """
        传播家庭压力
        
        父母压力高 -> 影响孩子
        孩子状态差 -> 影响父母
        
        参数:
            child_state: 孩子状态
            family_state: 家庭状态
            
        返回:
            压力变化字典
        """
        changes = {}
        
        # 1. 父母压力影响孩子
        parent_stress = cls._calculate_parent_stress(family_state)
        child_stress_change = cls._parent_stress_to_child(
            parent_stress, 
            child_state
        )
        
        if abs(child_stress_change) > 0.1:
            child_state.stress = max(0.0, min(100.0, child_state.stress + child_stress_change))
            changes["child_stress_from_parents"] = child_stress_change
            logger.debug(f"父母压力传染给孩子: {child_stress_change:.2f}")
        
        # 2. 孩子状态影响父母
        if child_state.stress > 60:
            parent_stress_change = cls._child_stress_to_parent(child_state)
            
            # 主要影响父母，祖父母受影响较小
            family_state.father.update_stress(parent_stress_change * 1.0)
            family_state.mother.update_stress(parent_stress_change * 1.2)  # 母亲更敏感
            family_state.grandfather.update_stress(parent_stress_change * 0.5)
            family_state.grandmother.update_stress(parent_stress_change * 0.6)
            
            changes["parent_stress_from_child"] = parent_stress_change
            logger.debug(f"孩子压力传染给父母: {parent_stress_change:.2f}")
        
        # 3. 更新孩子的安全感
        cls._update_child_security(child_state, family_state)
        
        return changes
    
    @classmethod
    def _calculate_parent_stress(cls, family_state: FamilyState) -> float:
        """计算父母的平均压力水平"""
        # 父母权重更高
        total = (
            family_state.father.stress_level * 0.35 +
            family_state.mother.stress_level * 0.35 +
            family_state.grandfather.stress_level * 0.15 +
            family_state.grandmother.stress_level * 0.15
        )
        return total
    
    @classmethod
    def _parent_stress_to_child(cls, parent_stress: float, child_state: ChildState) -> float:
        """
        计算父母压力对孩子的影响
        
        影响因素：
        - 父母压力水平
        - 孩子的情绪敏感度
        - 孩子的安全感
        """
        # 基础影响
        base_effect = (parent_stress - 50) * cls.PARENT_TO_CHILD_FACTOR / 100
        
        # 孩子的敏感度放大效果
        sensitivity = 1.0 - child_state.personality.emotional_stability
        sensitivity_multiplier = 1.0 + sensitivity * 0.5
        
        # 安全感高可以缓冲负面影响
        security_buffer = child_state.security_feeling / 100.0
        if base_effect > 0:  # 负面影响
            base_effect *= (1.0 - security_buffer * 0.3)
        
        return base_effect * sensitivity_multiplier
    
    @classmethod
    def _child_stress_to_parent(cls, child_state: ChildState) -> float:
        """计算孩子压力对父母的影响"""
        # 孩子压力超过60才开始影响父母
        excess_stress = max(0, child_state.stress - 60)
        return excess_stress * cls.CHILD_TO_PARENT_FACTOR / 100
    
    @classmethod
    def _update_child_security(cls, child_state: ChildState, family_state: FamilyState) -> None:
        """更新孩子的安全感"""
        # 父母情绪稳定增加安全感
        parent_stability = (
            (100 - family_state.father.stress_level) +
            (100 - family_state.mother.stress_level)
        ) / 200.0
        
        # 亲子关系影响安全感
        relationship = (
            child_state.father_relationship +
            child_state.mother_relationship
        ) / 200.0
        
        # 计算安全感变化
        target_security = (parent_stability * 0.4 + relationship * 0.6) * 100
        current_security = child_state.security_feeling
        
        # 缓慢向目标靠近
        change = (target_security - current_security) * 0.1
        child_state.security_feeling = max(0.0, min(100.0, current_security + change))


class FamilyAtmosphere:
    """
    家庭氛围系统
    
    家庭整体氛围会影响所有成员
    """
    
    @classmethod
    def calculate_atmosphere(cls, child_state: ChildState, family_state: FamilyState) -> Dict[str, Any]:
        """
        计算家庭氛围
        
        返回:
            氛围指标字典
        """
        # 计算各项指标
        
        # 1. 经济压力 (0-100, 越高越有压力)
        monthly_income = (
            family_state.father.salary +
            family_state.mother.salary +
            family_state.grandfather.salary +
            family_state.grandmother.salary
        )
        savings_months = family_state.family_savings / max(1, monthly_income * 0.5)  # 能撑几个月
        economic_pressure = max(0, min(100, 100 - savings_months * 10))
        
        # 2. 情感温暖度 (0-100)
        avg_relationship = (
            child_state.father_relationship +
            child_state.mother_relationship +
            child_state.grandfather_relationship +
            child_state.grandmother_relationship
        ) / 4.0
        emotional_warmth = avg_relationship
        
        # 3. 压力水平 (0-100)
        family_stress = (
            family_state.father.stress_level +
            family_state.mother.stress_level +
            child_state.stress
        ) / 3.0
        
        # 4. 和谐度 (0-100)
        harmony = (emotional_warmth * 0.5 + (100 - family_stress) * 0.3 + (100 - economic_pressure) * 0.2)
        
        # 判断氛围类型
        if harmony > 75:
            atmosphere_type = "温馨和谐"
        elif harmony > 60:
            atmosphere_type = "平稳"
        elif harmony > 45:
            atmosphere_type = "有些紧张"
        elif harmony > 30:
            atmosphere_type = "紧张"
        else:
            atmosphere_type = "压抑"
        
        return {
            "type": atmosphere_type,
            "harmony": harmony,
            "economic_pressure": economic_pressure,
            "emotional_warmth": emotional_warmth,
            "family_stress": family_stress
        }
    
    @classmethod
    def apply_atmosphere_effects(cls, child_state: ChildState, 
                                  family_state: FamilyState,
                                  atmosphere: Dict[str, Any]) -> Dict[str, float]:
        """
        应用家庭氛围的效果
        
        参数:
            child_state: 孩子状态
            family_state: 家庭状态
            atmosphere: 氛围指标
            
        返回:
            状态变化字典
        """
        changes = {}
        harmony = atmosphere["harmony"]
        
        # 温馨的家庭氛围有利于孩子发展
        if harmony > 70:
            # 减少压力
            stress_change = -1.0
            child_state.stress = max(0, child_state.stress + stress_change)
            changes["stress"] = stress_change
            
            # 增加安全感
            security_change = 0.5
            child_state.security_feeling = min(100, child_state.security_feeling + security_change)
            changes["security_feeling"] = security_change
            
            # 增加自信
            confidence_change = 0.3
            child_state.self_confidence = min(100, child_state.self_confidence + confidence_change)
            changes["self_confidence"] = confidence_change
        
        elif harmony < 40:
            # 紧张的氛围
            stress_change = 1.5
            child_state.stress = min(100, child_state.stress + stress_change)
            changes["stress"] = stress_change
            
            # 降低安全感
            security_change = -1.0
            child_state.security_feeling = max(0, child_state.security_feeling + security_change)
            changes["security_feeling"] = security_change
        
        return changes


class MoodInfluence:
    """
    情绪影响行为效果
    
    父母的情绪状态会影响其行为的效果
    """
    
    # 情绪对行为效果的修正系数
    MOOD_MODIFIERS = {
        "平静": 1.0,
        "开心": 1.2,
        "骄傲": 1.1,
        "充满希望": 1.15,
        "焦虑": 0.8,
        "担忧": 0.85,
        "压力大": 0.7,
        "沮丧": 0.6,
        "生气": 0.5,
        "疲惫": 0.75
    }
    
    @classmethod
    def get_action_modifier(cls, family_member_emotional_state: str) -> float:
        """
        获取行为效果修正系数
        
        参数:
            family_member_emotional_state: 家庭成员的情绪状态
            
        返回:
            效果修正系数 (0.5-1.2)
        """
        return cls.MOOD_MODIFIERS.get(family_member_emotional_state, 1.0)
    
    @classmethod
    def modify_action_effects(cls, effects: Dict[str, float], 
                               emotional_state: str,
                               child_emotional_stability: float) -> Dict[str, float]:
        """
        根据情绪修正行为效果
        
        参数:
            effects: 原始效果字典
            emotional_state: 执行者的情绪状态
            child_emotional_stability: 孩子的情绪稳定性
            
        返回:
            修正后的效果字典
        """
        modifier = cls.get_action_modifier(emotional_state)
        modified = {}
        
        for key, value in effects.items():
            if key.startswith("_"):  # 跳过私有属性
                continue
                
            if value > 0:
                # 正面效果受情绪修正
                modified[key] = value * modifier
            else:
                # 负面效果：情绪不好时负面效果更严重
                negative_modifier = 2.0 - modifier  # 情绪好时0.8，情绪差时1.5
                modified[key] = value * negative_modifier
        
        return modified
