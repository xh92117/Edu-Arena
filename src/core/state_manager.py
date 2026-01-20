"""
统一状态管理接口
集中处理所有状态更新和边界检查
"""
from typing import Dict, Any, Optional, Callable
from src.core.state import ChildState, FamilyState
from src.core.constants import STATE_CHANGE_LIMITS, STATE_BOUNDS
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """
    统一状态管理器
    负责所有状态更新和边界检查
    """
    
    # 状态变化上限（从常量导入）
    MAX_CHANGE_LIMITS = STATE_CHANGE_LIMITS
    
    # 状态边界（从常量导入）
    STATE_BOUNDS = STATE_BOUNDS
    
    @staticmethod
    def update_child_state(
        child_state: ChildState,
        changes: Dict[str, float],
        apply_limits: bool = True
    ) -> Dict[str, float]:
        """
        统一更新孩子状态
        
        参数:
            child_state: 孩子状态对象
            changes: 状态变化字典 {属性名: 变化值}
            apply_limits: 是否应用变化上限
            
        返回:
            实际应用的变化字典
        """
        actual_changes = {}
        
        for attr_name, change_value in changes.items():
            if not hasattr(child_state, attr_name):
                logger.warning(f"ChildState没有属性: {attr_name}")
                continue
            
            # 应用变化上限
            if apply_limits and attr_name in StateManager.MAX_CHANGE_LIMITS:
                max_change = StateManager.MAX_CHANGE_LIMITS[attr_name]
                change_value = max(-max_change, min(max_change, change_value))
            
            # 获取当前值
            old_value = getattr(child_state, attr_name)
            
            # 计算新值
            new_value = old_value + change_value
            
            # 应用边界检查
            if attr_name in StateManager.STATE_BOUNDS:
                min_val, max_val = StateManager.STATE_BOUNDS[attr_name]
                new_value = max(min_val, min(max_val, new_value))
            
            # 更新状态
            setattr(child_state, attr_name, new_value)
            
            # 记录实际变化
            actual_changes[attr_name] = new_value - old_value
            
            logger.debug(f"更新{attr_name}: {old_value:.2f} -> {new_value:.2f} (变化: {change_value:.2f})")
        
        return actual_changes
    
    @staticmethod
    def update_family_state(
        family_state: FamilyState,
        changes: Dict[str, float],
        apply_limits: bool = True
    ) -> Dict[str, float]:
        """
        统一更新家庭状态
        
        参数:
            family_state: 家庭状态对象
            changes: 状态变化字典 {属性名: 变化值}
            apply_limits: 是否应用变化上限
            
        返回:
            实际应用的变化字典
        """
        actual_changes = {}
        
        for attr_name, change_value in changes.items():
            if attr_name == "family_savings":
                # 处理存款变化
                old_value = family_state.family_savings
                new_value = old_value + change_value
                
                # 应用边界检查
                if attr_name in StateManager.STATE_BOUNDS:
                    min_val, max_val = StateManager.STATE_BOUNDS[attr_name]
                    new_value = max(min_val, min(max_val, new_value))
                
                family_state.family_savings = new_value
                actual_changes[attr_name] = new_value - old_value
                
            elif attr_name.startswith(("father.", "mother.", "grandfather.", "grandmother.")):
                # 处理家庭成员属性变化
                parts = attr_name.split(".")
                if len(parts) == 2:
                    member_name, member_attr = parts
                    member_obj = getattr(family_state, member_name, None)
                    if member_obj and hasattr(member_obj, member_attr):
                        old_value = getattr(member_obj, member_attr)
                        new_value = old_value + change_value
                        setattr(member_obj, member_attr, new_value)
                        actual_changes[attr_name] = new_value - old_value
        
        return actual_changes
    
    @staticmethod
    def validate_state(child_state: ChildState, family_state: FamilyState) -> Dict[str, Any]:
        """
        验证状态是否在有效范围内
        
        参数:
            child_state: 孩子状态
            family_state: 家庭状态
            
        返回:
            验证结果字典
        """
        issues = []
        
        # 验证孩子状态
        for attr_name, (min_val, max_val) in StateManager.STATE_BOUNDS.items():
            if hasattr(child_state, attr_name):
                value = getattr(child_state, attr_name)
                if value < min_val or value > max_val:
                    issues.append(f"{attr_name}: {value} 超出范围 [{min_val}, {max_val}]")
        
        # 验证家庭状态
        if family_state.family_savings < 0:
            issues.append(f"family_savings: {family_state.family_savings} 不能为负")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    @staticmethod
    def clamp_value(value: float, attr_name: str) -> float:
        """
        将值限制在有效范围内
        
        参数:
            value: 原始值
            attr_name: 属性名称
            
        返回:
            限制后的值
        """
        if attr_name in StateManager.STATE_BOUNDS:
            min_val, max_val = StateManager.STATE_BOUNDS[attr_name]
            return max(min_val, min(max_val, value))
        return value
    
    @staticmethod
    def limit_change(change: float, attr_name: str) -> float:
        """
        限制单次变化的最大值
        
        参数:
            change: 原始变化值
            attr_name: 属性名称
            
        返回:
            限制后的变化值
        """
        if attr_name in StateManager.MAX_CHANGE_LIMITS:
            max_change = StateManager.MAX_CHANGE_LIMITS[attr_name]
            return max(-max_change, min(max_change, change))
        return change
