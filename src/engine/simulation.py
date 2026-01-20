import asyncio
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
import json
import os

from src.core.state import ChildState, FamilyState
from src.agents.factory import AgentFactory
from src.engine.dungeon_master import get_dungeon_master
from src.data.event_system import EnhancedEventSystem
from src.core.llm_client import LLMClientFactory
from src.core.config import get_default_config
from src.core.decision_coordinator import DecisionCoordinator
from src.core.dialogue_system import DialogueSystemFactory


class SimulationEnv:
    """模拟环境类，管理单个模拟实例的状态和运行"""
    
    def __init__(self, env_id: int, start_date: date = date(2010, 1, 1), model_name: str = "deepseek", config=None):
        self.env_id = env_id
        self.start_date = start_date
        self.config = config or get_default_config()

        # 初始化状态
        self.child_state = ChildState()
        self.family_state = FamilyState(current_date=start_date)

        # 初始化组件
        self.agent_factory = AgentFactory()
        llm_client = LLMClientFactory.create_client(self.config, model_name=model_name)
        self.dungeon_master = get_dungeon_master(llm_client)
        self.decision_coordinator = DecisionCoordinator(model_name=model_name, config=self.config)  # 传递模型名称和配置
        self.event_system = EnhancedEventSystem()

        # 模拟参数
        self.current_step = 0
        self.current_week = 0
        
    def get_current_events(self) -> List[Dict[str, Any]]:
        """获取当周事件列表（包括随机事件和连锁事件）"""
        return self.event_system.get_events_for_week(
            self.family_state.current_date,
            self.child_state,
            self.family_state
        )
    
    async def run_week(self) -> Dict[str, Any]:
        """运行一周的模拟流程"""
        # 0. 检查是否处于崩溃恢复期（如果上周崩溃，本周可能跳过决策）
        mental_breakdown = None
        
        # 1. 获取当周事件
        current_events = self.get_current_events()
        current_event_desc = "\n".join([event["description"] for event in current_events])

        # 2. 检查压力是否>90，如果>90则直接触发崩溃，跳过决策
        if self.child_state.stress > 90:
            # 直接触发崩溃，不执行决策
            mental_breakdown = {
                "triggered": True,
                "message": f"孩子因压力过大（{self.child_state.stress:.1f}）发生心理崩溃，本周无法正常学习和活动",
                "skip_decision": True
            }
            # 应用崩溃影响
            breakdown_handler = self.dungeon_master._handle_mental_breakdown(
                self.child_state,
                self.family_state
            )
            mental_breakdown.update(breakdown_handler)
            coordinated_decision = {
                "action_type": "无行动",
                "dialogue": "孩子因心理崩溃无法执行任何行动",
                "cost": 0,
                "member": "系统",
                "member_id": "system",
                "reasoning": "心理崩溃导致无法决策",
                "confidence": 0,
                "coordination_reason": "系统自动处理崩溃事件"
            }
            result = {
                "success": False,
                "message": "因心理崩溃跳过本周决策",
                "cost": 0,
                "member": "system",
                "state_changes": breakdown_handler.get("state_changes", {}),
                "mental_breakdown": mental_breakdown
            }
        else:
            # 2. 多成员决策协调
            coordinated_decision = await self.decision_coordinator.coordinate_decision(
                self.child_state,
                self.family_state,
                current_event_desc
            )

            # 3. DM评估结果（使用 member_id 作为内部处理用的英文标识）
            result = await self.dungeon_master.evaluate_outcome(
                self.child_state,
                self.family_state,
                coordinated_decision,
                current_event_desc,
                coordinated_decision.get("member_id", "family")
            )
            
            # 检查评估结果中是否包含崩溃事件
            if result.get("mental_breakdown"):
                mental_breakdown = result["mental_breakdown"]
            
            # 3.1 记录对话效果（用于后续对话连续性）
            self._record_dialogue_outcome(coordinated_decision, result)

        # 4. 应用事件影响（传入事件数量用于衰减计算）
        event_count = len(current_events)
        for event_index, event in enumerate(current_events):
            self.event_system.apply_event_impact(
                event, 
                self.child_state, 
                self.family_state,
                event_count=event_count,
                event_index=event_index
            )

        # 5. 更新状态
        self.family_state.advance_date(weeks=1, config=self.config)
        
        # 5.1 更新IQ（动态变化）
        self.child_state.update_iq(self.family_state.current_date, config=self.config)
        
        # 5.2 更新发展敏感期
        self.child_state.update_development_sensitivity(self.family_state.current_date)
        
        # 5.3 情绪传染系统
        try:
            from src.core.emotional_system import EmotionalContagion, FamilyAtmosphere
            
            # 情绪传染
            emotional_changes = EmotionalContagion.propagate_family_stress(
                self.child_state, self.family_state
            )
            
            # 家庭氛围影响
            atmosphere = FamilyAtmosphere.calculate_atmosphere(
                self.child_state, self.family_state
            )
            atmosphere_effects = FamilyAtmosphere.apply_atmosphere_effects(
                self.child_state, self.family_state, atmosphere
            )
            
            # 更新孩子情绪状态
            stress_change = result.get("state_changes", {}).get("stress", 0)
            relationship_change = sum([
                result.get("state_changes", {}).get(f"{m}_relationship", 0)
                for m in ["father", "mother", "grandfather", "grandmother"]
            ]) / 4.0
            self.child_state.update_emotional_state(stress_change, relationship_change)
            
            # 更新孩子的次要属性
            action_success = result.get("success", True)
            self.child_state.update_secondary_attributes(
                coordinated_decision.get("action_type", "陪伴"),
                action_success,
                relationship_change > 0
            )
            
        except Exception as e:
            # 情绪系统出错不影响主流程
            import logging
            logging.getLogger(__name__).warning(f"情绪系统更新失败: {e}")
        
        self.current_step += 1
        self.current_week += 1
        
        # 5. 构建日志数据，确保所有日期对象都被转换为字符串
        child_state_data = self.child_state.model_dump()
        family_state_data = self.family_state.model_dump()

        # 将date对象转换为ISO字符串
        family_state_data["current_date"] = family_state_data["current_date"].isoformat()

        log_data = {
            "timestamp": family_state_data["current_date"],
            "env_id": self.env_id,
            "week": self.current_week,
            "child_state": child_state_data,
            "family_state": family_state_data,
            "current_events": current_events,
            "coordinated_decision": coordinated_decision,
            "dm_result": result
        }
        
        # 如果有崩溃事件，添加到日志
        if mental_breakdown:
            log_data["mental_breakdown"] = mental_breakdown
        
        return log_data
    
    def reset(self):
        """重置模拟环境"""
        self.child_state = ChildState()
        self.family_state = FamilyState(current_date=self.start_date)
        self.current_step = 0
        self.current_week = 0

    def _record_dialogue_outcome(self, decision: Dict[str, Any], result: Dict[str, Any]) -> None:
        """记录对话效果，支持拟人化对话记忆"""
        member_id = decision.get("member_id")
        if member_id not in ["father", "mother", "grandfather", "grandmother"]:
            return
        
        member_obj = getattr(self.family_state, member_id, None)
        if not member_obj:
            return
        
        dialogue_system = DialogueSystemFactory.get_dialogue_system(
            member=member_id,
            personality=member_obj.personality
        )
        
        # 使用当前周时间记录对话
        dialogue_system.record_dialogue(
            week=self.current_week,
            date=self.family_state.current_date.isoformat(),
            action_type=decision.get("action_type", "陪伴"),
            dialogue=decision.get("dialogue", ""),
            was_effective=bool(result.get("success", True))
        )
