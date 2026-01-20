"""
自适应类人决策智能体

实现真正类人的决策能力：
1. 决策记忆 - 记住历史决策和效果
2. 反思学习 - 根据反馈调整策略
3. 动态人格 - 情绪状态影响决策
4. 长期规划 - 有明确的教育目标
5. 经验总结 - 从成功/失败中学习
"""

import json
import logging
import random
from datetime import date
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum

from src.agents.base import FamilyAgent
from src.core.state import ChildState, FamilyState
from src.core.config import SimulationConfig
from src.core.llm_client import LLMClientFactory, LLMClient, MockLLMClient

logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """情绪状态"""
    HAPPY = "开心"
    CALM = "平静"
    ANXIOUS = "焦虑"
    FRUSTRATED = "沮丧"
    HOPEFUL = "充满希望"
    WORRIED = "担忧"
    PROUD = "骄傲"
    DISAPPOINTED = "失望"


class EducationPhilosophy(Enum):
    """教育理念（会随经验调整）"""
    STRICT = "严格型"           # 重视成绩和纪律
    NURTURING = "关爱型"        # 重视情感和陪伴
    BALANCED = "平衡型"         # 兼顾学习和快乐
    ACHIEVEMENT = "成就型"      # 重视特长和突破
    FREEDOM = "自由型"          # 尊重孩子自主


@dataclass
class DecisionMemory:
    """决策记忆"""
    week: int
    date: str
    action_type: str
    dialogue: str
    cost: float
    reasoning: str
    
    # 决策时的状态快照
    child_knowledge: float
    child_stress: float
    child_health: float
    relationship: float
    family_savings: float
    
    # 决策效果（下一步更新）
    knowledge_change: float = 0.0
    stress_change: float = 0.0
    health_change: float = 0.0
    relationship_change: float = 0.0
    
    # 评估
    was_successful: Optional[bool] = None
    lesson_learned: str = ""


@dataclass
class ParentPersonality:
    """动态父母人格"""
    
    # 基础性格（相对稳定）
    strictness: float = 0.5      # 严厉程度 0-1
    expressiveness: float = 0.5  # 情感表达 0-1
    patience: float = 0.5        # 耐心程度 0-1
    optimism: float = 0.5        # 乐观程度 0-1
    
    # 动态状态（会变化）
    current_emotion: EmotionalState = EmotionalState.CALM
    stress_level: float = 0.3    # 父母自身压力 0-1
    confidence: float = 0.5      # 教育信心 0-1
    
    # 教育理念（会随经验调整）
    philosophy: EducationPhilosophy = EducationPhilosophy.BALANCED
    
    # 偏好学习（从经验中习得）
    preferred_actions: Dict[str, float] = field(default_factory=dict)  # action -> 成功率
    avoided_actions: Dict[str, float] = field(default_factory=dict)    # action -> 失败率


@dataclass
class EducationGoal:
    """教育目标"""
    description: str
    target_metric: str      # knowledge, relationship, stress, health
    target_value: float
    deadline_age: float     # 目标年龄
    priority: int           # 优先级 1-5
    progress: float = 0.0   # 当前进度 0-1
    status: str = "进行中"   # 进行中、已完成、已放弃


class AdaptiveParentAgent(FamilyAgent):
    """
    自适应类人父母智能体
    
    核心特性：
    1. 决策记忆 - 记住过去的决策和效果
    2. 反思学习 - 分析什么有效什么无效
    3. 动态人格 - 情绪和信心会变化
    4. 长期规划 - 有明确的教育目标
    5. 自我调整 - 根据经验修改策略
    """
    
    # 记忆容量
    MAX_MEMORY_SIZE = 52  # 记住最近1年的决策
    
    # 反思频率（每N周进行一次深度反思）
    REFLECTION_INTERVAL = 4
    
    def __init__(self, model_name: str, member: str = "father", config: SimulationConfig = None):
        super().__init__(model_name, member)
        self.config = config or SimulationConfig()
        
        # 初始化 LLM 客户端
        self._llm_client: Optional[LLMClient] = None
        self._model_config: Optional[Dict[str, str]] = None
        self._is_mock = False
        self._init_llm_client()
        
        # 决策记忆系统
        self.memory: deque[DecisionMemory] = deque(maxlen=self.MAX_MEMORY_SIZE)
        self.total_decisions = 0
        
        # 动态人格
        self.personality = self._init_personality()
        
        # 教育目标
        self.goals: List[EducationGoal] = self._init_goals()
        
        # 经验总结
        self.learned_lessons: List[str] = []
        self.success_patterns: Dict[str, List[str]] = {}  # 情境 -> 有效策略
        self.failure_patterns: Dict[str, List[str]] = {}  # 情境 -> 无效策略
        
        # 上一次状态（用于计算变化）
        self._last_child_state: Optional[ChildState] = None
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        try:
            self._model_config = self.config.get_model_config(self.model_name)
            self._llm_client = LLMClientFactory.create_client(self.config, self.model_name)
            
            if isinstance(self._llm_client, MockLLMClient):
                self._is_mock = True
                logger.warning(f"{self.model_name} 使用 Mock 客户端")
            else:
                logger.info(f"{self.model_name} 自适应Agent LLM客户端初始化成功")
                
        except Exception as e:
            self._is_mock = True
            logger.warning(f"{self.model_name} LLM客户端初始化失败: {e}")
    
    def _init_personality(self) -> ParentPersonality:
        """初始化人格特质（基于角色有所不同）"""
        if self.member == "father":
            return ParentPersonality(
                strictness=0.6 + random.uniform(-0.1, 0.1),
                expressiveness=0.3 + random.uniform(-0.1, 0.1),
                patience=0.5 + random.uniform(-0.1, 0.1),
                optimism=0.5 + random.uniform(-0.1, 0.1),
                philosophy=random.choice([EducationPhilosophy.STRICT, EducationPhilosophy.BALANCED])
            )
        elif self.member == "mother":
            return ParentPersonality(
                strictness=0.4 + random.uniform(-0.1, 0.1),
                expressiveness=0.7 + random.uniform(-0.1, 0.1),
                patience=0.6 + random.uniform(-0.1, 0.1),
                optimism=0.6 + random.uniform(-0.1, 0.1),
                philosophy=random.choice([EducationPhilosophy.NURTURING, EducationPhilosophy.BALANCED])
            )
        elif self.member == "grandfather":
            return ParentPersonality(
                strictness=0.5 + random.uniform(-0.1, 0.1),
                expressiveness=0.4 + random.uniform(-0.1, 0.1),
                patience=0.7 + random.uniform(-0.1, 0.1),
                optimism=0.4 + random.uniform(-0.1, 0.1),
                philosophy=EducationPhilosophy.BALANCED
            )
        else:  # grandmother
            return ParentPersonality(
                strictness=0.2 + random.uniform(-0.1, 0.1),
                expressiveness=0.8 + random.uniform(-0.1, 0.1),
                patience=0.8 + random.uniform(-0.1, 0.1),
                optimism=0.7 + random.uniform(-0.1, 0.1),
                philosophy=EducationPhilosophy.NURTURING
            )
    
    def _init_goals(self) -> List[EducationGoal]:
        """初始化教育目标"""
        return [
            EducationGoal(
                description="培养良好的学习习惯",
                target_metric="knowledge",
                target_value=60.0,
                deadline_age=6.0,
                priority=3
            ),
            EducationGoal(
                description="保持健康快乐成长",
                target_metric="stress",
                target_value=30.0,  # 希望压力保持低于30
                deadline_age=18.0,
                priority=5
            ),
            EducationGoal(
                description="建立亲密的亲子关系",
                target_metric="relationship",
                target_value=80.0,
                deadline_age=12.0,
                priority=4
            ),
            EducationGoal(
                description="小学阶段打好基础",
                target_metric="knowledge",
                target_value=70.0,
                deadline_age=12.0,
                priority=4
            )
        ]
    
    def update_from_outcome(self, outcome: Dict[str, Any]):
        """
        根据决策结果更新记忆和人格
        
        这是实现"学习"的关键方法
        """
        if not self.memory:
            return
        
        last_memory = self.memory[-1]
        
        # 更新决策效果
        state_changes = outcome.get("state_changes", {})
        last_memory.knowledge_change = state_changes.get("knowledge", 0)
        last_memory.stress_change = state_changes.get("stress", 0)
        last_memory.health_change = state_changes.get("physical_health", 0)
        last_memory.relationship_change = state_changes.get(f"{self.member}_relationship", 0)
        
        # 评估决策是否成功
        success = self._evaluate_decision_success(last_memory)
        last_memory.was_successful = success
        
        # 更新偏好
        action = last_memory.action_type
        if success:
            self.personality.preferred_actions[action] = \
                self.personality.preferred_actions.get(action, 0.5) * 0.9 + 0.1
            self.personality.confidence = min(1.0, self.personality.confidence + 0.02)
        else:
            self.personality.avoided_actions[action] = \
                self.personality.avoided_actions.get(action, 0.5) * 0.9 + 0.1
            self.personality.confidence = max(0.1, self.personality.confidence - 0.01)
        
        # 更新情绪
        self._update_emotion(last_memory)
        
        # 定期反思
        if self.total_decisions % self.REFLECTION_INTERVAL == 0:
            self._deep_reflection()
    
    def _evaluate_decision_success(self, memory: DecisionMemory) -> bool:
        """评估决策是否成功"""
        # 综合评分
        score = 0
        
        # 知识增长是好的
        if memory.knowledge_change > 0:
            score += 1
        
        # 压力降低是好的
        if memory.stress_change < 0:
            score += 1
        elif memory.stress_change > 5:
            score -= 2  # 压力大幅上升是坏的
        
        # 关系改善是好的
        if memory.relationship_change > 0:
            score += 1
        elif memory.relationship_change < -2:
            score -= 1  # 关系恶化是坏的
        
        # 健康改善是好的
        if memory.health_change > 0:
            score += 0.5
        
        return score > 0
    
    def _update_emotion(self, memory: DecisionMemory):
        """根据决策效果更新情绪"""
        if memory.was_successful:
            # 成功后的情绪
            if memory.relationship_change > 2:
                self.personality.current_emotion = EmotionalState.HAPPY
            elif memory.knowledge_change > 1:
                self.personality.current_emotion = EmotionalState.PROUD
            else:
                self.personality.current_emotion = EmotionalState.HOPEFUL
            
            self.personality.stress_level = max(0, self.personality.stress_level - 0.05)
        else:
            # 失败后的情绪
            if memory.stress_change > 5:
                self.personality.current_emotion = EmotionalState.WORRIED
            elif memory.relationship_change < -2:
                self.personality.current_emotion = EmotionalState.DISAPPOINTED
            else:
                self.personality.current_emotion = EmotionalState.ANXIOUS
            
            self.personality.stress_level = min(1, self.personality.stress_level + 0.05)
    
    def _deep_reflection(self):
        """
        深度反思：分析历史决策，总结经验教训
        
        这是实现"自我调整"的核心
        """
        if len(self.memory) < 4:
            return
        
        recent_memories = list(self.memory)[-12:]  # 最近12周
        
        # 统计成功/失败模式
        action_stats: Dict[str, Dict[str, int]] = {}
        for mem in recent_memories:
            action = mem.action_type
            if action not in action_stats:
                action_stats[action] = {"success": 0, "failure": 0}
            
            if mem.was_successful:
                action_stats[action]["success"] += 1
            else:
                action_stats[action]["failure"] += 1
        
        # 总结经验
        for action, stats in action_stats.items():
            total = stats["success"] + stats["failure"]
            if total >= 2:
                success_rate = stats["success"] / total
                
                if success_rate >= 0.7:
                    lesson = f"「{action}」效果不错，成功率{success_rate:.0%}"
                    if lesson not in self.learned_lessons:
                        self.learned_lessons.append(lesson)
                        logger.info(f"[{self.model_name}] 习得经验: {lesson}")
                
                elif success_rate <= 0.3:
                    lesson = f"「{action}」效果不好，成功率只有{success_rate:.0%}，需要调整"
                    if lesson not in self.learned_lessons:
                        self.learned_lessons.append(lesson)
                        logger.info(f"[{self.model_name}] 习得教训: {lesson}")
        
        # 根据经验调整教育理念
        self._adjust_philosophy()
    
    def _adjust_philosophy(self):
        """根据经验调整教育理念"""
        if len(self.memory) < 8:
            return
        
        recent = list(self.memory)[-8:]
        
        # 分析孩子状态趋势
        avg_stress_change = sum(m.stress_change for m in recent) / len(recent)
        avg_knowledge_change = sum(m.knowledge_change for m in recent) / len(recent)
        avg_relationship_change = sum(m.relationship_change for m in recent) / len(recent)
        
        old_philosophy = self.personality.philosophy
        
        # 如果压力持续上升，转向关爱型
        if avg_stress_change > 2:
            self.personality.philosophy = EducationPhilosophy.NURTURING
            self.personality.strictness = max(0.2, self.personality.strictness - 0.1)
        
        # 如果知识停滞且关系良好，可以适当严格
        elif avg_knowledge_change < 0.2 and avg_relationship_change > 0:
            if self.personality.philosophy == EducationPhilosophy.NURTURING:
                self.personality.philosophy = EducationPhilosophy.BALANCED
        
        # 如果一切顺利，保持平衡
        elif avg_stress_change < 0 and avg_knowledge_change > 0.3:
            self.personality.philosophy = EducationPhilosophy.BALANCED
        
        if old_philosophy != self.personality.philosophy:
            logger.info(f"[{self.model_name}] 教育理念调整: {old_philosophy.value} -> {self.personality.philosophy.value}")
    
    def _build_adaptive_prompt(self, child_state: ChildState, family_state: FamilyState, event: str) -> str:
        """
        构建自适应提示词
        
        关键区别：包含记忆、反思、情绪、目标
        """
        age = child_state.calculate_age(family_state.current_date)
        member_info = self._get_member_info()
        
        # 基础身份
        prompt = f"""你是一个真实的中国工薪阶层{member_info['role']}，正在养育你的{member_info['child_call']}。

## 你的个人状态
- 当前情绪：{self.personality.current_emotion.value}
- 压力水平：{'高' if self.personality.stress_level > 0.6 else '中' if self.personality.stress_level > 0.3 else '低'}
- 教育信心：{'充足' if self.personality.confidence > 0.6 else '一般' if self.personality.confidence > 0.3 else '不足'}
- 教育理念：{self.personality.philosophy.value}
- 性格特点：{'严厉' if self.personality.strictness > 0.6 else '温和'}，{'善于表达' if self.personality.expressiveness > 0.6 else '内敛'}

"""
        
        # 添加记忆回顾
        if self.memory:
            recent = list(self.memory)[-3:]
            prompt += "## 最近的教育经历\n"
            for mem in recent:
                result = "效果不错" if mem.was_successful else "效果一般"
                prompt += f"- 第{mem.week}周：采用「{mem.action_type}」，{result}\n"
            prompt += "\n"
        
        # 添加经验总结
        if self.learned_lessons:
            prompt += "## 你总结的经验教训\n"
            for lesson in self.learned_lessons[-5:]:
                prompt += f"- {lesson}\n"
            prompt += "\n"
        
        # 添加教育目标
        active_goals = [g for g in self.goals if g.status == "进行中"]
        if active_goals:
            prompt += "## 你的教育目标\n"
            for goal in sorted(active_goals, key=lambda g: -g.priority)[:3]:
                prompt += f"- {goal.description}（优先级{goal.priority}）\n"
            prompt += "\n"
        
        # 添加偏好和回避
        if self.personality.preferred_actions:
            top_preferred = sorted(self.personality.preferred_actions.items(), 
                                   key=lambda x: -x[1])[:3]
            if top_preferred:
                prompt += "## 你发现有效的方式\n"
                for action, rate in top_preferred:
                    if rate > 0.6:
                        prompt += f"- {action}（经常有效）\n"
                prompt += "\n"
        
        if self.personality.avoided_actions:
            top_avoided = sorted(self.personality.avoided_actions.items(), 
                                 key=lambda x: -x[1])[:2]
            if top_avoided:
                prompt += "## 你发现效果不好的方式\n"
                for action, rate in top_avoided:
                    if rate > 0.6:
                        prompt += f"- {action}（经常无效，尽量避免）\n"
                prompt += "\n"
        
        # 当前情境
        relationship_key = f"{self.member}_relationship"
        relationship_value = getattr(child_state, relationship_key, 70.0)
        
        prompt += f"""## 当前情况
- 日期：{family_state.current_date.strftime('%Y年%m月')}
- {member_info['child_call']}年龄：{age:.1f}岁
- 知识水平：{child_state.knowledge:.0f}/100
- 压力状态：{child_state.stress:.0f}/100 {'⚠️需要关注' if child_state.stress > 60 else ''}
- 与你的关系：{relationship_value:.0f}/100
- 身体健康：{child_state.physical_health:.0f}/100
- 兴趣偏好：{', '.join(child_state.interests.get_top_interests(3)) if hasattr(child_state, 'interests') else '未知'}
- 当前敏感期：{', '.join(getattr(child_state, 'development_sensitivity', None).get_active_sensitivities().keys()) if hasattr(child_state, 'development_sensitivity') else '无'}
- 家庭存款：{family_state.family_savings:.0f}元

本周事件：{event if event else '平常的一周'}

## 你的思考
作为{member_info['role']}，基于你的经验和当前情况，你会怎么做？
- 考虑你过去的经验教训
- 考虑你当前的情绪状态
- 考虑孩子的实际需求
- 考虑家庭经济情况

请用JSON格式回答：
```json
{{
    "inner_thought": "你内心的真实想法（考虑到过去的经验和当前情绪）",
    "action_type": "你的选择",
    "dialogue": "你对{member_info['child_call']}说的话",
    "cost": 花费金额,
    "reasoning": "为什么做这个选择（基于你的经验和目标）"
}}
```

注意：你是一个真实的{member_info['role']}，会犯错、会焦虑、会反思、会调整。不要像AI那样完美，要像真人一样。
"""
        
        return prompt
    
    def _get_member_info(self) -> Dict[str, str]:
        """获取成员信息"""
        info = {
            "father": {"role": "父亲", "child_call": "女儿"},
            "mother": {"role": "母亲", "child_call": "女儿"},
            "grandfather": {"role": "爷爷", "child_call": "孙女"},
            "grandmother": {"role": "奶奶", "child_call": "孙女"}
        }
        return info.get(self.member, {"role": "家长", "child_call": "孩子"})
    
    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        做出决策（核心方法）
        """
        self.total_decisions += 1
        
        # 如果有上一次状态，更新记忆效果
        if self._last_child_state and self.memory:
            self._update_memory_effects(child_state)
        
        # 更新目标进度
        self._update_goal_progress(child_state, family_state)
        
        # 生成决策
        if self._is_mock or self._llm_client is None:
            decision = await self._adaptive_fallback_decision(child_state, family_state, event)
        else:
            decision = await self._llm_decision(child_state, family_state, event)
        
        # 记录决策
        self._record_decision(decision, child_state, family_state)
        
        # 保存当前状态
        self._last_child_state = ChildState(
            birth_date=child_state.birth_date,
            iq=child_state.iq,
            knowledge=child_state.knowledge,
            stress=child_state.stress,
            father_relationship=child_state.father_relationship,
            mother_relationship=child_state.mother_relationship,
            grandfather_relationship=child_state.grandfather_relationship,
            grandmother_relationship=child_state.grandmother_relationship,
            physical_health=child_state.physical_health
        )
        
        return decision
    
    def _update_memory_effects(self, current_state: ChildState):
        """根据当前状态更新上一次决策的效果"""
        if not self.memory or not self._last_child_state:
            return
        
        last_mem = self.memory[-1]
        last_state = self._last_child_state
        
        # 计算变化
        last_mem.knowledge_change = current_state.knowledge - last_state.knowledge
        last_mem.stress_change = current_state.stress - last_state.stress
        last_mem.health_change = current_state.physical_health - last_state.physical_health
        
        relationship_key = f"{self.member}_relationship"
        last_mem.relationship_change = (
            getattr(current_state, relationship_key, 0) - 
            getattr(last_state, relationship_key, 0)
        )
        
        # 评估并学习
        last_mem.was_successful = self._evaluate_decision_success(last_mem)
        self._update_emotion(last_mem)
        
        # 更新偏好
        action = last_mem.action_type
        if last_mem.was_successful:
            self.personality.preferred_actions[action] = \
                self.personality.preferred_actions.get(action, 0.5) * 0.8 + 0.2
        else:
            self.personality.avoided_actions[action] = \
                self.personality.avoided_actions.get(action, 0.5) * 0.8 + 0.2
        
        # 定期反思
        if self.total_decisions % self.REFLECTION_INTERVAL == 0:
            self._deep_reflection()
    
    def _update_goal_progress(self, child_state: ChildState, family_state: FamilyState):
        """更新教育目标进度"""
        age = child_state.calculate_age(family_state.current_date)
        
        for goal in self.goals:
            if goal.status != "进行中":
                continue
            
            # 获取当前值
            if goal.target_metric == "knowledge":
                current_value = child_state.knowledge
            elif goal.target_metric == "stress":
                current_value = 100 - child_state.stress  # 压力越低越好
            elif goal.target_metric == "relationship":
                current_value = getattr(child_state, f"{self.member}_relationship", 50)
            elif goal.target_metric == "health":
                current_value = child_state.physical_health
            else:
                continue
            
            # 计算进度
            goal.progress = min(1.0, current_value / goal.target_value)
            
            # 检查是否完成或过期
            if goal.progress >= 1.0:
                goal.status = "已完成"
                self.learned_lessons.append(f"达成目标：{goal.description}")
                self.personality.confidence = min(1.0, self.personality.confidence + 0.1)
            elif age > goal.deadline_age:
                goal.status = "已放弃"
                self.personality.confidence = max(0.1, self.personality.confidence - 0.05)
    
    def _record_decision(self, decision: Dict[str, Any], child_state: ChildState, family_state: FamilyState):
        """记录决策到记忆"""
        relationship_key = f"{self.member}_relationship"
        
        memory = DecisionMemory(
            week=self.total_decisions,
            date=family_state.current_date.strftime('%Y-%m-%d'),
            action_type=decision.get("action_type", "陪伴"),
            dialogue=decision.get("dialogue", ""),
            cost=decision.get("cost", 0),
            reasoning=decision.get("reasoning", ""),
            child_knowledge=child_state.knowledge,
            child_stress=child_state.stress,
            child_health=child_state.physical_health,
            relationship=getattr(child_state, relationship_key, 50),
            family_savings=family_state.family_savings
        )
        
        self.memory.append(memory)
    
    async def _llm_decision(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """使用 LLM 生成决策"""
        try:
            prompt = self._build_adaptive_prompt(child_state, family_state, event)
            
            messages = [
                {"role": "system", "content": "你是一个真实的中国家长，正在做出教育决策。请根据你的经验和当前情况做出选择。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._llm_client.chat_completion(
                messages=messages,
                model=self._model_config["model"],
                temperature=0.8,  # 稍高的温度，更像人类
                max_tokens=600
            )
            
            content = response["choices"][0]["message"]["content"]
            decision = self._parse_decision(content)
            
            decision["llm_generated"] = True
            decision["model"] = self.model_name
            decision["agent_emotion"] = self.personality.current_emotion.value
            decision["agent_philosophy"] = self.personality.philosophy.value
            
            logger.info(f"[{self.model_name}] 自适应决策: {decision['action_type']} (情绪:{self.personality.current_emotion.value})")
            return decision
            
        except Exception as e:
            logger.warning(f"[{self.model_name}] LLM决策失败: {e}")
            return await self._adaptive_fallback_decision(child_state, family_state, event)
    
    def _parse_decision(self, content: str) -> Dict[str, Any]:
        """解析LLM返回的决策"""
        import re
        
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        
        if json_match:
            try:
                decision = json.loads(json_match.group())
                
                # 确保必要字段存在
                if "action_type" not in decision:
                    decision["action_type"] = "陪伴"
                if "dialogue" not in decision:
                    decision["dialogue"] = "孩子，今天我们一起待一会儿。"
                if "cost" not in decision:
                    decision["cost"] = 0
                if "reasoning" not in decision:
                    decision["reasoning"] = decision.get("inner_thought", "")
                
                return decision
                
            except json.JSONDecodeError:
                pass
        
        # 解析失败，使用默认
        return {
            "action_type": "陪伴",
            "dialogue": "孩子，今天爸爸陪陪你。",
            "cost": 0,
            "reasoning": "无法解析LLM响应"
        }
    
    # 年龄阶段可用行为（与 llm_agents.py 保持一致）
    AGE_APPROPRIATE_ACTIONS = {
        "infant": [
            "亲子互动", "日常照料", "感官刺激", "户外活动", "早期阅读",
            "安抚陪伴", "社交接触", "陪伴", "启蒙教育", "健康教育", "游戏互动", "鼓励"
        ],
        "preschool": [
            "陪伴", "启蒙教育", "游戏互动", "简单辅导", "鼓励", "简单兴趣培养",
            "健康教育", "创新活动", "沟通", "户外活动", "早期阅读", "亲子互动"
        ],
        "primary": [
            "辅导", "鼓励", "花钱培训", "陪伴", "严格要求", "监督学习",
            "健康教育", "创新活动", "个性化计划", "实践活动", "沟通",
            "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养"
        ]
    }
    
    async def _adaptive_fallback_decision(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        自适应降级决策
        
        与普通Mock不同：会参考记忆和人格，并考虑年龄适配
        """
        member_info = self._get_member_info()
        age = child_state.calculate_age(family_state.current_date)
        age_group = child_state.get_age_group(family_state.current_date)
        
        # 获取年龄适配的可用行为
        available_actions = self.AGE_APPROPRIATE_ACTIONS.get(age_group, self.AGE_APPROPRIATE_ACTIONS["primary"])
        
        # 婴幼儿期特殊处理
        if age_group == "infant":
            if age < 1:
                # 0-1岁：主要是照料和亲子互动
                preferred = ["亲子互动", "日常照料", "安抚陪伴", "感官刺激"]
            else:
                # 1-3岁：可以增加更多互动
                preferred = ["亲子互动", "户外活动", "游戏互动", "早期阅读", "启蒙教育"]
        elif age_group == "preschool":
            # 3-6岁：根据情绪和理念选择
            if self.personality.current_emotion in [EmotionalState.WORRIED, EmotionalState.ANXIOUS]:
                preferred = ["陪伴", "游戏互动", "户外活动"]
            elif self.personality.philosophy == EducationPhilosophy.STRICT:
                preferred = ["简单辅导", "启蒙教育", "简单兴趣培养"]
            else:
                preferred = ["游戏互动", "启蒙教育", "陪伴", "鼓励"]
        else:
            # 6岁以上：原有逻辑
            if self.personality.current_emotion in [EmotionalState.WORRIED, EmotionalState.ANXIOUS]:
                preferred = ["陪伴", "鼓励", "游戏互动"]
            elif self.personality.current_emotion == EmotionalState.FRUSTRATED:
                if self.personality.strictness > 0.5:
                    preferred = ["严格要求", "监督学习", "辅导"]
                else:
                    preferred = ["沟通", "陪伴"]
            elif self.personality.current_emotion == EmotionalState.PROUD:
                if self.memory:
                    last_action = self.memory[-1].action_type
                    preferred = [last_action, "鼓励", "启蒙教育"]
                else:
                    preferred = ["鼓励", "启蒙教育"]
            else:
                if self.personality.philosophy == EducationPhilosophy.STRICT:
                    preferred = ["辅导", "监督学习", "严格要求"]
                elif self.personality.philosophy == EducationPhilosophy.NURTURING:
                    preferred = ["陪伴", "鼓励", "游戏互动", "沟通"]
                else:
                    preferred = ["陪伴", "辅导", "启蒙教育", "鼓励"]
        
        # 过滤掉不在可用列表中的行为
        preferred = [a for a in preferred if a in available_actions]
        
        # 排除失败率高的选项
        for action in list(preferred):
            if self.personality.avoided_actions.get(action, 0) > 0.7:
                preferred.remove(action)
        
        # 根据孩子状态调整
        if child_state.stress > 70:
            stress_relief = ["陪伴", "游戏互动", "鼓励", "亲子互动", "安抚陪伴"]
            preferred = [a for a in stress_relief if a in available_actions]
        elif child_state.knowledge < 40 and age >= 6:
            if "辅导" in available_actions and "辅导" not in preferred:
                preferred.insert(0, "辅导")
        
        # 确保有选项
        if not preferred:
            preferred = [available_actions[0]] if available_actions else ["陪伴"]
        
        action_type = random.choice(preferred[:3])
        
        # 生成对话
        dialogue = self._generate_dialogue(action_type, member_info, age)
        
        # 确定花费
        cost = self._determine_cost(action_type, family_state)
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost,
            "reasoning": f"基于{self.personality.philosophy.value}理念和{self.personality.current_emotion.value}情绪",
            "llm_generated": False,
            "model": self.model_name,
            "agent_emotion": self.personality.current_emotion.value,
            "agent_philosophy": self.personality.philosophy.value
        }
    
    def _generate_dialogue(self, action_type: str, member_info: Dict[str, str], age: float) -> str:
        """生成对话（根据年龄调整语言风格）"""
        child = member_info["child_call"]
        role = member_info["role"]
        
        # 婴儿期对话模板（0-1岁）- 温柔的语调，简单的词语
        if age < 1:
            infant_templates = {
                "亲子互动": [
                    f"（轻轻抱着孩子）宝宝乖，{role}在呢。",
                    f"（温柔地）小宝贝，笑一个给{role}看。",
                    f"（抱起孩子）来，{role}抱抱。"
                ],
                "日常照料": [
                    f"（轻声说）宝宝乖，{role}给你换尿布。",
                    f"（哄着）乖宝宝，吃饱饱，睡香香。",
                    f"（温柔地）宝贝，{role}给你洗个澡。"
                ],
                "安抚陪伴": [
                    f"（轻轻哼歌）宝宝不哭，{role}在这儿呢。",
                    f"（拍着孩子）乖，乖，{role}陪着你。",
                    f"（温柔地）宝贝别怕，{role}抱着你。"
                ],
                "感官刺激": [
                    f"（拿起玩具）宝宝看，这是什么呀？",
                    f"（放音乐）听，好听的音乐。",
                    f"（指着东西）宝宝看，红色的花。"
                ],
                "户外活动": [
                    f"（推婴儿车）宝宝，咱们出去晒太阳。",
                    f"（抱着孩子）看，外面的花花草草。",
                    f"（走在公园）呼吸新鲜空气，真舒服。"
                ]
            }
            options = infant_templates.get(action_type, [f"（轻声说）宝宝乖，{role}在。"])
            return random.choice(options)
        
        # 幼儿期对话模板（1-3岁）- 简单句子，充满爱意
        elif age < 3:
            toddler_templates = {
                "亲子互动": [
                    f"宝贝，来{role}这儿，抱抱。",
                    f"小乖乖，{role}陪你玩好不好？",
                    f"宝贝真棒！{role}爱你。"
                ],
                "户外活动": [
                    f"宝贝，咱们去公园玩滑梯好不好？",
                    f"走，{role}带你出去看小鸟。",
                    f"外面天气真好，咱们去晒太阳。"
                ],
                "游戏互动": [
                    f"宝贝，来玩积木！",
                    f"咱们捉迷藏好不好？",
                    f"{role}陪你玩球球。"
                ],
                "早期阅读": [
                    f"宝贝，{role}给你讲故事。",
                    f"来看看这本绘本，好漂亮的画。",
                    f"这是小兔子，会蹦蹦跳跳。"
                ],
                "启蒙教育": [
                    f"宝贝，这是苹果，红红的。",
                    f"来，跟{role}念，一二三。",
                    f"看，这是小狗，汪汪叫。"
                ]
            }
            options = toddler_templates.get(action_type, [f"宝贝，{role}陪你。"])
            return random.choice(options)
        
        # 学龄前及以上对话模板
        templates = {
            "陪伴": [
                f"{child}，今天{role}哪儿也不去，就陪着你。",
                f"{child}，咱们出去走走吧。",
                f"{child}，今天{role}专门陪你。"
            ],
            "辅导": [
                f"{child}，来，{role}帮你看看功课。",
                f"{child}，有不懂的题目吗？{role}来帮你。",
                f"{child}，咱们一起学习一会儿。"
            ],
            "鼓励": [
                f"{child}，{role}相信你可以的！",
                f"{child}，你最近进步很大，继续加油！",
                f"{child}，不管怎样，{role}都为你骄傲。"
            ],
            "游戏互动": [
                f"{child}，今天咱们玩个游戏吧。",
                f"{child}，{role}陪你玩一会儿。",
                f"{child}，想玩什么？{role}陪你。"
            ],
            "启蒙教育": [
                f"{child}，今天{role}教你认识一些新东西。",
                f"{child}，来，咱们一起学点有趣的。",
                f"{child}，{role}给你讲个故事吧。"
            ],
            "严格要求": [
                f"{child}，最近学习不能松懈，要认真一点。",
                f"{child}，{role}是为你好，该努力了。",
                f"{child}，现在不努力，以后怎么办？"
            ],
            "沟通": [
                f"{child}，最近怎么样？有什么想和{role}说的吗？",
                f"{child}，{role}想和你聊聊。",
                f"{child}，有心事可以告诉{role}。"
            ],
            "简单辅导": [
                f"{child}，来，{role}帮你复习一下。",
                f"{child}，这道题{role}教你。"
            ],
            "户外活动": [
                f"{child}，天气不错，咱们出去玩。",
                f"{child}，去公园跑跑步吧。"
            ],
            "健康教育": [
                f"{child}，要多运动，身体才会棒。",
                f"{child}，早睡早起身体好。"
            ]
        }
        
        options = templates.get(action_type, [f"{child}，今天{role}陪你。"])
        return random.choice(options)
    
    def _determine_cost(self, action_type: str, family_state: FamilyState) -> float:
        """确定花费"""
        base_costs = {
            # 婴幼儿期行为
            "亲子互动": (0, 20),
            "日常照料": (50, 150),  # 尿布、奶粉等日常开销
            "感官刺激": (0, 50),    # 简单玩具
            "户外活动": (0, 30),
            "早期阅读": (0, 50),    # 绘本
            "安抚陪伴": (0, 0),
            "社交接触": (0, 30),
            # 通用行为
            "陪伴": (0, 50),
            "辅导": (0, 0),
            "鼓励": (0, 0),
            "游戏互动": (0, 30),
            "启蒙教育": (0, 30),
            "花钱培训": (150, 300),
            "简单兴趣培养": (80, 150),
            "严格要求": (0, 0),
            "监督学习": (0, 0),
            "沟通": (0, 0),
            "健康教育": (0, 50),
            "简单辅导": (0, 0),
            "创新活动": (50, 100),
            "实践活动": (30, 80),
        }
        
        min_cost, max_cost = base_costs.get(action_type, (0, 50))
        
        # 经济约束
        if family_state.family_savings < 5000:
            return 0
        elif family_state.family_savings < 10000:
            max_cost = min(max_cost, 50)
        
        return random.uniform(min_cost, max_cost)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取Agent状态摘要（用于调试和可视化）"""
        return {
            "model": self.model_name,
            "member": self.member,
            "total_decisions": self.total_decisions,
            "memory_size": len(self.memory),
            "emotion": self.personality.current_emotion.value,
            "philosophy": self.personality.philosophy.value,
            "confidence": self.personality.confidence,
            "stress_level": self.personality.stress_level,
            "learned_lessons": len(self.learned_lessons),
            "preferred_actions": list(self.personality.preferred_actions.keys()),
            "avoided_actions": list(self.personality.avoided_actions.keys()),
            "active_goals": len([g for g in self.goals if g.status == "进行中"]),
            "completed_goals": len([g for g in self.goals if g.status == "已完成"])
        }
