"""
决策心理系统

实现更拟人化的决策过程：
1. 决策犹豫 - 不确定时的纠结
2. 后悔机制 - 对过去决策的反思
3. 认知偏差 - 攀比心理、沉没成本等
4. 信息不对称 - 父母不完全了解孩子
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class CognitiveBias(str, Enum):
    """认知偏差类型"""
    COMPARISON = "攀比心理"          # 看到别人家孩子好就焦虑
    SUNK_COST = "沉没成本谬误"        # 已经投入了就不想放弃
    AVAILABILITY = "可得性偏差"       # 最近听说的影响判断
    CONFIRMATION = "确认偏见"         # 只看到支持自己观点的
    OVERCONFIDENCE = "过度自信"       # 对自己的教育方式过于自信
    LOSS_AVERSION = "损失厌恶"        # 对可能的损失过度担心
    ANCHORING = "锚定效应"            # 被初始信息影响


@dataclass
class RegretRecord:
    """后悔记录"""
    week: int
    decision: str
    expected_outcome: str
    actual_outcome: str
    regret_intensity: float  # 0-1，后悔程度
    lesson_learned: str
    would_do_differently: str


@dataclass
class HesitationRecord:
    """犹豫记录"""
    week: int
    options: List[str]
    final_choice: str
    hesitation_time: float  # 模拟的犹豫时间（秒）
    confidence_before: float
    confidence_after: float
    changed_mind: bool


class DecisionPsychology:
    """
    决策心理系统
    
    模拟真实人类的决策心理过程
    """
    
    def __init__(self, member: str, personality_traits: Dict[str, float] = None):
        """
        初始化决策心理系统
        
        参数:
            member: 家庭成员
            personality_traits: 性格特质
        """
        self.member = member
        self.personality_traits = personality_traits or {}
        
        # 决策信心 (0-1)
        self.confidence = 0.6
        
        # 认知偏差倾向
        self.bias_susceptibility: Dict[CognitiveBias, float] = self._init_biases()
        
        # 后悔历史
        self.regret_history: deque = deque(maxlen=24)  # 最近24周
        
        # 犹豫历史
        self.hesitation_history: deque = deque(maxlen=12)
        
        # 记忆中的"别人家的孩子"
        self.comparison_targets: List[Dict[str, Any]] = []
        
        # 对孩子状态的认知（可能与实际不符）
        self.perceived_child_state: Dict[str, float] = {}
        
        # 焦虑水平
        self.anxiety_level: float = 0.3
    
    def _init_biases(self) -> Dict[CognitiveBias, float]:
        """初始化认知偏差倾向"""
        biases = {}
        
        # 不同成员有不同的偏差倾向
        if self.member == "father":
            biases[CognitiveBias.COMPARISON] = 0.4
            biases[CognitiveBias.SUNK_COST] = 0.5
            biases[CognitiveBias.OVERCONFIDENCE] = 0.4
            biases[CognitiveBias.LOSS_AVERSION] = 0.5
        elif self.member == "mother":
            biases[CognitiveBias.COMPARISON] = 0.6  # 更容易攀比
            biases[CognitiveBias.AVAILABILITY] = 0.5
            biases[CognitiveBias.CONFIRMATION] = 0.4
            biases[CognitiveBias.LOSS_AVERSION] = 0.6
        elif self.member == "grandfather":
            biases[CognitiveBias.ANCHORING] = 0.6  # 更受旧观念影响
            biases[CognitiveBias.CONFIRMATION] = 0.5
            biases[CognitiveBias.OVERCONFIDENCE] = 0.3
        else:  # grandmother
            biases[CognitiveBias.LOSS_AVERSION] = 0.7  # 更担心孙女受苦
            biases[CognitiveBias.AVAILABILITY] = 0.4
        
        return biases
    
    def experience_hesitation(self, 
                               options: List[Dict[str, Any]],
                               child_state: Any,
                               family_state: Any,
                               week: int) -> Tuple[Dict[str, Any], bool]:
        """
        经历决策犹豫
        
        参数:
            options: 可选的决策选项
            child_state: 孩子状态
            family_state: 家庭状态
            week: 当前周数
            
        返回:
            (最终选择, 是否改变了主意)
        """
        if len(options) < 2:
            return options[0] if options else None, False
        
        # 计算初始偏好
        initial_scores = self._score_options(options, child_state, family_state)
        initial_choice = max(initial_scores.keys(), key=lambda x: initial_scores[x])
        initial_confidence = initial_scores[initial_choice]
        
        # 犹豫条件：
        # 1. 最高分和次高分接近
        # 2. 信心不足
        # 3. 有认知偏差干扰
        sorted_scores = sorted(initial_scores.values(), reverse=True)
        score_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        
        # 计算犹豫概率
        hesitation_probability = 0.0
        hesitation_probability += 0.3 if score_gap < 0.2 else 0.0
        hesitation_probability += 0.2 if self.confidence < 0.5 else 0.0
        hesitation_probability += 0.2 if self.anxiety_level > 0.6 else 0.0
        
        changed_mind = False
        final_choice = initial_choice
        
        if random.random() < hesitation_probability:
            # 经历犹豫
            logger.debug(f"[{self.member}] 经历决策犹豫...")
            
            # 可能改变主意
            if random.random() < 0.3:
                # 选择第二选项
                other_options = [k for k in initial_scores.keys() if k != initial_choice]
                if other_options:
                    final_choice = random.choice(other_options)
                    changed_mind = True
                    logger.info(f"[{self.member}] 改变主意: {initial_choice} -> {final_choice}")
        
        # 记录犹豫
        record = HesitationRecord(
            week=week,
            options=[o.get("action_type", "") for o in options],
            final_choice=final_choice,
            hesitation_time=random.uniform(1, 10),  # 模拟犹豫时间
            confidence_before=initial_confidence,
            confidence_after=initial_scores.get(final_choice, initial_confidence),
            changed_mind=changed_mind
        )
        self.hesitation_history.append(record)
        
        # 找到对应的完整选项
        for option in options:
            if option.get("action_type") == final_choice:
                return option, changed_mind
        
        return options[0], changed_mind
    
    def _score_options(self, 
                        options: List[Dict[str, Any]],
                        child_state: Any,
                        family_state: Any) -> Dict[str, float]:
        """为选项打分"""
        scores = {}
        
        for option in options:
            action_type = option.get("action_type", "")
            score = 0.5  # 基础分
            
            # 根据孩子状态调整
            if child_state.stress > 70:
                if action_type in ["陪伴", "游戏互动", "鼓励"]:
                    score += 0.3
                elif action_type in ["严格要求", "监督学习"]:
                    score -= 0.2
            
            if child_state.knowledge < 40:
                if action_type in ["辅导", "简单辅导", "启蒙教育"]:
                    score += 0.2
            
            # 经济因素
            cost = option.get("cost", 0)
            if family_state.family_savings < 10000 and cost > 100:
                score -= 0.3
            
            # 认知偏差影响
            score = self._apply_cognitive_biases(score, action_type, child_state)
            
            scores[action_type] = max(0.0, min(1.0, score))
        
        return scores
    
    def _apply_cognitive_biases(self, 
                                 score: float, 
                                 action_type: str,
                                 child_state: Any) -> float:
        """应用认知偏差"""
        # 攀比心理
        if CognitiveBias.COMPARISON in self.bias_susceptibility:
            if self.comparison_targets:
                comparison_bias = self.bias_susceptibility[CognitiveBias.COMPARISON]
                if action_type in ["花钱培训", "辅导", "严格要求"]:
                    score += comparison_bias * 0.2 * self.anxiety_level
        
        # 沉没成本
        if CognitiveBias.SUNK_COST in self.bias_susceptibility:
            # 如果之前在某方面投入很多，倾向于继续
            sunk_cost_bias = self.bias_susceptibility[CognitiveBias.SUNK_COST]
            # 简化：假设之前如果选了培训就倾向于继续
            if action_type == "花钱培训":
                score += sunk_cost_bias * 0.15
        
        # 损失厌恶
        if CognitiveBias.LOSS_AVERSION in self.bias_susceptibility:
            loss_aversion = self.bias_susceptibility[CognitiveBias.LOSS_AVERSION]
            # 担心孩子落后
            if child_state.knowledge < 50:
                if action_type in ["辅导", "监督学习"]:
                    score += loss_aversion * 0.15
        
        return score
    
    def experience_regret(self,
                          decision: str,
                          expected: str,
                          actual: Dict[str, Any],
                          week: int) -> Optional[RegretRecord]:
        """
        经历后悔
        
        参数:
            decision: 做出的决策
            expected: 期望的结果
            actual: 实际的结果
            week: 当前周数
            
        返回:
            后悔记录，如果没有后悔返回None
        """
        # 判断是否应该后悔
        success = actual.get("success", True)
        stress_change = actual.get("state_changes", {}).get("stress", 0)
        relationship_change = sum([
            actual.get("state_changes", {}).get(f"{m}_relationship", 0)
            for m in ["father", "mother", "grandfather", "grandmother"]
        ]) / 4.0
        
        # 后悔条件
        should_regret = False
        regret_intensity = 0.0
        
        if not success:
            should_regret = True
            regret_intensity = 0.6
        elif stress_change > 5:
            should_regret = random.random() < 0.5
            regret_intensity = 0.4
        elif relationship_change < -2:
            should_regret = random.random() < 0.4
            regret_intensity = 0.5
        
        if not should_regret:
            return None
        
        # 生成后悔记录
        alternatives = self._generate_alternatives(decision)
        lesson = self._generate_lesson(decision, actual)
        
        record = RegretRecord(
            week=week,
            decision=decision,
            expected_outcome=expected,
            actual_outcome=str(actual.get("message", "")),
            regret_intensity=regret_intensity,
            lesson_learned=lesson,
            would_do_differently=alternatives
        )
        
        self.regret_history.append(record)
        
        # 后悔会降低信心
        self.confidence = max(0.1, self.confidence - regret_intensity * 0.1)
        
        # 后悔会增加焦虑
        self.anxiety_level = min(1.0, self.anxiety_level + regret_intensity * 0.1)
        
        logger.info(f"[{self.member}] 经历后悔: {decision} -> {lesson}")
        
        return record
    
    def _generate_alternatives(self, decision: str) -> str:
        """生成替代方案"""
        alternatives = {
            "严格要求": "也许应该多鼓励，少批评",
            "监督学习": "可能陪伴比监督更有效",
            "花钱培训": "也许在家辅导就够了",
            "辅导": "可能应该让孩子自己探索",
            "游戏互动": "也许应该更注重学习"
        }
        return alternatives.get(decision, "下次应该更慎重考虑")
    
    def _generate_lesson(self, decision: str, actual: Dict[str, Any]) -> str:
        """生成教训"""
        if not actual.get("success"):
            return f"「{decision}」不适合当前情况"
        
        stress_change = actual.get("state_changes", {}).get("stress", 0)
        if stress_change > 5:
            return "给孩子的压力太大了"
        
        return "需要更好地理解孩子的需求"
    
    def add_comparison_target(self, 
                               name: str,
                               achievement: str,
                               source: str = "邻居"):
        """
        添加"别人家的孩子"
        
        参数:
            name: 名字
            achievement: 成就
            source: 来源（邻居、亲戚、同事等）
        """
        target = {
            "name": name,
            "achievement": achievement,
            "source": source,
            "mentioned_count": 1
        }
        
        # 检查是否已存在
        for existing in self.comparison_targets:
            if existing["name"] == name:
                existing["mentioned_count"] += 1
                return
        
        self.comparison_targets.append(target)
        
        # 触发攀比焦虑
        if CognitiveBias.COMPARISON in self.bias_susceptibility:
            self.anxiety_level = min(1.0, self.anxiety_level + 0.1)
            logger.debug(f"[{self.member}] 听说{source}的{name}{achievement}，有点焦虑")
    
    def update_perceived_state(self, 
                                actual_state: Any,
                                awareness: float = 0.7):
        """
        更新对孩子状态的认知
        
        参数:
            actual_state: 孩子的实际状态
            awareness: 家长的觉察程度 (0-1)
        """
        # 家长不一定完全了解孩子的真实状态
        noise = 1.0 - awareness
        
        self.perceived_child_state = {
            "stress": actual_state.stress + random.uniform(-10, 10) * noise,
            "knowledge": actual_state.knowledge + random.uniform(-5, 5) * noise,
            "happiness": 100 - actual_state.stress + random.uniform(-15, 15) * noise,
            # 关系认知可能有偏差
            f"{self.member}_relationship": getattr(
                actual_state, f"{self.member}_relationship", 70
            ) + random.uniform(-10, 10) * noise
        }
        
        # 确保在合理范围内
        for key in self.perceived_child_state:
            self.perceived_child_state[key] = max(0, min(100, self.perceived_child_state[key]))
    
    def learn_from_experience(self):
        """从经验中学习"""
        # 分析最近的后悔记录
        if len(self.regret_history) >= 3:
            recent_regrets = list(self.regret_history)[-6:]
            
            # 统计哪些决策容易后悔
            regret_actions = {}
            for record in recent_regrets:
                action = record.decision
                regret_actions[action] = regret_actions.get(action, 0) + 1
            
            # 如果某个行为多次后悔，降低对它的偏好
            for action, count in regret_actions.items():
                if count >= 2:
                    logger.info(f"[{self.member}] 学到教训: 「{action}」经常效果不好")
        
        # 分析犹豫模式
        if len(self.hesitation_history) >= 5:
            changed_mind_count = sum(
                1 for h in list(self.hesitation_history)[-10:]
                if h.changed_mind
            )
            
            # 如果经常改变主意，可能需要提高信心
            if changed_mind_count > 5:
                logger.debug(f"[{self.member}] 决策不够果断，需要提高信心")
    
    def get_psychology_summary(self) -> Dict[str, Any]:
        """获取心理状态摘要"""
        return {
            "决策信心": f"{self.confidence:.2f}",
            "焦虑水平": f"{self.anxiety_level:.2f}",
            "后悔次数": len(self.regret_history),
            "犹豫次数": len(self.hesitation_history),
            "改变主意次数": sum(1 for h in self.hesitation_history if h.changed_mind),
            "攀比对象数": len(self.comparison_targets),
            "主要认知偏差": [
                bias.value for bias, level in self.bias_susceptibility.items()
                if level > 0.5
            ]
        }
