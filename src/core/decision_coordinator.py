import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

from src.core.state import ChildState, FamilyState
from src.core.config import SimulationConfig
from src.agents.factory import AgentFactory
from src.core.dialogue_system import DialogueSystemFactory

logger = logging.getLogger(__name__)


class DecisionPriority(Enum):
    """决策优先级"""
    HIGH = 3
    MEDIUM = 2
    LOW = 1


class ConflictResolutionStrategy(Enum):
    """冲突解决策略"""
    PRIORITY_BASED = "priority_based"  # 基于优先级
    VOTING = "voting"  # 投票
    COMPROMISE = "compromise"  # 妥协
    NEGOTIATION = "negotiation"  # 协商


@dataclass
class DecisionProposal:
    """决策提议"""
    member: str
    action_type: str
    dialogue: str
    cost: float
    priority: DecisionPriority
    reasoning: str
    confidence: float  # 0-1之间的置信度


@dataclass
class ConflictResolution:
    """冲突解决结果"""
    strategy: ConflictResolutionStrategy
    final_decision: DecisionProposal
    resolution_reason: str
    involved_members: List[str]


class DecisionCoordinator:
    """
    决策协调器，负责多成员决策协调
    
    支持真正的 LLM 自主决策（如果 API 已配置）或规则 Mock 决策（降级模式）
    """

    def __init__(self, model_name: str = "deepseek", config: SimulationConfig = None):
        self.agent_factory = AgentFactory()
        self.members = ["father", "mother", "grandfather", "grandmother"]
        self.model_name = model_name.lower()  # 存储模型名称（转换为小写确保一致性）
        self.config = config  # 保存配置，用于创建 LLM Agent
        self._agent_cache: Dict[str, Any] = {}  # 成员级缓存，持久化记忆
        
        # 设置 AgentFactory 的全局配置
        if config:
            AgentFactory.set_config(config)
        
        logger.info(f"DecisionCoordinator初始化，使用模型: {self.model_name}")

        # 成员决策优先级配置
        self.member_priorities = {
            "father": DecisionPriority.HIGH,
            "mother": DecisionPriority.HIGH,
            "grandfather": DecisionPriority.MEDIUM,
            "grandmother": DecisionPriority.LOW
        }

        # 成员影响力权重（用于冲突解决）
        self.member_weights = {
            "father": 1.0,
            "mother": 1.0,
            "grandfather": 0.8,
            "grandmother": 0.6
        }

    async def coordinate_decision(
        self,
        child_state: ChildState,
        family_state: FamilyState,
        event: str
    ) -> Dict[str, Any]:
        """
        协调多成员决策过程

        参数:
            child_state: 孩子当前状态
            family_state: 家庭当前状态
            event: 当前事件描述

        返回:
            最终决策结果
        """
        logger.info("开始多成员决策协调")

        # 1. 收集所有成员的决策提议
        proposals = await self._collect_proposals(child_state, family_state, event)

        if len(proposals) == 0:
            logger.warning("没有收到任何决策提议")
            return self._create_default_decision()

        if len(proposals) == 1:
            # 只有一个提议，直接采用
            proposal = proposals[0]
            decision = self._create_decision_result(proposal, f"只有{self._get_member_name(proposal.member)}的提议")
            return self._humanize_decision(decision, child_state, family_state)

        # 2. 检查是否有冲突（传入状态信息）
        conflict_result = self._detect_conflicts(
            proposals,
            child_state=child_state,
            family_state=family_state
        )

        if not conflict_result["has_conflict"]:
            # 没有冲突，选择最高优先级的提议
            best_proposal = self._select_best_proposal(proposals)
            decision = self._create_decision_result(best_proposal, "无冲突，采用最优提议")
            return self._humanize_decision(decision, child_state, family_state)

        # 3. 解决冲突（传入状态信息用于协商）
        resolution = await self._resolve_conflicts(
            proposals, 
            conflict_result,
            child_state=child_state,
            family_state=family_state
        )

        logger.info(f"决策协调完成，使用{resolution.strategy.value}策略")
        decision = self._create_decision_result(resolution.final_decision, resolution.resolution_reason)
        return self._humanize_decision(decision, child_state, family_state)

    async def _collect_proposals(
        self,
        child_state: ChildState,
        family_state: FamilyState,
        event: str
    ) -> List[DecisionProposal]:
        """
        收集所有成员的决策提议

        返回:
            决策提议列表
        """
        proposals = []

        for member in self.members:
            try:
                # 创建对应成员的agent，使用配置的模型名称
                # 验证模型名称有效性
                if not self.model_name:
                    logger.error(f"模型名称未设置，使用默认模型deepseek")
                    model_name = "deepseek"
                else:
                    model_name = self.model_name
                agent = self._get_or_create_agent(model_name, member)
                logger.debug(f"为{self._get_member_name(member)}获取{model_name}模型的Agent")

                # 获取决策
                decision = await agent.decide(child_state, family_state, event)

                # 创建提议对象
                proposal = DecisionProposal(
                    member=member,
                    action_type=decision["action_type"],
                    dialogue=decision["dialogue"],
                    cost=decision.get("cost", 0),
                    priority=self.member_priorities[member],
                    reasoning=self._generate_reasoning(member, decision, child_state, family_state),
                    confidence=self._calculate_confidence(member, decision, child_state, family_state)
                )

                proposals.append(proposal)
                member_cn = self._get_member_name(member)
                logger.info(f"收集到{member_cn}的决策: {decision['action_type']}")

            except Exception as e:
                member_cn = self._get_member_name(member)
                logger.warning(f"获取{member_cn}决策失败: {e}")
                continue

        return proposals

    def _get_or_create_agent(self, model_name: str, member: str):
        """获取或创建成员Agent（持久化记忆与情绪）"""
        cache_key = f"{model_name}:{member}"
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]
        
        agent = self.agent_factory.create_agent(model_name, member=member, config=self.config)
        self._agent_cache[cache_key] = agent
        return agent

    def _detect_conflicts(
        self, 
        proposals: List[DecisionProposal],
        child_state: ChildState = None,
        family_state: FamilyState = None
    ) -> Dict[str, Any]:
        """
        检测决策冲突（增强版：考虑状态影响）

        返回:
            冲突检测结果
        """
        if len(proposals) <= 1:
            return {"has_conflict": False}

        # 检查是否有不同的行动类型
        action_types = set(p.action_type for p in proposals)
        cost_ranges = [p.cost for p in proposals]
        
        # 计算状态影响差异（如果提供了状态信息）
        state_impact_conflict = False
        impact_scores = []
        
        if child_state and family_state:
            # 从dungeon_master导入ACTION_EFFECTS来评估影响
            from src.engine.dungeon_master import ACTION_EFFECTS
            
            for proposal in proposals:
                effects = ACTION_EFFECTS.get(proposal.action_type, {})
                # 计算综合影响分数
                impact_score = 0.0
                
                # 知识影响（正权重）
                if "knowledge" in effects:
                    impact_score += effects["knowledge"] * 1.0
                
                # 压力影响（负权重，因为压力增加是负面的）
                if "stress" in effects:
                    impact_score -= abs(effects["stress"]) * 0.8
                
                # 关系影响（正权重）
                relationship_keys = ["father_relationship", "mother_relationship", 
                                   "grandfather_relationship", "grandmother_relationship"]
                for key in relationship_keys:
                    if key in effects:
                        impact_score += effects[key] * 0.5
                
                # 健康影响（正权重）
                if "physical_health" in effects:
                    impact_score += effects["physical_health"] * 0.7
                
                # 根据当前状态调整影响分数
                # 如果压力已经很高，增加压力的行为影响分数会更低
                if child_state.stress > 70 and effects.get("stress", 0) > 0:
                    impact_score -= effects["stress"] * 0.5
                
                # 如果知识储备低，增加知识的行为影响分数会更高
                if child_state.knowledge < 50 and effects.get("knowledge", 0) > 0:
                    impact_score += effects["knowledge"] * 0.3
                
                impact_scores.append(impact_score)
            
            # 如果影响分数差异很大（>5），认为有状态影响冲突
            if len(impact_scores) > 1:
                impact_diff = max(impact_scores) - min(impact_scores)
                state_impact_conflict = impact_diff > 5.0

        # 检查成本冲突
        cost_conflict = (max(cost_ranges) - min(cost_ranges) > 100)
        
        # 检查行动类型冲突
        action_type_conflict = len(action_types) > 1

        has_conflict = action_type_conflict or cost_conflict or state_impact_conflict

        if has_conflict:
            conflict_type = "action_type"
            if state_impact_conflict and not action_type_conflict:
                conflict_type = "state_impact"
            elif cost_conflict and not action_type_conflict:
                conflict_type = "cost"
            
            result = {
                "has_conflict": True,
                "conflict_type": conflict_type,
                "action_types": list(action_types),
                "cost_range": f"{min(cost_ranges)}-{max(cost_ranges)}"
            }
            
            if state_impact_conflict and impact_scores:
                result["impact_scores"] = impact_scores
                result["impact_diff"] = max(impact_scores) - min(impact_scores)
            
            return result

        return {"has_conflict": False}

    async def _resolve_conflicts(
        self,
        proposals: List[DecisionProposal],
        conflict_info: Dict[str, Any],
        child_state: ChildState = None,
        family_state: FamilyState = None
    ) -> ConflictResolution:
        """
        解决决策冲突

        返回:
            冲突解决结果
        """
        conflict_type = conflict_info.get("conflict_type", "unknown")

        # 根据冲突类型选择不同的解决策略
        if conflict_type == "cost":
            # 成本冲突：优先选择经济实惠的方案
            return await self._resolve_cost_conflict(proposals)
        elif conflict_type == "action_type":
            # 行动类型冲突：进行协商（传入状态信息）
            return await self._resolve_action_conflict(
                proposals,
                child_state=child_state,
                family_state=family_state
            )
        else:
            # 默认使用优先级策略
            return self._resolve_by_priority(proposals)

    async def _resolve_cost_conflict(self, proposals: List[DecisionProposal]) -> ConflictResolution:
        """解决成本冲突"""
        # 计算成本效益比（考虑家庭经济状况）
        best_proposal = min(proposals, key=lambda p: p.cost)

        return ConflictResolution(
            strategy=ConflictResolutionStrategy.COMPROMISE,
            final_decision=best_proposal,
            resolution_reason="成本冲突，通过选择最具经济效益的方案解决",
            involved_members=[p.member for p in proposals]
        )

    async def _resolve_action_conflict(
        self, 
        proposals: List[DecisionProposal],
        child_state: ChildState = None,
        family_state: FamilyState = None
    ) -> ConflictResolution:
        """解决行动类型冲突"""
        # 尝试协商机制（传入状态信息）
        negotiated_result = await self._negotiate_decisions(
            proposals, 
            child_state=child_state,
            family_state=family_state
        )

        if negotiated_result:
            return ConflictResolution(
                strategy=ConflictResolutionStrategy.NEGOTIATION,
                final_decision=negotiated_result,
                resolution_reason=negotiated_result.reasoning,
                involved_members=[p.member for p in proposals]
            )
        else:
            # 协商失败，使用投票机制
            return self._resolve_by_voting(proposals)

    async def _negotiate_decisions(
        self, 
        proposals: List[DecisionProposal],
        child_state: ChildState = None,
        family_state: FamilyState = None
    ) -> Optional[DecisionProposal]:
        """
        尝试通过协商达成一致（增强版）

        返回:
            协商结果，如果协商失败返回None
        """
        action_types = set(p.action_type for p in proposals)
        
        # 协商规则表：定义不同冲突的协商方案
        negotiation_rules = [
            # 规则1: 陪伴 vs 辅导 -> 简单辅导（折中方案）
            {
                "conflict": ({"陪伴", "辅导"}, None),
                "compromise": "简单辅导",
                "reason": "平衡陪伴和学习的折中方案"
            },
            # 规则2: 鼓励 vs 严格要求 -> 根据压力值选择
            {
                "conflict": ({"鼓励", "严格要求"}, lambda cs: cs.stress > 60),
                "compromise": "鼓励",
                "reason": "压力较高时优先选择鼓励"
            },
            # 规则3: 鼓励 vs 严格要求 -> 压力低时选择严格要求
            {
                "conflict": ({"鼓励", "严格要求"}, lambda cs: cs.stress <= 60),
                "compromise": "严格要求",
                "reason": "压力较低时可以选择严格要求"
            },
            # 规则4: 花钱培训 vs 辅导 -> 根据经济状况选择
            {
                "conflict": ({"花钱培训", "辅导"}, lambda fs: fs.family_savings < 20000),
                "compromise": "辅导",
                "reason": "经济紧张时选择免费辅导"
            },
            # 规则5: 花钱培训 vs 辅导 -> 经济宽裕时选择培训
            {
                "conflict": ({"花钱培训", "辅导"}, lambda fs: fs.family_savings >= 20000),
                "compromise": "花钱培训",
                "reason": "经济宽裕时选择更有效的培训"
            },
            # 规则6: 监督学习 vs 陪伴 -> 根据知识储备选择
            {
                "conflict": ({"监督学习", "陪伴"}, lambda cs: cs.knowledge < 50),
                "compromise": "监督学习",
                "reason": "知识储备不足时优先学习"
            },
            # 规则7: 监督学习 vs 陪伴 -> 知识充足时选择陪伴
            {
                "conflict": ({"监督学习", "陪伴"}, lambda cs: cs.knowledge >= 50),
                "compromise": "陪伴",
                "reason": "知识储备充足时优先陪伴"
            },
            # 规则8: 健康教育 vs 辅导 -> 根据健康值选择
            {
                "conflict": ({"健康教育", "辅导"}, lambda cs: cs.physical_health < 60),
                "compromise": "健康教育",
                "reason": "健康值较低时优先关注健康"
            },
            # 规则9: 创新活动 vs 实践活动 -> 根据年龄选择
            {
                "conflict": ({"创新活动", "实践活动"}, None),
                "compromise": "实践活动",
                "reason": "实践活动更适合大多数情况"
            },
        ]
        
        # 尝试应用协商规则
        for rule in negotiation_rules:
            conflict_actions, condition = rule["conflict"]
            
            # 检查是否匹配冲突类型
            if conflict_actions.issubset(action_types):
                # 检查条件（如果有）
                if condition is not None:
                    if child_state and not condition(child_state):
                        continue
                    if family_state and not condition(family_state):
                        continue
                
                # 找到匹配的协商方案
                compromise_action = rule["compromise"]
                compromise_proposal = self._create_compromise_proposal(
                    proposals, 
                    compromise_action,
                    rule["reason"]
                )
                if compromise_proposal:
                    logger.info(f"协商成功：{rule['reason']} -> {compromise_action}")
                    return compromise_proposal
        
        # 如果没有匹配的规则，尝试基于成本的平均方案
        if len(proposals) == 2:
            avg_cost = sum(p.cost for p in proposals) / len(proposals)
            # 选择成本接近平均值的方案
            best_match = min(proposals, key=lambda p: abs(p.cost - avg_cost))
            logger.info(f"协商：基于成本平均值选择 {best_match.action_type}")
            return best_match
        
        return None  # 协商失败

    def _create_compromise_proposal(
        self, 
        proposals: List[DecisionProposal], 
        action_type: str,
        reason: str = "家庭协商后的妥协方案"
    ) -> Optional[DecisionProposal]:
        """创建妥协提议"""
        # 寻找支持妥协方案的成员
        for proposal in proposals:
            if proposal.action_type == action_type:
                # 如果已有匹配的提议，使用它但更新reasoning
                return DecisionProposal(
                    member=proposal.member,
                    action_type=proposal.action_type,
                    dialogue=proposal.dialogue,
                    cost=proposal.cost,
                    priority=proposal.priority,
                    reasoning=reason,
                    confidence=proposal.confidence
                )

        # 如果没有现成的，基于现有提议创建新的
        base_proposal = proposals[0]  # 使用第一个作为基础
        
        # 计算平均成本，但不超过最高成本
        avg_cost = sum(p.cost for p in proposals) / len(proposals)
        max_cost = max(p.cost for p in proposals)
        final_cost = min(avg_cost, max_cost)

        return DecisionProposal(
            member="family_consensus",  # 家庭共识
            action_type=action_type,
            dialogue=f"经过家庭讨论，我们决定选择{action_type}。",
            cost=final_cost,
            priority=DecisionPriority.MEDIUM,
            reasoning=reason,
            confidence=0.8
        )

    def _resolve_by_voting(self, proposals: List[DecisionProposal]) -> ConflictResolution:
        """通过投票解决冲突"""
        # 基于成员权重进行投票
        # 使用列表索引而不是对象作为键，因为 DecisionProposal 不可哈希
        votes = []
        for i, proposal in enumerate(proposals):
            weight = self.member_weights.get(proposal.member, 0.5)
            votes.append((i, weight, proposal))

        # 选择权重最高的提议
        winner_idx, winner_weight, winner = max(votes, key=lambda x: x[1])

        return ConflictResolution(
            strategy=ConflictResolutionStrategy.VOTING,
            final_decision=winner,
            resolution_reason="通过家庭成员权重投票决定",
            involved_members=[p.member for p in proposals]
        )

    def _resolve_by_priority(self, proposals: List[DecisionProposal]) -> ConflictResolution:
        """基于优先级解决冲突"""
        # 选择最高优先级的提议
        highest_priority = max(p.priority.value for p in proposals)
        high_priority_proposals = [p for p in proposals if p.priority.value == highest_priority]

        # 如果有多个相同优先级的，选择第一个
        winner = high_priority_proposals[0]

        return ConflictResolution(
            strategy=ConflictResolutionStrategy.PRIORITY_BASED,
            final_decision=winner,
            resolution_reason="基于家庭成员优先级选择",
            involved_members=[p.member for p in proposals]
        )

    def _select_best_proposal(self, proposals: List[DecisionProposal]) -> DecisionProposal:
        """选择最优提议"""
        # 基于优先级和置信度选择
        scored_proposals = []
        for proposal in proposals:
            score = proposal.priority.value * proposal.confidence
            scored_proposals.append((score, proposal))

        return max(scored_proposals, key=lambda x: x[0])[1]

    def _generate_reasoning(self, member: str, decision: Dict[str, Any],
                          child_state: ChildState, family_state: FamilyState) -> str:
        """生成决策推理"""
        reasoning_parts = []

        # 基于孩子状态的推理
        if child_state.stress > 80:
            reasoning_parts.append("孩子压力较大，需要更多关爱")
        elif child_state.knowledge < 40:
            reasoning_parts.append("孩子知识储备不足，需要加强学习")

        # 基于家庭状态的推理
        savings = family_state.family_savings
        if savings < 10000:
            reasoning_parts.append("家庭经济紧张，应节约开支")

        # 成员特定的推理
        if member == "father":
            reasoning_parts.append("作为父亲，应该承担主要教育责任")
        elif member == "mother":
            reasoning_parts.append("关注孩子的全面发展和身心健康")

        return "；".join(reasoning_parts) if reasoning_parts else "基于当前情况做出的决策"

    def _calculate_confidence(self, member: str, decision: Dict[str, Any],
                            child_state: ChildState, family_state: FamilyState) -> float:
        """计算决策置信度"""
        base_confidence = 0.8

        # 根据孩子状态调整置信度
        if child_state.stress > 70 and decision["action_type"] in ["鼓励", "陪伴"]:
            base_confidence += 0.1
        elif child_state.knowledge < 50 and decision["action_type"] in ["辅导", "花钱培训"]:
            base_confidence += 0.1

        # 根据经济状况调整
        if family_state.family_savings < decision.get("cost", 0):
            base_confidence -= 0.2

        return max(0.1, min(1.0, base_confidence))

    def _get_member_name(self, member: str) -> str:
        """获取成员的中文名称"""
        names = {
            "father": "父亲",
            "mother": "母亲",
            "grandfather": "祖父",
            "grandmother": "祖母",
            "family": "家庭",
            "family_consensus": "家庭共识",
            "system": "系统"
        }
        return names.get(member, member)

    def _create_default_decision(self) -> Dict[str, Any]:
        """创建默认决策"""
        return {
            "action_type": "陪伴",
            "dialogue": "孩子，爸爸妈妈陪你好好聊聊今天的情况。",
            "cost": 0,
            "member": "家庭",
            "member_id": "family",
            "reasoning": "默认决策，由于无法获取成员意见",
            "confidence": 0.5,
            "coordination_reason": "默认决策"
        }

    def _create_decision_result(self, proposal: DecisionProposal, reason: str) -> Dict[str, Any]:
        """创建决策结果"""
        # 将成员名称转换为中文
        member_cn = self._get_member_name(proposal.member)
        return {
            "action_type": proposal.action_type,
            "dialogue": proposal.dialogue,
            "cost": proposal.cost,
            "member": member_cn,
            "member_id": proposal.member,  # 保留英文ID用于内部处理
            "reasoning": proposal.reasoning,
            "confidence": proposal.confidence,
            "coordination_reason": reason
        }

    def _humanize_decision(
        self,
        decision: Dict[str, Any],
        child_state: ChildState,
        family_state: FamilyState
    ) -> Dict[str, Any]:
        """对决策进行拟人化处理（对话风格与孩子回应）"""
        member_id = decision.get("member_id")
        if member_id not in ["father", "mother", "grandfather", "grandmother"]:
            return decision
        
        member_obj = getattr(family_state, member_id, None)
        if not member_obj:
            return decision
        
        # 获取对话系统（按成员+性格复用）
        dialogue_system = DialogueSystemFactory.get_dialogue_system(
            member=member_id,
            personality=member_obj.personality
        )
        
        # 构造对话上下文
        context = dialogue_system.get_dialogue_context()
        context["current_action"] = decision.get("action_type", "陪伴")
        
        # 统一处理情绪值
        child_emotion = child_state.emotional_state
        if hasattr(child_emotion, "value"):
            child_emotion = child_emotion.value
        
        relationship = getattr(child_state, f"{member_id}_relationship", 50.0)
        child_age = child_state.calculate_age(family_state.current_date)
        parent_emotion = getattr(member_obj, "emotional_state", "平静")
        
        generated_dialogue = dialogue_system.generate_dialogue(
            action_type=decision.get("action_type", "陪伴"),
            child_age=child_age,
            child_stress=child_state.stress,
            relationship=relationship,
            emotional_state=parent_emotion,
            context=context
        )
        
        existing_dialogue = (decision.get("dialogue") or "").strip()
        llm_generated = decision.get("llm_generated", False)
        
        # 非LLM或对话质量较低时，使用拟人化系统生成
        if (not llm_generated) or len(existing_dialogue) < 6:
            decision["dialogue"] = generated_dialogue
            decision["dialogue_generated_by"] = "personalized_dialogue_system"
        else:
            decision["dialogue_generated_by"] = "agent"
        
        # 生成孩子回应
        decision["child_response"] = dialogue_system.simulate_child_response(
            action_type=decision.get("action_type", "陪伴"),
            child_stress=child_state.stress,
            child_emotional_state=child_emotion,
            relationship=relationship
        )
        
        # 记录对话元信息
        decision["parent_emotional_state"] = parent_emotion
        decision["parent_personality"] = member_obj.personality
        
        return decision