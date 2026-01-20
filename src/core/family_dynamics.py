"""
家庭动态系统

实现家庭成员之间的互动和冲突：
1. 代际冲突（父母 vs 祖父母的教育理念分歧）
2. 家庭决策博弈
3. 溺爱与严格的平衡
4. 家庭会议机制
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EducationStyle(str, Enum):
    """教育风格"""
    STRICT = "严格型"           # 重视纪律和成绩
    NURTURING = "关爱型"        # 重视情感和陪伴
    PERMISSIVE = "放任型"       # 较少干预
    AUTHORITATIVE = "权威型"    # 严格但有道理
    SPOILING = "溺爱型"         # 过度保护和满足


class ConflictType(str, Enum):
    """冲突类型"""
    EDUCATION_STYLE = "教育理念冲突"
    SPENDING = "教育支出分歧"
    DISCIPLINE = "管教方式分歧"
    EXPECTATIONS = "期望值差异"
    SPOILING = "溺爱问题"
    RESPONSIBILITIES = "责任分配"


@dataclass
class FamilyConflict:
    """家庭冲突记录"""
    conflict_type: ConflictType
    parties: Tuple[str, str]  # 冲突双方
    description: str
    severity: float  # 0-1，冲突严重程度
    resolved: bool = False
    resolution: str = ""
    week: int = 0


@dataclass 
class FamilyDynamicsState:
    """家庭动态状态"""
    # 成员间关系
    father_mother_harmony: float = 80.0  # 父母和谐度
    parent_grandparent_harmony: float = 70.0  # 两代人和谐度
    grandparents_harmony: float = 90.0  # 祖父母间和谐度
    
    # 教育理念统一度
    education_consensus: float = 60.0
    
    # 祖辈溺爱程度
    grandparent_spoiling: float = 50.0
    
    # 家庭权力结构
    father_authority: float = 0.3
    mother_authority: float = 0.3
    grandfather_authority: float = 0.2
    grandmother_authority: float = 0.2
    
    # 冲突历史
    conflict_history: List[FamilyConflict] = field(default_factory=list)
    
    # 最近的冲突
    recent_conflict: Optional[FamilyConflict] = None


class FamilyDynamicsSystem:
    """
    家庭动态系统
    
    管理家庭成员之间的互动、冲突和决策
    """
    
    # 教育风格对应的行为偏好
    STYLE_ACTION_PREFERENCES = {
        EducationStyle.STRICT: ["严格要求", "监督学习", "辅导"],
        EducationStyle.NURTURING: ["陪伴", "鼓励", "沟通", "游戏互动"],
        EducationStyle.PERMISSIVE: ["游戏互动", "陪伴", "健康教育"],
        EducationStyle.AUTHORITATIVE: ["辅导", "沟通", "鼓励", "个性化计划"],
        EducationStyle.SPOILING: ["陪伴", "游戏互动"]  # 并且容易增加不必要的花费
    }
    
    def __init__(self):
        self.state = FamilyDynamicsState()
        self.member_styles: Dict[str, EducationStyle] = {}
        self._initialize_member_styles()
    
    def _initialize_member_styles(self):
        """初始化成员教育风格"""
        # 父母倾向于严格或权威
        self.member_styles["father"] = random.choice([
            EducationStyle.STRICT,
            EducationStyle.AUTHORITATIVE,
            EducationStyle.NURTURING
        ])
        self.member_styles["mother"] = random.choice([
            EducationStyle.NURTURING,
            EducationStyle.AUTHORITATIVE,
            EducationStyle.STRICT
        ])
        # 祖父母更可能溺爱
        self.member_styles["grandfather"] = random.choice([
            EducationStyle.PERMISSIVE,
            EducationStyle.SPOILING,
            EducationStyle.STRICT  # 也可能老派严格
        ])
        self.member_styles["grandmother"] = random.choice([
            EducationStyle.SPOILING,
            EducationStyle.NURTURING,
            EducationStyle.PERMISSIVE
        ])
    
    def check_for_conflicts(self, 
                            proposed_actions: Dict[str, Dict[str, Any]],
                            child_state: Any,
                            family_state: Any,
                            week: int) -> Optional[FamilyConflict]:
        """
        检查是否会产生家庭冲突
        
        参数:
            proposed_actions: 各成员提议的行动 {member: action_dict}
            child_state: 孩子状态
            family_state: 家庭状态
            week: 当前周数
            
        返回:
            冲突对象，如果没有冲突返回None
        """
        conflict = None
        
        # 1. 检查教育理念冲突
        conflict = self._check_education_style_conflict(proposed_actions, week)
        if conflict:
            return conflict
        
        # 2. 检查支出冲突
        conflict = self._check_spending_conflict(proposed_actions, family_state, week)
        if conflict:
            return conflict
        
        # 3. 检查溺爱冲突
        conflict = self._check_spoiling_conflict(proposed_actions, child_state, week)
        if conflict:
            return conflict
        
        return None
    
    def _check_education_style_conflict(self, 
                                         proposed_actions: Dict[str, Dict[str, Any]],
                                         week: int) -> Optional[FamilyConflict]:
        """检查教育理念冲突"""
        # 检查父母和祖父母的提议是否冲突
        parent_actions = [
            proposed_actions.get("father", {}).get("action_type"),
            proposed_actions.get("mother", {}).get("action_type")
        ]
        grandparent_actions = [
            proposed_actions.get("grandfather", {}).get("action_type"),
            proposed_actions.get("grandmother", {}).get("action_type")
        ]
        
        # 严格 vs 溺爱 冲突
        strict_actions = {"严格要求", "监督学习"}
        lenient_actions = {"游戏互动", "陪伴"}
        
        parent_strict = any(a in strict_actions for a in parent_actions if a)
        grandparent_lenient = any(a in lenient_actions for a in grandparent_actions if a)
        
        if parent_strict and grandparent_lenient:
            # 有冲突
            if random.random() < 0.3:  # 30%概率触发明显冲突
                return FamilyConflict(
                    conflict_type=ConflictType.EDUCATION_STYLE,
                    parties=("parents", "grandparents"),
                    description="父母想要严格要求孩子学习，但祖父母认为孩子应该开心玩耍",
                    severity=0.5,
                    week=week
                )
        
        return None
    
    def _check_spending_conflict(self,
                                  proposed_actions: Dict[str, Dict[str, Any]],
                                  family_state: Any,
                                  week: int) -> Optional[FamilyConflict]:
        """检查支出冲突"""
        total_cost = sum(
            action.get("cost", 0) 
            for action in proposed_actions.values()
            if action
        )
        
        # 如果总花费超过家庭月收入的20%
        monthly_income = (
            family_state.father.salary +
            family_state.mother.salary
        )
        
        if total_cost > monthly_income * 0.2:
            if random.random() < 0.4:
                return FamilyConflict(
                    conflict_type=ConflictType.SPENDING,
                    parties=("father", "mother"),
                    description="关于孩子教育支出的分歧，一方认为应该多投入，另一方担心经济压力",
                    severity=0.4,
                    week=week
                )
        
        return None
    
    def _check_spoiling_conflict(self,
                                  proposed_actions: Dict[str, Dict[str, Any]],
                                  child_state: Any,
                                  week: int) -> Optional[FamilyConflict]:
        """检查溺爱冲突"""
        grandparent_styles = [
            self.member_styles.get("grandfather"),
            self.member_styles.get("grandmother")
        ]
        
        # 如果祖父母是溺爱型，且孩子压力已经很低（不需要更多放松）
        if EducationStyle.SPOILING in grandparent_styles:
            if child_state.stress < 30 and child_state.knowledge < 40:
                if random.random() < 0.25:
                    return FamilyConflict(
                        conflict_type=ConflictType.SPOILING,
                        parties=("parents", "grandparents"),
                        description="父母担心祖父母太溺爱孩子，影响学习积极性",
                        severity=0.4,
                        week=week
                    )
        
        return None
    
    def resolve_conflict(self, conflict: FamilyConflict) -> Dict[str, Any]:
        """
        解决冲突
        
        参数:
            conflict: 冲突对象
            
        返回:
            解决结果
        """
        resolution_strategies = [
            "compromise",  # 妥协
            "authority",   # 权威决定
            "postpone",    # 搁置
            "discuss"      # 讨论解决
        ]
        
        strategy = random.choice(resolution_strategies)
        result = {
            "strategy": strategy,
            "conflict_type": conflict.conflict_type.value,
            "parties": conflict.parties
        }
        
        if strategy == "compromise":
            conflict.resolved = True
            conflict.resolution = "双方各退一步，达成妥协"
            result["outcome"] = "妥协解决"
            result["harmony_change"] = -5  # 轻微影响和谐度
            
        elif strategy == "authority":
            # 根据权力结构决定
            if "parents" in conflict.parties:
                conflict.resolved = True
                conflict.resolution = "父母决定采用自己的教育方式"
                result["outcome"] = "父母做主"
                result["harmony_change"] = -10  # 较大影响和谐度
            else:
                conflict.resolution = "最终由家庭权威决定"
                result["outcome"] = "权威决定"
                result["harmony_change"] = -8
                
        elif strategy == "postpone":
            conflict.resolved = False
            conflict.resolution = "暂时搁置，以后再议"
            result["outcome"] = "搁置争议"
            result["harmony_change"] = -3
            
        else:  # discuss
            conflict.resolved = random.random() < 0.7  # 70%概率成功
            if conflict.resolved:
                conflict.resolution = "通过家庭会议讨论解决"
                result["outcome"] = "讨论解决"
                result["harmony_change"] = 0  # 可能还增进理解
            else:
                conflict.resolution = "讨论未果，各持己见"
                result["outcome"] = "未能解决"
                result["harmony_change"] = -5
        
        # 记录冲突
        self.state.conflict_history.append(conflict)
        self.state.recent_conflict = conflict
        
        # 更新和谐度
        if "parents" in conflict.parties and "grandparents" in conflict.parties:
            self.state.parent_grandparent_harmony = max(
                0, 
                self.state.parent_grandparent_harmony + result["harmony_change"]
            )
        
        return result
    
    def simulate_family_meeting(self,
                                 child_state: Any,
                                 family_state: Any,
                                 topic: str) -> Dict[str, Any]:
        """
        模拟家庭会议
        
        参数:
            child_state: 孩子状态
            family_state: 家庭状态
            topic: 讨论话题
            
        返回:
            会议结果
        """
        meeting = {
            "topic": topic,
            "participants": ["father", "mother", "grandfather", "grandmother"],
            "opinions": {},
            "outcome": None
        }
        
        # 收集各方意见
        opinions = {
            "father": self._generate_opinion("father", topic, child_state),
            "mother": self._generate_opinion("mother", topic, child_state),
            "grandfather": self._generate_opinion("grandfather", topic, child_state),
            "grandmother": self._generate_opinion("grandmother", topic, child_state)
        }
        meeting["opinions"] = opinions
        
        # 根据权力结构和意见相似度决定结果
        # 简化版：父母意见优先，但需要一定程度的共识
        parent_agree = opinions["father"]["stance"] == opinions["mother"]["stance"]
        grandparent_agree = opinions["grandfather"]["stance"] == opinions["grandmother"]["stance"]
        
        if parent_agree:
            meeting["outcome"] = opinions["father"]["stance"]
            meeting["consensus"] = "父母达成一致"
        elif grandparent_agree and not parent_agree:
            # 如果父母不一致但祖父母一致，可能祖父母意见有一定影响
            if random.random() < 0.3:
                meeting["outcome"] = opinions["grandfather"]["stance"]
                meeting["consensus"] = "祖父母意见占优"
            else:
                meeting["outcome"] = "继续讨论"
                meeting["consensus"] = "未达成共识"
        else:
            # 默认采用最保守的方案
            meeting["outcome"] = "保持现状"
            meeting["consensus"] = "各执己见，维持现状"
        
        return meeting
    
    def _generate_opinion(self, member: str, topic: str, child_state: Any) -> Dict[str, Any]:
        """生成成员对话题的意见"""
        style = self.member_styles.get(member, EducationStyle.NURTURING)
        
        opinion = {
            "member": member,
            "style": style.value,
            "stance": None,
            "reasoning": ""
        }
        
        # 根据话题和风格生成立场
        if topic == "增加培训班":
            if style in [EducationStyle.STRICT, EducationStyle.AUTHORITATIVE]:
                opinion["stance"] = "支持"
                opinion["reasoning"] = "学习要趁早，多学点总是好的"
            elif style == EducationStyle.SPOILING:
                opinion["stance"] = "反对"
                opinion["reasoning"] = "孩子太累了，应该让她玩"
            else:
                opinion["stance"] = "中立"
                opinion["reasoning"] = "看孩子自己的意愿"
                
        elif topic == "学习时间安排":
            if style == EducationStyle.STRICT:
                opinion["stance"] = "增加"
                opinion["reasoning"] = "现在不努力，以后怎么办"
            elif style in [EducationStyle.SPOILING, EducationStyle.PERMISSIVE]:
                opinion["stance"] = "减少"
                opinion["reasoning"] = "孩子太辛苦了"
            else:
                opinion["stance"] = "保持"
                opinion["reasoning"] = "目前挺好的"
        
        else:
            opinion["stance"] = "中立"
            opinion["reasoning"] = "听其他人的意见"
        
        return opinion
    
    def update_dynamics(self, 
                        action_result: Dict[str, Any],
                        child_state: Any,
                        family_state: Any) -> Dict[str, float]:
        """
        根据行动结果更新家庭动态
        
        参数:
            action_result: 行动结果
            child_state: 孩子状态
            family_state: 家庭状态
            
        返回:
            动态变化
        """
        changes = {}
        
        # 成功的行动增加共识度
        if action_result.get("success", False):
            self.state.education_consensus = min(
                100, self.state.education_consensus + 1.0
            )
            changes["education_consensus"] = 1.0
        else:
            self.state.education_consensus = max(
                0, self.state.education_consensus - 2.0
            )
            changes["education_consensus"] = -2.0
        
        # 如果孩子状态改善，家庭和谐度提升
        if child_state.stress < 50 and child_state.knowledge > 50:
            self.state.father_mother_harmony = min(
                100, self.state.father_mother_harmony + 0.5
            )
            self.state.parent_grandparent_harmony = min(
                100, self.state.parent_grandparent_harmony + 0.3
            )
        
        return changes
    
    def get_dynamics_summary(self) -> Dict[str, Any]:
        """获取家庭动态摘要"""
        return {
            "父母和谐度": f"{self.state.father_mother_harmony:.1f}",
            "两代人和谐度": f"{self.state.parent_grandparent_harmony:.1f}",
            "教育共识度": f"{self.state.education_consensus:.1f}",
            "祖辈溺爱度": f"{self.state.grandparent_spoiling:.1f}",
            "近期冲突数": len([c for c in self.state.conflict_history[-12:] if not c.resolved]),
            "成员教育风格": {k: v.value for k, v in self.member_styles.items()}
        }


# 全局家庭动态系统实例
_family_dynamics_system: Optional[FamilyDynamicsSystem] = None


def get_family_dynamics_system() -> FamilyDynamicsSystem:
    """获取家庭动态系统实例"""
    global _family_dynamics_system
    if _family_dynamics_system is None:
        _family_dynamics_system = FamilyDynamicsSystem()
    return _family_dynamics_system


def reset_family_dynamics_system():
    """重置家庭动态系统"""
    global _family_dynamics_system
    _family_dynamics_system = None
