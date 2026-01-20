"""
社交系统

模拟孩子的社交环境和同龄人影响：
1. 同学关系
2. 邻居小朋友
3. 培训班同学
4. 同龄人比较效应
"""

import random
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """关系类型"""
    CLASSMATE = "同学"
    NEIGHBOR = "邻居"
    TRAINING_BUDDY = "培训班同学"
    RELATIVE = "亲戚家孩子"
    ONLINE_FRIEND = "网友"


class SocialStatus(str, Enum):
    """社交状态"""
    POPULAR = "受欢迎"
    NORMAL = "普通"
    SHY = "害羞内向"
    ISOLATED = "被孤立"
    LEADER = "小团体领袖"


@dataclass
class Peer:
    """同龄人"""
    name: str
    age: float
    relationship_type: RelationshipType
    academic_level: float  # 0-100 学习成绩
    social_skill: float    # 0-100 社交能力
    family_background: str  # 家庭背景描述
    
    # 与主角的关系
    friendship_level: float = 50.0  # 0-100 友谊程度
    influence_strength: float = 0.5  # 0-1 对主角的影响力
    
    # 行为特征
    is_positive_influence: bool = True
    
    def __post_init__(self):
        # 根据学习成绩判断是正面还是负面影响
        if self.academic_level > 70:
            self.is_positive_influence = True
        elif self.academic_level < 40:
            self.is_positive_influence = False
        else:
            self.is_positive_influence = random.random() < 0.6


@dataclass
class SocialEvent:
    """社交事件"""
    event_type: str
    description: str
    participants: List[str]
    impact: Dict[str, float]
    week: int


class SocialSystem:
    """
    社交系统
    
    管理孩子的社交圈和同龄人影响
    """
    
    # 常见的同龄人名字
    PEER_NAMES = [
        "小明", "小红", "小华", "小丽", "小强", "小芳", "小伟", "小婷",
        "天天", "乐乐", "果果", "朵朵", "豆豆", "贝贝", "欢欢", "晨晨",
        "浩浩", "涵涵", "萌萌", "彤彤", "琳琳", "佳佳", "悦悦", "思思"
    ]
    
    # 家庭背景模板
    FAMILY_BACKGROUNDS = [
        "公务员家庭，条件优越",
        "普通工薪家庭",
        "个体户家庭，经济宽裕",
        "教师家庭，重视教育",
        "农村进城务工家庭",
        "单亲家庭",
        "知识分子家庭",
        "企业高管家庭"
    ]
    
    def __init__(self):
        self.peers: Dict[str, Peer] = {}
        self.social_status = SocialStatus.NORMAL
        self.social_events: List[SocialEvent] = []
        self.best_friend: Optional[str] = None
        self.bullying_status = "无"  # 无/轻微/严重
        
        # 初始化一些默认的同龄人
        self._initialize_default_peers()
    
    def _initialize_default_peers(self):
        """初始化默认的同龄人"""
        # 添加几个邻居小朋友
        for i in range(random.randint(2, 4)):
            self.add_peer(
                relationship_type=RelationshipType.NEIGHBOR,
                age_offset=random.uniform(-1, 2)
            )
    
    def add_peer(self, 
                  relationship_type: RelationshipType,
                  age_offset: float = 0,
                  name: str = None,
                  academic_level: float = None) -> Peer:
        """
        添加同龄人
        
        参数:
            relationship_type: 关系类型
            age_offset: 年龄偏移（相对于主角）
            name: 名字（可选）
            academic_level: 学习水平（可选）
            
        返回:
            创建的同龄人对象
        """
        # 生成名字
        if name is None:
            available_names = [n for n in self.PEER_NAMES if n not in self.peers]
            if not available_names:
                available_names = self.PEER_NAMES
            name = random.choice(available_names)
        
        # 生成属性
        if academic_level is None:
            # 学习水平正态分布
            academic_level = max(20, min(100, random.gauss(60, 20)))
        
        peer = Peer(
            name=name,
            age=0 + age_offset,  # 会在运行时更新
            relationship_type=relationship_type,
            academic_level=academic_level,
            social_skill=random.uniform(30, 80),
            family_background=random.choice(self.FAMILY_BACKGROUNDS),
            friendship_level=random.uniform(30, 70),
            influence_strength=random.uniform(0.2, 0.6)
        )
        
        self.peers[name] = peer
        logger.debug(f"添加同龄人: {name} ({relationship_type.value})")
        
        return peer
    
    def update_social_circle(self, child_age: float, child_state: Any):
        """
        更新社交圈（随年龄变化）
        
        参数:
            child_age: 孩子年龄
            child_state: 孩子状态
        """
        # 更新所有同龄人的年龄
        for peer in self.peers.values():
            # 简化：假设所有同龄人年龄类似
            peer.age = child_age + random.uniform(-1, 1)
        
        # 入学时添加同学
        if 6 <= child_age < 6.5 and not any(
            p.relationship_type == RelationshipType.CLASSMATE 
            for p in self.peers.values()
        ):
            # 添加同学
            for _ in range(random.randint(3, 6)):
                self.add_peer(RelationshipType.CLASSMATE)
            logger.info("入学了，添加了新同学")
        
        # 更新社交状态
        self._update_social_status(child_state)
    
    def _update_social_status(self, child_state: Any):
        """根据孩子状态更新社交状态"""
        social_skill = child_state.social_skill
        extroversion = child_state.personality.extroversion
        
        if social_skill > 70 and extroversion > 0.6:
            self.social_status = SocialStatus.POPULAR
        elif social_skill < 40 or extroversion < 0.3:
            self.social_status = SocialStatus.SHY
        elif social_skill > 60 and extroversion > 0.5:
            self.social_status = random.choice([SocialStatus.NORMAL, SocialStatus.LEADER])
        else:
            self.social_status = SocialStatus.NORMAL
    
    def calculate_peer_influence(self, child_state: Any) -> Dict[str, float]:
        """
        计算同龄人对孩子的影响
        
        参数:
            child_state: 孩子状态
            
        返回:
            各项属性的变化
        """
        effects = {
            "knowledge": 0.0,
            "stress": 0.0,
            "social_skill": 0.0,
            "self_confidence": 0.0
        }
        
        if not self.peers:
            return effects
        
        # 计算好友的平均影响
        total_influence = 0.0
        for peer in self.peers.values():
            if peer.friendship_level > 60:  # 只有关系好的才有影响
                influence = peer.influence_strength * (peer.friendship_level / 100)
                total_influence += influence
                
                # 学习影响
                if peer.academic_level > child_state.knowledge:
                    # 好的同学带动学习
                    effects["knowledge"] += influence * 0.3
                elif peer.academic_level < child_state.knowledge - 20:
                    # 差的同学可能有负面影响
                    effects["knowledge"] -= influence * 0.1
                
                # 社交能力影响
                effects["social_skill"] += influence * 0.2
        
        # 如果有好朋友，减少压力
        if self.best_friend and self.best_friend in self.peers:
            effects["stress"] -= 1.0
            effects["self_confidence"] += 0.5
        
        # 被孤立会增加压力
        if self.social_status == SocialStatus.ISOLATED:
            effects["stress"] += 2.0
            effects["self_confidence"] -= 1.0
        
        return effects
    
    def generate_social_event(self, 
                               child_age: float,
                               child_state: Any,
                               week: int) -> Optional[SocialEvent]:
        """
        随机生成社交事件
        
        参数:
            child_age: 孩子年龄
            child_state: 孩子状态
            week: 当前周数
            
        返回:
            社交事件或None
        """
        if child_age < 3 or random.random() > 0.15:  # 15%概率触发
            return None
        
        event_templates = [
            {
                "type": "好友聚会",
                "description": "和小伙伴们一起玩耍",
                "impact": {"stress": -3, "social_skill": 1},
                "min_age": 3
            },
            {
                "type": "学校比赛",
                "description": "参加学校的比赛活动",
                "impact": {"knowledge": 2, "stress": 3, "self_confidence": 2},
                "min_age": 6
            },
            {
                "type": "朋友矛盾",
                "description": "和朋友发生了小矛盾",
                "impact": {"stress": 5, "social_skill": -1},
                "min_age": 4
            },
            {
                "type": "生日派对",
                "description": "参加同学的生日派对",
                "impact": {"stress": -2, "social_skill": 2},
                "min_age": 4
            },
            {
                "type": "交到新朋友",
                "description": "认识了新朋友",
                "impact": {"stress": -1, "social_skill": 3, "self_confidence": 1},
                "min_age": 3
            }
        ]
        
        # 筛选适合年龄的事件
        suitable_events = [
            e for e in event_templates 
            if child_age >= e["min_age"]
        ]
        
        if not suitable_events:
            return None
        
        template = random.choice(suitable_events)
        
        # 选择参与者
        participants = []
        if self.peers:
            participants = random.sample(
                list(self.peers.keys()),
                min(len(self.peers), random.randint(1, 3))
            )
        
        event = SocialEvent(
            event_type=template["type"],
            description=template["description"],
            participants=participants,
            impact=template["impact"],
            week=week
        )
        
        self.social_events.append(event)
        logger.info(f"社交事件: {event.event_type} - {event.description}")
        
        return event
    
    def apply_social_event(self, 
                            event: SocialEvent,
                            child_state: Any) -> Dict[str, float]:
        """
        应用社交事件的影响
        
        参数:
            event: 社交事件
            child_state: 孩子状态
            
        返回:
            状态变化
        """
        changes = {}
        
        for attr, change in event.impact.items():
            if hasattr(child_state, attr):
                current = getattr(child_state, attr)
                new_value = max(0.0, min(100.0, current + change))
                setattr(child_state, attr, new_value)
                changes[attr] = change
        
        return changes
    
    def update_friendship(self, 
                           peer_name: str,
                           change: float,
                           reason: str = ""):
        """
        更新与某个同龄人的友谊
        
        参数:
            peer_name: 同龄人名字
            change: 变化量
            reason: 原因
        """
        if peer_name not in self.peers:
            return
        
        peer = self.peers[peer_name]
        peer.friendship_level = max(0, min(100, peer.friendship_level + change))
        
        # 检查是否成为最好的朋友
        if peer.friendship_level > 80 and self.best_friend is None:
            self.best_friend = peer_name
            logger.info(f"交到了最好的朋友: {peer_name}")
        
        # 如果友谊降到很低，可能断交
        if peer.friendship_level < 10 and peer_name == self.best_friend:
            self.best_friend = None
            logger.info(f"和{peer_name}的友谊破裂了")
    
    def get_comparison_pressure(self, child_state: Any) -> float:
        """
        计算来自同龄人的比较压力
        
        参数:
            child_state: 孩子状态
            
        返回:
            压力值 (0-20)
        """
        pressure = 0.0
        
        for peer in self.peers.values():
            if peer.academic_level > child_state.knowledge + 15:
                # 比自己优秀很多的同龄人带来压力
                pressure += peer.influence_strength * 3
        
        return min(20, pressure)
    
    def get_social_summary(self) -> Dict[str, Any]:
        """获取社交状态摘要"""
        return {
            "社交圈人数": len(self.peers),
            "社交状态": self.social_status.value,
            "最好的朋友": self.best_friend or "暂无",
            "同学数": len([p for p in self.peers.values() 
                         if p.relationship_type == RelationshipType.CLASSMATE]),
            "邻居小朋友数": len([p for p in self.peers.values() 
                              if p.relationship_type == RelationshipType.NEIGHBOR]),
            "社交事件数": len(self.social_events)
        }


# 全局社交系统实例
_social_system: Optional[SocialSystem] = None


def get_social_system() -> SocialSystem:
    """获取社交系统实例"""
    global _social_system
    if _social_system is None:
        _social_system = SocialSystem()
    return _social_system


def reset_social_system():
    """重置社交系统"""
    global _social_system
    _social_system = None
