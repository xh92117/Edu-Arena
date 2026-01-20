import json
import logging
from typing import Dict, Any, Type, List, Optional
from src.agents.base import FamilyAgent
from src.core.state import ChildState, FamilyState
from src.core.config import SimulationConfig
import random

logger = logging.getLogger(__name__)

# 导入 LLM Agent
try:
    from src.agents.llm_agents import (
        LLM_AGENT_CLASSES,
        LLMDecisionAgent
    )
    LLM_AGENTS_AVAILABLE = True
except ImportError:
    LLM_AGENTS_AVAILABLE = False
    logger.warning("LLM Agent 模块未找到，将只使用 Mock Agent")

# 导入自适应 Agent
try:
    from src.agents.adaptive_agent import AdaptiveParentAgent
    ADAPTIVE_AGENT_AVAILABLE = True
except ImportError:
    ADAPTIVE_AGENT_AVAILABLE = False
    logger.warning("自适应 Agent 模块未找到")

# 基于年龄的行为类型定义
AGE_APPROPRIATE_ACTIONS = {
    "infant": {  # 0-3岁：适合婴幼儿的基础活动
        "actions": ["陪伴", "启蒙教育", "健康教育", "游戏互动"],
        "cost_range": [50, 0, 0, 30]
    },
    "preschool": {  # 3-6岁：启蒙类活动，避免学术性培训
        "actions": ["陪伴", "启蒙教育", "游戏互动", "简单辅导", "鼓励", "简单兴趣培养"],
        "cost_range": [50, 0, 30, 0, 0, 100]
    },
    "primary": {  # 6岁以上：适龄的兴趣班、特长班
        "actions": ["辅导", "鼓励", "花钱培训", "陪伴", "严格要求", "监督学习", "健康教育", "创新活动", "个性化计划", "实践活动"] + ["启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养"],
        "cost_range": [0, 0, 200, 100, 0, 0, 0, 100, 0, 80] + [0, 30, 0, 100]
    }
}

# 行为类型映射到适合的年龄阶段
ACTION_AGE_MAPPING = {
    "陪伴": ["infant", "preschool", "primary"],
    "启蒙教育": ["infant", "preschool", "primary"],
    "健康教育": ["infant", "preschool", "primary"],
    "游戏互动": ["infant", "preschool", "primary"],
    "简单辅导": ["preschool", "primary"],
    "鼓励": ["preschool", "primary"],
    "简单兴趣培养": ["preschool", "primary"],
    "辅导": ["primary"],
    "花钱培训": ["primary"],
    "严格要求": ["primary"],
    "监督学习": ["primary"],
    "创新活动": ["primary"],
    "个性化计划": ["primary"],
    "实践活动": ["primary"]
}

# 家庭成员名称映射（英文到中文）
MEMBER_NAMES = {
    "father": "爸爸",
    "mother": "妈妈",
    "grandfather": "爷爷",
    "grandmother": "奶奶"
}

# 兴趣/敏感期 -> 行为偏好映射（用于拟人化决策）
INTEREST_ACTION_BIAS = {
    "阅读": ["早期阅读", "启蒙教育", "简单辅导", "辅导"],
    "音乐": ["简单兴趣培养", "启蒙教育"],
    "美术": ["简单兴趣培养", "创新活动"],
    "运动": ["户外活动", "健康教育", "游戏互动", "实践活动"],
    "科学": ["创新活动", "实践活动", "启蒙教育"],
    "游戏": ["游戏互动"],
    "社交": ["社交接触", "沟通", "游戏互动"],
    "自然": ["户外活动", "实践活动"]
}

SENSITIVITY_ACTION_BIAS = {
    "语言": ["早期阅读", "启蒙教育", "沟通"],
    "秩序": ["日常照料", "监督学习"],
    "感官": ["感官刺激", "游戏互动"],
    "动作": ["户外活动", "游戏互动", "实践活动"],
    "社交": ["社交接触", "沟通", "游戏互动"],
    "数学": ["启蒙教育", "简单辅导", "辅导"],
    "阅读": ["早期阅读", "启蒙教育"]
}


def get_member_name_cn(member: str) -> str:
    """获取家庭成员的中文名称"""
    return MEMBER_NAMES.get(member, "家长")

def _calculate_action_weights(actions: List[str], child_state: ChildState) -> List[float]:
    """根据兴趣和敏感期计算行为权重"""
    weights = [1.0 for _ in actions]
    
    top_interests = []
    if hasattr(child_state, "interests"):
        top_interests = child_state.interests.get_top_interests(3)
    
    active_sensitivities = {}
    if hasattr(child_state, "development_sensitivity"):
        active_sensitivities = child_state.development_sensitivity.get_active_sensitivities()
    
    for idx, action in enumerate(actions):
        # 兴趣加权
        for interest in top_interests:
            if action in INTEREST_ACTION_BIAS.get(interest, []):
                weights[idx] += 0.4
        
        # 敏感期加权（按强度调节）
        for sensitivity, strength in active_sensitivities.items():
            if action in SENSITIVITY_ACTION_BIAS.get(sensitivity, []):
                weights[idx] += min(0.8, strength) * 0.6
    
    return weights


# 通用年龄适配决策函数
def get_age_appropriate_action(child_state: ChildState, family_state: FamilyState, base_action_types: List[str], base_cost_range: List[float]) -> tuple:
    """
    获取适合孩子年龄的行为类型和成本
    
    参数:
        child_state: 孩子当前状态
        family_state: 家庭当前状态
        base_action_types: 基础行为类型列表
        base_cost_range: 基础行为成本范围
        
    返回:
        (action_type, cost): 适合年龄的行为类型和对应的成本
    """
    # 计算孩子当前年龄和年龄阶段
    age = child_state.calculate_age(family_state.current_date)
    age_group = child_state.get_age_group(family_state.current_date)
    
    # 过滤出适合当前年龄的行为类型
    appropriate_actions = []
    appropriate_costs = []
    
    for action, cost in zip(base_action_types, base_cost_range):
        if age_group in ACTION_AGE_MAPPING.get(action, []):
            appropriate_actions.append(action)
            appropriate_costs.append(cost)
    
    # 如果没有适合的行为，使用默认行为
    if not appropriate_actions:
        appropriate_actions = ["陪伴"]
        appropriate_costs = [50.0]
    
    # 随机选择一个适合的行为
    weights = _calculate_action_weights(appropriate_actions, child_state)
    action_idx = random.choices(range(len(appropriate_actions)), weights=weights, k=1)[0]
    return appropriate_actions[action_idx], appropriate_costs[action_idx]


class MockDeepSeekAgent(FamilyAgent):
    """
    DeepSeek模型适配器，使用Mock数据模拟API返回结果
    """
    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        使用Mock数据模拟DeepSeek模型的决策结果
        
        参数:
            child_state: 孩子当前的状态
            family_state: 家庭当前的状态
            event: 当前周发生的事件描述
            
        返回:
            模拟的决策结果
        """
        # Mock数据，根据孩子状态和事件生成不同的决策
        base_action_types = ["辅导", "鼓励", "花钱培训", "陪伴", "严格要求", "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养"]
        base_cost_range = [0, 50, 200, 100, 0, 0, 30, 0, 100]
        
        # 根据多维度状态调整决策偏好（增强版）
        preferred_action_types = base_action_types.copy()
        preferred_cost_range = base_cost_range.copy()
        
        # 规则1: 压力极高（>80）时，优先减压
        if child_state.stress > 80:
            stress_relief_actions = ["陪伴", "游戏互动", "鼓励"]
            preferred_action_types = [action for action in preferred_action_types if action in stress_relief_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in stress_relief_actions]
        # 规则2: 压力较高（70-80）时，平衡减压和学习
        elif child_state.stress > 70:
            stress_relief_actions = ["鼓励", "陪伴", "游戏互动", "启蒙教育", "简单辅导"]
            preferred_action_types = [action for action in preferred_action_types if action in stress_relief_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in stress_relief_actions]
        # 规则3: 知识储备极低（<30）时，优先学习
        elif child_state.knowledge < 30:
            learning_actions = ["辅导", "简单辅导", "启蒙教育", "花钱培训"]
            preferred_action_types = [action for action in preferred_action_types if action in learning_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in learning_actions]
        # 规则4: 知识储备较低（30-50）时，平衡学习和兴趣
        elif child_state.knowledge < 50:
            learning_actions = ["辅导", "简单辅导", "启蒙教育", "花钱培训", "简单兴趣培养"]
            preferred_action_types = [action for action in preferred_action_types if action in learning_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in learning_actions]
        # 规则5: 健康值较低（<60）时，关注健康
        elif child_state.physical_health < 60:
            health_actions = ["陪伴", "健康教育", "游戏互动"]
            preferred_action_types = [action for action in preferred_action_types if action in health_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in health_actions]
        # 规则6: 关系值较低时，优先改善关系
        relationship_key = f"{self.member}_relationship"
        relationship_value = getattr(child_state, relationship_key, 100.0)
        if relationship_value < 50:
            relationship_actions = ["陪伴", "沟通", "鼓励", "游戏互动"]
            preferred_action_types = [action for action in preferred_action_types if action in relationship_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in relationship_actions]
        # 规则7: 经济紧张时，避免高成本活动
        if family_state.family_savings < 10000:
            # 移除高成本活动
            expensive_actions = ["花钱培训", "简单兴趣培养"]
            preferred_action_types = [action for action in preferred_action_types if action not in expensive_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in preferred_action_types]
        
        # 获取适合年龄的行为
        action_type, cost = get_age_appropriate_action(child_state, family_state, preferred_action_types, preferred_cost_range)
        
        # 根据不同角色生成对应的对话
        if self.member in ["grandfather", "grandmother"]:
            child_call = "孙女"
        else:
            child_call = "女儿"
        self_call = get_member_name_cn(self.member)
        
        # 基于年龄和行为类型生成合适的对话
        dialogues = {
            "陪伴": f"{child_call}，今天{self_call}陪你去{random.choice(['公园', '动物园', '游乐场'])}玩，放松一下。",
            "启蒙教育": f"{child_call}，今天{self_call}教你认识一些新的{random.choice(['动物', '植物', '颜色', '数字'])}。",
            "游戏互动": f"{child_call}，今天{self_call}陪你玩{random.choice(['积木', '拼图', '绘画', '角色扮演'])}游戏。",
            "简单辅导": f"{child_call}，今天{self_call}帮你复习一下{random.choice(['字母', '数字', '简单汉字'])}。",
            "简单兴趣培养": f"{child_call}，{self_call}给你报了一个{random.choice(['绘画', '舞蹈', '音乐'])}兴趣班，培养一下你的兴趣。",
            "辅导": f"{child_call}，今天{self_call}帮你辅导一下功课，{event}期间更要好好学习。",
            "鼓励": f"{child_call}，最近表现不错，继续加油！{event}虽然有影响，但我们一起克服。",
            "花钱培训": f"{child_call}，{self_call}给你报了一个{random.choice(['数学', '英语', '编程'])}培训班，希望对你有帮助。",
            "严格要求": f"{child_call}，最近学习有点松懈了，{event}不是借口，要更加努力才行！"
        }
        
        # 默认对话，防止找不到对应的对话
        default_dialogue = f"{child_call}，今天{self_call}陪你做一些有趣的事情。"
        dialogue = dialogues.get(action_type, default_dialogue)
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost
        }


class MockQwenAgent(FamilyAgent):
    """
    Qwen模型适配器，使用Mock数据模拟API返回结果
    """
    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        使用Mock数据模拟Qwen模型的决策结果
        
        参数:
            child_state: 孩子当前的状态
            family_state: 家庭当前的状态
            event: 当前周发生的事件描述
            
        返回:
            模拟的决策结果
        """
        # Mock数据，Qwen模型更注重平衡
        base_action_types = ["辅导", "鼓励", "陪伴", "花钱培训", "沟通", "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养"]
        base_cost_range = [0, 0, 50, 150, 0, 0, 30, 0, 100]
        
        # 根据多维度状态调整决策偏好（增强版）
        relationship_key = f"{self.member}_relationship"
        relationship_value = getattr(child_state, relationship_key, 100.0)
        
        # 调整决策偏好
        preferred_action_types = base_action_types.copy()
        preferred_cost_range = base_cost_range.copy()
        
        # 规则1: 关系极差（<30）时，优先沟通
        if relationship_value < 30:
            relationship_actions = ["沟通", "陪伴", "游戏互动"]
            preferred_action_types = [action for action in preferred_action_types if action in relationship_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in relationship_actions]
        # 规则2: 关系较差（30-50）时，平衡沟通和学习
        elif relationship_value < 50:
            relationship_actions = ["沟通", "陪伴", "游戏互动", "启蒙教育", "简单辅导"]
            preferred_action_types = [action for action in preferred_action_types if action in relationship_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in relationship_actions]
        # 规则3: 压力高时，优先减压
        elif child_state.stress > 70:
            stress_relief_actions = ["鼓励", "陪伴", "沟通", "游戏互动"]
            preferred_action_types = [action for action in preferred_action_types if action in stress_relief_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in stress_relief_actions]
        # 规则4: 知识储备低时，优先学习
        elif child_state.knowledge < 50:
            learning_actions = ["辅导", "简单辅导", "启蒙教育", "花钱培训"]
            preferred_action_types = [action for action in preferred_action_types if action in learning_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in learning_actions]
        
        # 规则5: 经济紧张时，避免高成本活动
        if family_state.family_savings < 15000:
            expensive_actions = ["花钱培训"]
            preferred_action_types = [action for action in preferred_action_types if action not in expensive_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in preferred_action_types]
        
        # 获取适合年龄的行为
        action_type, cost = get_age_appropriate_action(child_state, family_state, preferred_action_types, preferred_cost_range)
        
        # 获取适合当前角色的称呼
        if self.member in ["grandfather", "grandmother"]:
            child_call = "孙女"
        else:
            child_call = "女儿"
        self_call = get_member_name_cn(self.member)
        
        dialogues = {
            "辅导": f"{child_call}，来，{self_call}帮你看看这个题目，最近学习怎么样？",
            "鼓励": f"{child_call}，你已经很棒了，继续保持！{event}我们一起面对。",
            "陪伴": f"{child_call}，今天{self_call}带你去{random.choice(['公园', '图书馆', '博物馆'])}，放松一下心情。",
            "花钱培训": f"{child_call}，{self_call}给你买了一套{random.choice(['学习资料', '练习册', '在线课程'])}，希望对你有帮助。",
            "沟通": f"{child_call}，最近有什么心事吗？可以和{self_call}说说，我们一起解决。",
            "启蒙教育": f"{child_call}，今天{self_call}教你认识一些有趣的东西，好不好？",
            "游戏互动": f"{child_call}，今天{self_call}陪你玩个好玩的游戏，怎么样？",
            "简单辅导": f"{child_call}，来，{self_call}帮你复习一下今天学的内容。",
            "简单兴趣培养": f"{child_call}，{self_call}给你报了一个简单的兴趣班，我们试试看。"
        }
        
        # 默认对话，防止找不到对应的对话
        default_dialogue = f"{child_call}，今天{self_call}陪你聊聊天，怎么样？"
        dialogue = dialogues.get(action_type, default_dialogue)
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost
        }


class MockKimiAgent(FamilyAgent):
    """
    Kimi模型适配器，使用Mock数据模拟API返回结果
    """
    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        使用Mock数据模拟Kimi模型的决策结果
        
        参数:
            child_state: 孩子当前的状态
            family_state: 家庭当前的状态
            event: 当前周发生的事件描述
            
        返回:
            模拟的决策结果
        """
        # Mock数据，Kimi模型更注重实际效果
        base_action_types = ["辅导", "花钱培训", "严格要求", "鼓励", "监督学习", "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养"]
        base_cost_range = [0, 250, 0, 0, 0, 0, 30, 0, 100]
        
        # 根据多维度状态调整决策偏好（增强版）
        preferred_action_types = base_action_types.copy()
        preferred_cost_range = base_cost_range.copy()
        
        # 规则1: 知识储备极低（<40）时，优先培训
        if child_state.knowledge < 40:
            learning_actions = ["辅导", "简单辅导", "启蒙教育", "花钱培训", "监督学习"]
            preferred_action_types = [action for action in preferred_action_types if action in learning_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in learning_actions]
        # 规则2: 知识储备较低（40-60）时，平衡学习和鼓励
        elif child_state.knowledge < 60:
            learning_actions = ["辅导", "简单辅导", "启蒙教育", "花钱培训", "简单兴趣培养", "鼓励"]
            preferred_action_types = [action for action in preferred_action_types if action in learning_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in learning_actions]
        # 规则3: 压力高时，避免严格要求
        elif child_state.stress > 70:
            stress_relief_actions = ["鼓励", "陪伴", "游戏互动", "启蒙教育"]
            preferred_action_types = [action for action in preferred_action_types if action in stress_relief_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in stress_relief_actions]
        
        # 规则4: 经济紧张时，避免高成本活动
        if family_state.family_savings < 12000:
            expensive_actions = ["花钱培训"]
            preferred_action_types = [action for action in preferred_action_types if action not in expensive_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in preferred_action_types]
        
        # 获取适合年龄的行为
        action_type, cost = get_age_appropriate_action(child_state, family_state, preferred_action_types, preferred_cost_range)
        
        # 获取适合当前角色的称呼
        if self.member in ["grandfather", "grandmother"]:
            child_call = "孙女"
        else:
            child_call = "女儿"
        self_call = get_member_name_cn(self.member)
        
        dialogues = {
            "辅导": f"{child_call}，{self_call}今天专门研究了你的教材，来帮你辅导一下。",
            "花钱培训": f"{child_call}，{self_call}给你报了一个一对一辅导，希望能提高你的成绩。",
            "严格要求": f"{child_call}，最近作业完成得不太认真，{event}期间更要自律！",
            "鼓励": f"{child_call}，你的进步{self_call}都看在眼里，继续努力！",
            "监督学习": f"{child_call}，今天{self_call}陪你一起学习，咱们制定一个学习计划。",
            "启蒙教育": f"{child_call}，今天{self_call}教你一些有用的知识，好不好？",
            "游戏互动": f"{child_call}，今天{self_call}陪你玩个能学到东西的游戏。",
            "简单辅导": f"{child_call}，来，{self_call}帮你复习一下今天学的内容。",
            "简单兴趣培养": f"{child_call}，{self_call}给你报了一个{random.choice(['绘画', '舞蹈', '音乐'])}兴趣班，培养一下你的兴趣。"
        }
        
        # 默认对话，防止找不到对应的对话
        default_dialogue = f"{child_call}，今天{self_call}陪你做一些有意义的事情。"
        dialogue = dialogues.get(action_type, default_dialogue)
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost
        }


class MockChatGPTAgent(FamilyAgent):
    """
    ChatGPT模型适配器，使用Mock数据模拟API返回结果
    """
    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        使用Mock数据模拟ChatGPT模型的决策结果
        
        参数:
            child_state: 孩子当前的状态
            family_state: 家庭当前的状态
            event: 当前周发生的事件描述
            
        返回:
            模拟的决策结果
        """
        # Mock数据，ChatGPT模型更注重全面发展
        base_action_types = ["辅导", "鼓励", "陪伴", "花钱培训", "健康教育", "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养"]
        base_cost_range = [0, 0, 80, 180, 0, 0, 30, 0, 100]
        
        # 根据多维度状态调整决策偏好（增强版）
        preferred_action_types = base_action_types.copy()
        preferred_cost_range = base_cost_range.copy()
        
        # 规则1: 健康值极低（<50）时，优先健康
        if child_state.physical_health < 50:
            health_actions = ["陪伴", "健康教育", "游戏互动"]
            preferred_action_types = [action for action in preferred_action_types if action in health_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in health_actions]
        # 规则2: 健康值较低（50-60）时，平衡健康和学习
        elif child_state.physical_health < 60:
            health_actions = ["陪伴", "健康教育", "游戏互动", "启蒙教育", "简单辅导"]
            preferred_action_types = [action for action in preferred_action_types if action in health_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in health_actions]
        # 规则3: 压力高时，优先减压
        elif child_state.stress > 70:
            stress_relief_actions = ["鼓励", "陪伴", "游戏互动", "健康教育"]
            preferred_action_types = [action for action in preferred_action_types if action in stress_relief_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in stress_relief_actions]
        # 规则4: 知识储备低时，平衡学习和健康
        elif child_state.knowledge < 50:
            balanced_actions = ["辅导", "简单辅导", "启蒙教育", "陪伴", "健康教育"]
            preferred_action_types = [action for action in preferred_action_types if action in balanced_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in balanced_actions]
        
        # 规则5: 经济紧张时，避免高成本活动
        if family_state.family_savings < 15000:
            expensive_actions = ["花钱培训"]
            preferred_action_types = [action for action in preferred_action_types if action not in expensive_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in preferred_action_types]
        
        # 获取适合年龄的行为
        action_type, cost = get_age_appropriate_action(child_state, family_state, preferred_action_types, preferred_cost_range)
        
        # 获取适合当前角色的称呼
        if self.member in ["grandfather", "grandmother"]:
            child_call = "孙女"
        else:
            child_call = "女儿"
        self_call = get_member_name_cn(self.member)
        
        dialogues = {
            "辅导": f"{child_call}，{self_call}帮你分析一下最近的学习情况，制定一个改进计划。",
            "鼓励": f"{child_call}，你已经做得很好了，相信自己，继续前进！",
            "陪伴": f"{child_call}，今天{self_call}陪你去{random.choice(['爬山', '打球', '游泳'])}，增强体质。",
            "花钱培训": f"{child_call}，{self_call}给你报了一个{random.choice(['英语', '艺术', '体育'])}兴趣班，全面发展。",
            "健康教育": f"{child_call}，学习很重要，但身体更重要，要注意休息和锻炼。",
            "启蒙教育": f"{child_call}，今天{self_call}教你一些关于健康的小知识。",
            "游戏互动": f"{child_call}，今天{self_call}陪你玩个健康的游戏。",
            "简单辅导": f"{child_call}，来，{self_call}帮你复习一下今天学的内容。",
            "简单兴趣培养": f"{child_call}，{self_call}给你报了一个{random.choice(['绘画', '舞蹈', '音乐'])}兴趣班，培养一下你的兴趣。"
        }
        
        # 默认对话，防止找不到对应的对话
        default_dialogue = f"{child_call}，今天{self_call}陪你做一些有趣的事情。"
        dialogue = dialogues.get(action_type, default_dialogue)
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost
        }


class MockGeminiAgent(FamilyAgent):
    """
    Gemini模型适配器，使用Mock数据模拟API返回结果
    """
    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        使用Mock数据模拟Gemini模型的决策结果
        
        参数:
            child_state: 孩子当前的状态
            family_state: 家庭当前的状态
            event: 当前周发生的事件描述
            
        返回:
            模拟的决策结果
        """
        # Mock数据，Gemini模型更注重创新教育
        base_action_types = ["辅导", "鼓励", "花钱培训", "创新活动", "讨论交流", "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养"]
        base_cost_range = [0, 0, 150, 100, 0, 0, 30, 0, 100]
        
        # 根据多维度状态调整决策偏好（增强版）
        preferred_action_types = base_action_types.copy()
        preferred_cost_range = base_cost_range.copy()
        
        # 规则1: 知识储备高（>70）时，鼓励创新
        if child_state.knowledge > 70:
            innovation_actions = ["创新活动", "讨论交流", "花钱培训", "简单兴趣培养"]
            preferred_action_types = [action for action in preferred_action_types if action in innovation_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in innovation_actions]
        # 规则2: 知识储备低时，优先基础学习
        elif child_state.knowledge < 50:
            learning_actions = ["辅导", "简单辅导", "启蒙教育", "花钱培训"]
            preferred_action_types = [action for action in preferred_action_types if action in learning_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in learning_actions]
        # 规则3: 压力高时，优先减压
        elif child_state.stress > 70:
            stress_relief_actions = ["鼓励", "讨论交流", "游戏互动", "启蒙教育"]
            preferred_action_types = [action for action in preferred_action_types if action in stress_relief_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in stress_relief_actions]
        
        # 规则4: 经济紧张时，避免高成本活动
        if family_state.family_savings < 15000:
            expensive_actions = ["花钱培训"]
            preferred_action_types = [action for action in preferred_action_types if action not in expensive_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in preferred_action_types]
        
        # 获取适合年龄的行为
        action_type, cost = get_age_appropriate_action(child_state, family_state, preferred_action_types, preferred_cost_range)
        
        # 获取适合当前角色的称呼
        if self.member in ["grandfather", "grandmother"]:
            child_call = "孙女"
        else:
            child_call = "女儿"
        self_call = get_member_name_cn(self.member)
        
        dialogues = {
            "辅导": f"{child_call}，{self_call}教你一个新的学习方法，可能会对你有帮助。",
            "鼓励": f"{child_call}，你的想法很有创意，继续保持这种思维方式！",
            "花钱培训": f"{child_call}，{self_call}给你报了一个{random.choice(['编程', '机器人', '创客'])}班，培养你的创新能力。",
            "创新活动": f"{child_call}，今天我们一起做个小实验，培养你的动手能力。",
            "讨论交流": f"{child_call}，你对{event}有什么看法？我们一起讨论一下。",
            "启蒙教育": f"{child_call}，今天{self_call}教你一些有趣的科学知识。",
            "游戏互动": f"{child_call}，今天{self_call}陪你玩个创新游戏。",
            "简单辅导": f"{child_call}，来，{self_call}帮你复习一下今天学的内容。",
            "简单兴趣培养": f"{child_call}，{self_call}给你报了一个{random.choice(['绘画', '机器人', '科学'])}兴趣班，培养一下你的兴趣。"
        }
        
        # 默认对话，防止找不到对应的对话
        default_dialogue = f"{child_call}，今天{self_call}陪你做一些有趣的事情。"
        dialogue = dialogues.get(action_type, default_dialogue)
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost
        }


class MockClaudeAgent(FamilyAgent):
    """
    Claude模型适配器，使用Mock数据模拟API返回结果
    """
    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        使用Mock数据模拟Claude模型的决策结果
        
        参数:
            child_state: 孩子当前的状态
            family_state: 家庭当前的状态
            event: 当前周发生的事件描述
            
        返回:
            模拟的决策结果
        """
        # Mock数据，Claude模型更注重情感关怀
        base_action_types = ["鼓励", "陪伴", "沟通", "辅导", "花钱培训", "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养"]
        base_cost_range = [0, 60, 0, 0, 120, 0, 30, 0, 100]
        
        # 根据多维度状态调整决策偏好（增强版）
        relationship_key = f"{self.member}_relationship"
        relationship_value = getattr(child_state, relationship_key, 100.0)
        
        # 调整决策偏好
        preferred_action_types = base_action_types.copy()
        preferred_cost_range = base_cost_range.copy()
        
        # 规则1: 关系极差（<30）时，优先沟通
        if relationship_value < 30:
            relationship_actions = ["沟通", "陪伴", "鼓励"]
            preferred_action_types = [action for action in preferred_action_types if action in relationship_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in relationship_actions]
        # 规则2: 关系较差（30-50）时，平衡沟通和学习
        elif relationship_value < 50:
            relationship_actions = ["沟通", "陪伴", "鼓励", "游戏互动", "启蒙教育"]
            preferred_action_types = [action for action in preferred_action_types if action in relationship_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in relationship_actions]
        # 规则3: 压力高时，优先情感关怀
        elif child_state.stress > 70:
            stress_relief_actions = ["鼓励", "陪伴", "沟通", "游戏互动"]
            preferred_action_types = [action for action in preferred_action_types if action in stress_relief_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in stress_relief_actions]
        # 规则4: 知识储备低时，平衡学习和情感
        elif child_state.knowledge < 50:
            balanced_actions = ["辅导", "简单辅导", "启蒙教育", "鼓励", "陪伴"]
            preferred_action_types = [action for action in preferred_action_types if action in balanced_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in balanced_actions]
        
        # 规则5: 经济紧张时，避免高成本活动
        if family_state.family_savings < 15000:
            expensive_actions = ["花钱培训"]
            preferred_action_types = [action for action in preferred_action_types if action not in expensive_actions]
            preferred_cost_range = [cost for action, cost in zip(base_action_types, base_cost_range) if action in preferred_action_types]
        
        # 获取适合年龄的行为
        action_type, cost = get_age_appropriate_action(child_state, family_state, preferred_action_types, preferred_cost_range)
        
        # 获取适合当前角色的称呼
        if self.member in ["grandfather", "grandmother"]:
            child_call = "孙女"
        else:
            child_call = "女儿"
        self_call = get_member_name_cn(self.member)
        
        dialogues = {
            "鼓励": f"{child_call}，{self_call}知道你最近很努力，不管结果如何，你都是最棒的！",
            "陪伴": f"{child_call}，今天{self_call}什么都不做，专门陪你，你想做什么？",
            "沟通": f"{child_call}，你最近是不是有什么心事？可以和{self_call}说说吗？",
            "辅导": f"{child_call}，{self_call}虽然工作忙，但会尽量抽时间帮你辅导功课。",
            "花钱培训": f"{child_call}，{self_call}给你报了一个{random.choice(['心理辅导', '兴趣班', '学习方法'])}课程，希望能帮助你。",
            "启蒙教育": f"{child_call}，今天{self_call}教你一些有趣的知识。",
            "游戏互动": f"{child_call}，今天{self_call}陪你玩个放松的游戏。",
            "简单辅导": f"{child_call}，来，{self_call}帮你复习一下今天学的内容。",
            "简单兴趣培养": f"{child_call}，{self_call}给你报了一个{random.choice(['绘画', '舞蹈', '音乐'])}兴趣班，培养一下你的兴趣。"
        }
        
        # 默认对话，防止找不到对应的对话
        default_dialogue = f"{child_call}，今天{self_call}陪你做一些有趣的事情。"
        dialogue = dialogues.get(action_type, default_dialogue)
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost
        }


class MockGrokAgent(FamilyAgent):
    """
    Grok模型适配器，使用Mock数据模拟API返回结果
    """
    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        使用Mock数据模拟Grok模型的决策结果
        
        参数:
            child_state: 孩子当前的状态
            family_state: 家庭当前的状态
            event: 当前周发生的事件描述
            
        返回:
            模拟的决策结果
        """
        # Mock数据，Grok模型更注重个性化教育
        base_action_types = ["辅导", "个性化计划", "鼓励", "花钱培训", "实践活动", "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养"]
        base_cost_range = [0, 0, 0, 180, 80, 0, 30, 0, 100]
        
        # 调整决策偏好，增加个性化相关活动
        preferred_action_types = base_action_types.copy()
        preferred_cost_range = base_cost_range.copy()
        
        # 获取适合年龄的行为
        action_type, cost = get_age_appropriate_action(child_state, family_state, preferred_action_types, preferred_cost_range)
        
        # 获取适合当前角色的称呼
        if self.member in ["grandfather", "grandmother"]:
            child_call = "孙女"
        else:
            child_call = "女儿"
        self_call = get_member_name_cn(self.member)
        
        dialogues = {
            "辅导": f"{child_call}，根据你的学习特点，{self_call}专门准备了辅导内容。",
            "个性化计划": f"{child_call}，{self_call}给你制定了一个个性化学习计划，咱们一起执行。",
            "鼓励": f"{child_call}，你的进步非常明显，继续保持！",
            "花钱培训": f"{child_call}，{self_call}给你报了一个个性化辅导课程，根据你的情况量身定制。",
            "实践活动": f"{child_call}，今天我们一起去{random.choice(['科技馆', '工厂', '大学'])}参观，增长见识。",
            "启蒙教育": f"{child_call}，今天{self_call}教你一些有趣的知识。",
            "游戏互动": f"{child_call}，今天{self_call}陪你玩个个性化游戏。",
            "简单辅导": f"{child_call}，来，{self_call}帮你复习一下今天学的内容。",
            "简单兴趣培养": f"{child_call}，{self_call}给你报了一个{random.choice(['绘画', '舞蹈', '音乐'])}兴趣班，培养一下你的兴趣。"
        }
        
        # 默认对话，防止找不到对应的对话
        default_dialogue = f"{child_call}，今天{self_call}陪你做一些有趣的事情。"
        dialogue = dialogues.get(action_type, default_dialogue)
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost
        }


class AgentFactory:
    """
    智能体工厂类，用于创建不同模型的智能体实例
    采用工厂模式设计，实现模型实例化的统一入口
    
    支持两种模式：
    1. LLM 模式：使用真正的 LLM API 进行自主决策
    2. Mock 模式：使用规则生成决策（用于测试或 API 不可用时）
    """
    
    # Mock 模型适配器映射表
    _mock_agent_classes: Dict[str, Type[FamilyAgent]] = {
        "deepseek": MockDeepSeekAgent,
        "qwen": MockQwenAgent,
        "kimi": MockKimiAgent,
        "chatgpt": MockChatGPTAgent,
        "gemini": MockGeminiAgent,
        "claude": MockClaudeAgent,
        "grok": MockGrokAgent
    }
    
    # 缓存配置实例
    _config: Optional[SimulationConfig] = None
    
    # 是否强制使用 Mock 模式
    _force_mock: bool = False
    
    # 是否使用自适应 Agent（类人模式）
    _use_adaptive: bool = True
    
    # Agent 缓存（用于保持记忆连续性）
    _agent_cache: Dict[str, Any] = {}
    
    @classmethod
    def set_config(cls, config: SimulationConfig):
        """
        设置全局配置
        
        参数:
            config: 模拟配置实例
        """
        cls._config = config
        logger.info("AgentFactory 配置已更新")
    
    @classmethod
    def set_force_mock(cls, force_mock: bool):
        """
        设置是否强制使用 Mock 模式
        
        参数:
            force_mock: 是否强制使用 Mock
        """
        cls._force_mock = force_mock
        logger.info(f"AgentFactory 强制 Mock 模式: {force_mock}")
    
    @classmethod
    def set_use_adaptive(cls, use_adaptive: bool):
        """
        设置是否使用自适应 Agent（类人模式）
        
        参数:
            use_adaptive: 是否使用自适应模式
        """
        cls._use_adaptive = use_adaptive
        logger.info(f"AgentFactory 自适应模式: {use_adaptive}")
    
    @classmethod
    def clear_agent_cache(cls):
        """清空 Agent 缓存"""
        cls._agent_cache.clear()
        logger.info("AgentFactory Agent 缓存已清空")
    
    @classmethod
    def create_agent(cls, model_name: str, member: str = "father", config: SimulationConfig = None, 
                     use_cache: bool = True) -> FamilyAgent:
        """
        创建指定模型的智能体实例
        
        优先级：
        1. 自适应 Agent（类人模式，带记忆和学习能力）
        2. LLM Agent（真正的大模型决策）
        3. Mock Agent（规则决策）
        
        参数:
            model_name: 模型名称，支持：deepseek, qwen, kimi, chatgpt, gemini, claude, grok
            member: 家庭成员角色，可选值：father, mother, grandfather, grandmother
            config: 可选的配置实例，如果为 None 则使用全局配置
            use_cache: 是否使用缓存（保持记忆连续性）
            
        返回:
            对应的智能体实例
            
        异常:
            ValueError: 不支持的模型名称
        """
        model_name = model_name.lower()
        
        # 验证模型名称
        if model_name not in cls._mock_agent_classes:
            supported_models = ", ".join(cls._mock_agent_classes.keys())
            raise ValueError(f"不支持的模型名称: {model_name}，支持的模型有: {supported_models}")
        
        # 使用传入的配置或全局配置
        use_config = config or cls._config
        
        # 生成缓存键
        cache_key = f"{model_name}_{member}"
        
        # 检查缓存（对于自适应Agent，必须使用缓存以保持记忆）
        if use_cache and cache_key in cls._agent_cache:
            logger.debug(f"[{model_name}] 使用缓存的 Agent (保持记忆)")
            return cls._agent_cache[cache_key]
        
        # 如果强制 Mock 模式，直接返回 Mock Agent
        if cls._force_mock:
            logger.info(f"[{model_name}] 强制使用 Mock Agent")
            agent = cls._create_mock_agent(model_name, member)
            if use_cache:
                cls._agent_cache[cache_key] = agent
            return agent
        
        # 优先尝试创建自适应 Agent（类人模式）
        if cls._use_adaptive and ADAPTIVE_AGENT_AVAILABLE and use_config is not None:
            try:
                agent = AdaptiveParentAgent(model_name, member, use_config)
                logger.info(f"[{model_name}] 创建自适应 Agent（类人模式：记忆+学习+情绪）")
                if use_cache:
                    cls._agent_cache[cache_key] = agent
                return agent
            except Exception as e:
                logger.warning(f"[{model_name}] 创建自适应 Agent 失败: {e}")
        
        # 尝试创建 LLM Agent
        if LLM_AGENTS_AVAILABLE and use_config is not None:
            try:
                # 检查是否有该模型的 API 配置
                model_config = use_config.get_model_config(model_name)
                
                # 验证 API 密钥是否有效（不是占位符）
                api_key = model_config.get("api_key", "")
                if api_key and not cls._is_placeholder_key(api_key):
                    # 创建 LLM Agent
                    agent_class = LLM_AGENT_CLASSES.get(model_name)
                    if agent_class:
                        agent = agent_class(model_name, member, use_config)
                        logger.info(f"[{model_name}] 创建 LLM Agent（大模型决策）")
                        if use_cache:
                            cls._agent_cache[cache_key] = agent
                        return agent
                else:
                    logger.info(f"[{model_name}] API 密钥无效，使用 Mock Agent")
                    
            except Exception as e:
                logger.warning(f"[{model_name}] 创建 LLM Agent 失败: {e}")
        
        # 降级到 Mock Agent
        agent = cls._create_mock_agent(model_name, member)
        if use_cache:
            cls._agent_cache[cache_key] = agent
        return agent
    
    @classmethod
    def _is_placeholder_key(cls, api_key: str) -> bool:
        """
        检查 API 密钥是否为占位符
        
        参数:
            api_key: API 密钥
            
        返回:
            是否为占位符
        """
        if not api_key:
            return True
        
        # 常见的占位符模式
        placeholders = [
            "your_",
            "xxx",
            "placeholder",
            "example",
            "test_key",
            "sk-xxx",
            "api_key_here"
        ]
        
        api_key_lower = api_key.lower()
        return any(p in api_key_lower for p in placeholders) or len(api_key) < 10
    
    @classmethod
    def _create_mock_agent(cls, model_name: str, member: str) -> FamilyAgent:
        """
        创建 Mock Agent
        
        参数:
            model_name: 模型名称
            member: 家庭成员角色
            
        返回:
            Mock Agent 实例
        """
        agent_class = cls._mock_agent_classes[model_name]
        return agent_class(model_name, member)
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        """
        获取支持的模型列表
        
        返回:
            支持的模型名称列表
        """
        return list(cls._mock_agent_classes.keys())
    
    @classmethod
    def get_supported_members(cls) -> list[str]:
        """
        获取支持的家庭成员列表
        
        返回:
            支持的家庭成员名称列表
        """
        return ["father", "mother", "grandfather", "grandmother"]
    
    @classmethod
    def get_model_status(cls, config: SimulationConfig = None) -> Dict[str, Dict[str, Any]]:
        """
        获取所有模型的状态信息
        
        参数:
            config: 可选的配置实例
            
        返回:
            模型状态字典，包含每个模型的运行模式
        """
        use_config = config or cls._config or SimulationConfig()
        status = {}
        
        for model_name in cls._mock_agent_classes.keys():
            try:
                model_config = use_config.get_model_config(model_name)
                api_key = model_config.get("api_key", "")
                
                has_valid_key = api_key and not cls._is_placeholder_key(api_key)
                
                # 确定运行模式
                if cls._use_adaptive and ADAPTIVE_AGENT_AVAILABLE:
                    mode = "Adaptive"  # 自适应类人模式
                    mode_desc = "类人"
                elif has_valid_key and LLM_AGENTS_AVAILABLE:
                    mode = "LLM"
                    mode_desc = "大模型"
                else:
                    mode = "Mock"
                    mode_desc = "规则"
                
                status[model_name] = {
                    "has_api_key": has_valid_key,
                    "mode": mode,
                    "mode_desc": mode_desc,
                    "adaptive": cls._use_adaptive and ADAPTIVE_AGENT_AVAILABLE,
                    "base_url": model_config.get("base_url", ""),
                    "model": model_config.get("model", "")
                }
            except Exception as e:
                status[model_name] = {
                    "has_api_key": False,
                    "mode": "Mock",
                    "mode_desc": "规则",
                    "error": str(e)
                }
        
        return status
    
    @classmethod
    def print_model_status(cls, config: SimulationConfig = None):
        """
        打印所有模型的状态信息
        
        参数:
            config: 可选的配置实例
        """
        status = cls.get_model_status(config)
        
        print("\n" + "=" * 70)
        print("模型决策模式状态")
        print("=" * 70)
        
        for model_name, info in status.items():
            mode = info.get("mode", "Unknown")
            mode_desc = info.get("mode_desc", "未知")
            model_id = info.get("model", "N/A")
            
            if mode == "Adaptive":
                icon = "[Adaptive]"
            elif mode == "LLM":
                icon = "[LLM]"
            else:
                icon = "[Mock]"
            
            print(f"{icon:12} {model_name:12} | {mode_desc:6} | 模型: {model_id}")
        
        print("=" * 70)
        print("[Adaptive] = 类人模式(记忆+学习+情绪)")
        print("[LLM]      = 大模型决策")
        print("[Mock]     = 规则决策")
        print("=" * 70 + "\n")


# 单元测试
async def test_agent_factory():
    """
    测试AgentFactory的功能
    """
    print("=== 测试AgentFactory ===")
    
    # 测试获取支持的模型列表
    supported_models = AgentFactory.get_supported_models()
    print(f"支持的模型: {supported_models}")
    
    # 测试获取支持的家庭成员列表
    supported_members = AgentFactory.get_supported_members()
    print(f"支持的家庭成员: {supported_members}")
    
    # 测试创建不同模型的智能体
    for model_name in supported_models:
        print(f"\n--- 测试{model_name}智能体 ---")
        try:
            # 测试创建不同家庭成员的智能体
            for member in supported_members[:2]:  # 只测试前两个家庭成员，减少测试时间
                print(f"\n创建{member}角色的{model_name}智能体")
                agent = AgentFactory.create_agent(model_name, member)
                print(f"成功创建{member}角色的{model_name}智能体: {agent.model_name}")
                
                # 测试decide方法
                child_state = ChildState(
                    knowledge=60.0, 
                    stress=40.0, 
                    father_relationship=70.0,
                    mother_relationship=75.0,
                    grandfather_relationship=65.0,
                    grandmother_relationship=68.0
                )
                family_state = FamilyState(family_savings=10000.0)
                event = "本周没有特别事件"
                
                decision = await agent.decide(child_state, family_state, event)
                print(f"决策结果: {decision}")
                
                # 验证返回格式
                assert "action_type" in decision, f"{model_name}返回结果缺少action_type字段"
                assert "dialogue" in decision, f"{model_name}返回结果缺少dialogue字段"
                assert "cost" in decision, f"{model_name}返回结果缺少cost字段"
                assert isinstance(decision["action_type"], str), f"{model_name}action_type字段类型错误"
                assert isinstance(decision["dialogue"], str), f"{model_name}dialogue字段类型错误"
                assert isinstance(decision["cost"], (int, float)), f"{model_name}cost字段类型错误"
                
                print(f"{member}角色的{model_name}智能体测试通过")
            
        except Exception as e:
            print(f"{model_name}智能体测试失败: {e}")
    
    print("\n=== 所有智能体测试完成 ===")


# 运行单元测试
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent_factory())
