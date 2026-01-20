"""
真正的 LLM 自主决策智能体

这个模块实现了真正调用 LLM API 进行自主决策的智能体。
每个智能体根据孩子状态、家庭状态和当前事件，通过 LLM 生成教育决策。
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
from abc import ABC

from src.agents.base import FamilyAgent
from src.core.state import ChildState, FamilyState
from src.core.config import SimulationConfig
from src.core.llm_client import LLMClientFactory, LLMClient, MockLLMClient

logger = logging.getLogger(__name__)


class LLMDecisionAgent(FamilyAgent):
    """
    基于 LLM 的自主决策智能体基类
    
    这个类真正调用 LLM API 来生成教育决策，而不是使用硬编码规则。
    """
    
    # 根据年龄阶段的可用行动类型
    AGE_APPROPRIATE_ACTIONS = {
        "infant": [  # 0-3岁：婴幼儿专属行为
            "亲子互动", "日常照料", "感官刺激", "户外活动", "早期阅读",
            "安抚陪伴", "社交接触", "陪伴", "启蒙教育", "健康教育", "游戏互动", "鼓励"
        ],
        "preschool": [  # 3-6岁：学前阶段
            "陪伴", "启蒙教育", "游戏互动", "简单辅导", "鼓励", "简单兴趣培养",
            "健康教育", "创新活动", "沟通", "户外活动", "早期阅读", "亲子互动",
            "感官刺激", "社交接触", "安抚陪伴"
        ],
        "primary": [  # 6岁以上：全部行为
            "辅导", "鼓励", "花钱培训", "陪伴", "严格要求", "监督学习", 
            "健康教育", "创新活动", "个性化计划", "实践活动", "沟通",
            "启蒙教育", "游戏互动", "简单辅导", "简单兴趣培养",
            "亲子互动", "户外活动", "早期阅读", "社交接触"
        ]
    }
    
    # LLM可能生成的各种表述到标准行为类型的映射
    ACTION_TYPE_MAPPING = {
        # 婴幼儿期常见表述映射
        "每日陪伴": "亲子互动",
        "每日陪伴 + 观察需求": "日常照料",
        "购买婴儿基础用品并每日陪伴": "日常照料",
        "换尿布": "日常照料",
        "喂奶": "日常照料",
        "哄睡": "安抚陪伴",
        "抱抱": "亲子互动",
        "逗玩": "亲子互动",
        "听音乐": "感官刺激",
        "看绘本": "早期阅读",
        "讲故事": "早期阅读",
        "晒太阳": "户外活动",
        "公园散步": "户外活动",
        "带孩子玩": "游戏互动",
        "玩游戏": "游戏互动",
        "做游戏": "游戏互动",
        "玩耍": "游戏互动",
        # 通用表述映射
        "陪孩子": "陪伴",
        "陪伴孩子": "陪伴",
        "陪她": "陪伴",
        "陪他": "陪伴",
        "辅导作业": "辅导",
        "辅导功课": "辅导",
        "检查作业": "辅导",
        "帮助学习": "辅导",
        "报培训班": "花钱培训",
        "报班": "花钱培训",
        "买教材": "花钱培训",
        "表扬": "鼓励",
        "夸奖": "鼓励",
        "批评": "严格要求",
        "批评教育": "严格要求",
        "盯着学习": "监督学习",
        "督促学习": "监督学习",
        "谈心": "沟通",
        "聊天": "沟通",
        "运动": "健康教育",
        "锻炼": "健康教育",
        "户外运动": "户外活动",
    }
    
    def __init__(self, model_name: str, member: str = "father", config: SimulationConfig = None):
        """
        初始化 LLM 决策智能体
        
        参数:
            model_name: 模型名称 (deepseek, qwen, kimi, chatgpt, gemini, claude, grok)
            member: 家庭成员角色
            config: 模拟配置，用于获取 API 密钥
        """
        super().__init__(model_name, member)
        self.config = config or SimulationConfig()
        self._llm_client: Optional[LLMClient] = None
        self._model_config: Optional[Dict[str, str]] = None
        self._is_mock = False  # 标记是否降级到 Mock
        
        # 尝试初始化 LLM 客户端
        self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        try:
            self._model_config = self.config.get_model_config(self.model_name)
            self._llm_client = LLMClientFactory.create_client(self.config, self.model_name)
            
            if isinstance(self._llm_client, MockLLMClient):
                self._is_mock = True
                logger.warning(f"{self.model_name} 使用 Mock 客户端（API 未配置或配置无效）")
            else:
                logger.info(f"{self.model_name} LLM 客户端初始化成功")
                
        except Exception as e:
            self._is_mock = True
            logger.warning(f"{self.model_name} LLM 客户端初始化失败: {e}，将使用 Mock 降级")
    
    def _get_enhanced_system_prompt(self, child_state: ChildState, family_state: FamilyState) -> str:
        """
        获取增强版系统提示，包含更详细的决策指导
        
        参数:
            child_state: 孩子状态
            family_state: 家庭状态
            
        返回:
            增强版系统提示
        """
        # 计算孩子年龄和年龄阶段
        age = child_state.calculate_age(family_state.current_date)
        age_group = child_state.get_age_group(family_state.current_date)
        
        # 获取适合年龄的行动类型
        available_actions = self.AGE_APPROPRIATE_ACTIONS.get(age_group, self.AGE_APPROPRIATE_ACTIONS["primary"])
        
        # 获取成员角色信息
        member_info = self._get_member_info()
        
        # 根据年龄阶段生成特定指导
        age_specific_guidance = self._get_age_specific_guidance(age, age_group, member_info)
        
        return f"""你现在需要扮演一个2010-2030年间的中国普通工薪阶层{member_info['role']}。

## 你的身份设定
- 身份：{member_info['identity']}
- 家庭：有一个{age:.1f}岁的{member_info['child_call']}，家庭属于二线城市普通工薪家庭
- 经济状况：家庭当前存款 {family_state.family_savings:.0f} 元
- 性格特点：{member_info['personality']}
- 语言风格：{member_info['language_style']}

{age_specific_guidance}

## 决策原则

### 1. 状态优先级处理
- 【紧急】压力值 > 80：必须优先减压
- 【警告】压力值 > 60：注意平衡，避免增加压力
- 【关注】健康值 < 60：关注身体健康
- 【修复】亲子关系 < 50：需要改善关系

### 2. 经济约束
- 家庭存款 < 5000：禁止任何花费
- 家庭存款 < 10000：避免高成本活动（>100元）

## 输出格式要求
你必须返回一个有效的 JSON 对象，格式如下：
```json
{{
    "action_type": "行动类型（必须从下面的可用行动中选择）",
    "dialogue": "你对孩子说的话或内心独白（要符合你的角色和语言风格）",
    "cost": 花费金额（整数，单位：元）,
    "reasoning": "你做出这个决策的理由"
}}
```

## ⚠️ 关键限制
当前{member_info['child_call']}年龄：**{age:.1f}岁**（{self._get_age_group_name(age_group)}阶段）

**可用的行动类型（只能从以下选择）**：
{', '.join(available_actions)}

**不要**使用上述列表之外的行动类型！
"""
    
    def _get_member_info(self) -> Dict[str, str]:
        """获取家庭成员角色信息"""
        member_info = {
            "father": {
                "role": "父亲",
                "identity": "普通技术工人，大专学历",
                "personality": "严厉但内心关爱，不善于直接表达爱意，更倾向于通过实际行动关心孩子",
                "language_style": "朴实无华，表达直接，偶尔使用谚语或俗语",
                "child_call": "女儿"
            },
            "mother": {
                "role": "母亲", 
                "identity": "办公室文员，大专学历",
                "personality": "温和细腻，善于表达爱意，关注孩子的身心健康",
                "language_style": "亲切温暖，使用口语化表达",
                "child_call": "女儿"
            },
            "grandfather": {
                "role": "祖父",
                "identity": "退休工人，高中学历",
                "personality": "传统稳重，重视教育，喜欢讲述过去的故事和人生道理",
                "language_style": "传统保守，带有时代特色的表达",
                "child_call": "孙女"
            },
            "grandmother": {
                "role": "祖母",
                "identity": "家庭主妇，初中学历",
                "personality": "慈祥和蔼，特别宠爱孙女，关注生活细节和饮食起居",
                "language_style": "温柔慈祥，充满爱意",
                "child_call": "孙女"
            }
        }
        return member_info.get(self.member, member_info["father"])
    
    def _get_age_group_name(self, age_group: str) -> str:
        """获取年龄阶段的中文名称"""
        names = {
            "infant": "婴幼儿",
            "preschool": "学前",
            "primary": "小学及以上"
        }
        return names.get(age_group, "未知")
    
    def _get_age_specific_guidance(self, age: float, age_group: str, member_info: Dict[str, str]) -> str:
        """
        根据年龄阶段生成特定的行为指导
        
        参数:
            age: 孩子年龄
            age_group: 年龄阶段
            member_info: 家庭成员信息
            
        返回:
            年龄特定指导文本
        """
        child_call = member_info.get('child_call', '孩子')
        role = member_info.get('role', '家长')
        
        if age_group == "infant":
            if age < 1:
                return f"""## 🍼 婴儿期特别指导（0-1岁）
你的{child_call}现在只有{age:.1f}岁，是一个婴儿！

**这个阶段最重要的是**：
- 安全感建立：通过肌肤接触、眼神交流让孩子感受到爱
- 基本需求满足：及时响应哭闹，保证吃饱睡好
- 感官刺激：适当的声音、色彩、触感刺激促进大脑发育

**推荐行为**：亲子互动、日常照料、安抚陪伴、感官刺激
**绝对禁止**：任何形式的学习要求、培训班

**对话风格**：婴儿听不懂复杂语言，用温柔的语调、简单的词语，如"宝宝乖""爸爸在"。"""
            else:
                return f"""## 👶 幼儿期特别指导（1-3岁）
你的{child_call}现在{age:.1f}岁，正处于快速发育期！

**这个阶段最重要的是**：
- 语言启蒙：多和孩子说话，讲故事，认识事物
- 运动发展：学走路、跑跳，户外活动很重要
- 安全依恋：继续建立稳固的亲子关系

**推荐行为**：亲子互动、户外活动、早期阅读、游戏互动、启蒙教育
**绝对禁止**：辅导作业、培训班、严格要求（孩子太小了！）

**对话风格**：用简单的句子，充满爱意，如"宝贝真棒""我们去公园玩"。"""

        elif age_group == "preschool":
            return f"""## 🎨 学前期特别指导（3-6岁）
你的{child_call}现在{age:.1f}岁，正是学前教育的关键期！

**这个阶段最重要的是**：
- 好习惯养成：作息规律、自理能力
- 社交能力：学会和小朋友相处
- 兴趣启蒙：通过游戏发现孩子的兴趣

**推荐行为**：启蒙教育、游戏互动、简单辅导、户外活动、早期阅读
**谨慎使用**：过度的学业压力

**对话风格**：可以用更完整的句子，讲道理但要有耐心。"""

        else:  # primary
            return f"""## 📚 学龄期指导（6岁以上）
你的{child_call}现在{age:.1f}岁，已经开始正式学习阶段。

**这个阶段需要平衡**：
- 学业：辅导作业、适当培训
- 身心健康：避免压力过大
- 亲子关系：保持沟通

**可用全部行为类型，但要根据孩子状态选择。**"""
    
    def _normalize_action_type(self, raw_action: str, age_group: str) -> str:
        """
        规范化行为类型：将LLM生成的各种表述映射到标准行为类型
        
        参数:
            raw_action: LLM生成的原始行为类型
            age_group: 当前年龄阶段
            
        返回:
            标准化后的行为类型
        """
        available_actions = self.AGE_APPROPRIATE_ACTIONS.get(age_group, self.AGE_APPROPRIATE_ACTIONS["primary"])
        
        # 1. 精确匹配：如果已经是标准行为类型
        if raw_action in available_actions:
            return raw_action
        
        # 2. 映射表匹配
        if raw_action in self.ACTION_TYPE_MAPPING:
            mapped_action = self.ACTION_TYPE_MAPPING[raw_action]
            if mapped_action in available_actions:
                logger.info(f"行为类型映射: {raw_action} -> {mapped_action}")
                return mapped_action
            # 映射结果不在允许列表中，继续尝试其他方法
        
        # 3. 关键词匹配（按优先级）
        keyword_mappings = [
            # 婴幼儿关键词
            (["抱", "逗", "哄", "安慰"], "亲子互动"),
            (["换尿布", "喂", "照料", "照顾"], "日常照料"),
            (["音乐", "颜色", "触摸", "声音"], "感官刺激"),
            (["晒太阳", "公园", "户外", "散步"], "户外活动"),
            (["故事", "绘本", "读书", "阅读"], "早期阅读"),
            (["安抚", "哭", "夜", "睡"], "安抚陪伴"),
            # 通用关键词
            (["陪", "一起"], "陪伴"),
            (["辅导", "作业", "功课", "学习"], "辅导"),
            (["游戏", "玩"], "游戏互动"),
            (["鼓励", "表扬", "夸"], "鼓励"),
            (["培训", "班", "课程"], "花钱培训"),
            (["严格", "批评", "要求"], "严格要求"),
            (["启蒙", "认识", "教"], "启蒙教育"),
            (["运动", "锻炼", "健康"], "健康教育"),
            (["聊", "谈", "沟通"], "沟通"),
        ]
        
        for keywords, action in keyword_mappings:
            for keyword in keywords:
                if keyword in raw_action:
                    if action in available_actions:
                        logger.info(f"关键词匹配: {raw_action} -> {action} (关键词: {keyword})")
                        return action
        
        # 4. 返回年龄段默认行为
        default_actions = {
            "infant": "亲子互动",
            "preschool": "陪伴",
            "primary": "陪伴"
        }
        default = default_actions.get(age_group, "陪伴")
        logger.warning(f"无法识别的行为类型「{raw_action}」，使用默认行为「{default}」")
        return default
    
    def _format_user_prompt(self, child_state: ChildState, family_state: FamilyState, event: str) -> str:
        """
        格式化用户提示，提供当前状态信息
        
        参数:
            child_state: 孩子当前状态
            family_state: 家庭当前状态
            event: 当前事件
            
        返回:
            格式化的用户提示
        """
        age = child_state.calculate_age(family_state.current_date)
        member_info = self._get_member_info()
        
        # 状态评估
        stress_status = "🔴 危险" if child_state.stress > 80 else ("🟡 警告" if child_state.stress > 60 else "🟢 正常")
        health_status = "🔴 危险" if child_state.physical_health < 50 else ("🟡 警告" if child_state.physical_health < 70 else "🟢 正常")
        knowledge_status = "🔴 不足" if child_state.knowledge < 40 else ("🟡 一般" if child_state.knowledge < 60 else "🟢 良好")
        
        # 获取当前成员的关系值
        relationship_key = f"{self.member}_relationship"
        relationship_value = getattr(child_state, relationship_key, 70.0)
        relationship_status = "🔴 紧张" if relationship_value < 50 else ("🟡 一般" if relationship_value < 70 else "🟢 良好")
        
        # 经济状态
        savings = family_state.family_savings
        economy_status = "🔴 紧张" if savings < 5000 else ("🟡 谨慎" if savings < 15000 else "🟢 宽裕")
        
        return f"""## 当前情况

### 时间
- 当前日期：{family_state.current_date.strftime('%Y年%m月%d日')}
- {member_info['child_call']}年龄：{age:.1f}岁

### {member_info['child_call']}状态
- 知识储备：{child_state.knowledge:.1f}/100 {knowledge_status}
- 压力值：{child_state.stress:.1f}/100 {stress_status}
- 身体健康：{child_state.physical_health:.1f}/100 {health_status}
- 与你的关系：{relationship_value:.1f}/100 {relationship_status}
- 与父亲关系：{child_state.father_relationship:.1f}/100
- 与母亲关系：{child_state.mother_relationship:.1f}/100
- 兴趣偏好：{', '.join(child_state.interests.get_top_interests(3)) if hasattr(child_state, 'interests') else '未知'}
- 当前敏感期：{', '.join(getattr(child_state, 'development_sensitivity', None).get_active_sensitivities().keys()) if hasattr(child_state, 'development_sensitivity') else '无'}

### 家庭经济状况
- 家庭存款：{savings:.0f}元 {economy_status}
- 父亲月薪：{family_state.father.salary:.0f}元
- 母亲月薪：{family_state.mother.salary:.0f}元

### 本周事件
{event}

---
请根据以上信息，以{member_info['role']}的身份做出本周的教育决策。记住你的决策会影响{member_info['child_call']}的成长！"""

    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        通过 LLM 生成教育决策
        
        参数:
            child_state: 孩子当前状态
            family_state: 家庭当前状态
            event: 当前周事件
            
        返回:
            决策结果字典
        """
        # 如果是 Mock 客户端，使用规则降级
        if self._is_mock or self._llm_client is None:
            return await self._fallback_decision(child_state, family_state, event)
        
        try:
            # 构建消息
            system_prompt = self._get_enhanced_system_prompt(child_state, family_state)
            user_prompt = self._format_user_prompt(child_state, family_state, event)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用 LLM API
            logger.info(f"[{self.model_name}] 正在调用 LLM 生成决策...")
            
            response = await self._llm_client.chat_completion(
                messages=messages,
                model=self._model_config["model"],
                temperature=0.7,
                max_tokens=500
            )
            
            # 解析响应
            content = response["choices"][0]["message"]["content"]
            logger.debug(f"[{self.model_name}] LLM 原始响应: {content[:200]}...")
            
            # 解析 JSON 决策
            decision = self._parse_decision(content, child_state, family_state)
            
            # 添加 LLM 标记
            decision["llm_generated"] = True
            decision["model"] = self.model_name
            
            logger.info(f"[{self.model_name}] LLM 决策成功: {decision['action_type']}")
            return decision
            
        except Exception as e:
            logger.warning(f"[{self.model_name}] LLM 调用失败: {e}，使用降级决策")
            return await self._fallback_decision(child_state, family_state, event)
    
    def _parse_decision(self, content: str, child_state: ChildState, family_state: FamilyState) -> Dict[str, Any]:
        """
        解析 LLM 返回的决策内容
        
        参数:
            content: LLM 返回的原始内容
            child_state: 孩子状态
            family_state: 家庭状态
            
        返回:
            解析后的决策字典
        """
        # 尝试提取 JSON
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        
        if json_match:
            try:
                decision = json.loads(json_match.group())
                
                # 验证必要字段
                if not all(key in decision for key in ["action_type", "dialogue", "cost"]):
                    raise ValueError("缺少必要字段")
                
                # 验证和规范化
                decision = self._validate_decision(decision, child_state, family_state)
                return decision
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}")
        
        # 解析失败，尝试从文本中提取关键信息
        return self._extract_decision_from_text(content, child_state, family_state)
    
    def _validate_decision(self, decision: Dict[str, Any], child_state: ChildState, family_state: FamilyState) -> Dict[str, Any]:
        """
        验证和规范化决策
        
        参数:
            decision: 原始决策
            child_state: 孩子状态
            family_state: 家庭状态
            
        返回:
            验证后的决策
        """
        age_group = child_state.get_age_group(family_state.current_date)
        available_actions = self.AGE_APPROPRIATE_ACTIONS.get(age_group, self.AGE_APPROPRIATE_ACTIONS["primary"])
        
        # 使用规范化方法处理 action_type
        raw_action = decision.get("action_type", "")
        if raw_action not in available_actions:
            # 尝试规范化
            normalized_action = self._normalize_action_type(raw_action, age_group)
            decision["action_type"] = normalized_action
            decision["_original_action"] = raw_action  # 保留原始值用于调试
        
        # 验证 cost
        try:
            cost = float(decision.get("cost", 0))
            # 经济约束检查
            if family_state.family_savings < 5000:
                cost = 0
            elif family_state.family_savings < 10000 and cost > 100:
                cost = 100
            decision["cost"] = max(0, min(cost, 500))  # 限制最大花费
        except (ValueError, TypeError):
            decision["cost"] = 0
        
        # 确保 dialogue 存在
        if not decision.get("dialogue"):
            member_info = self._get_member_info()
            age = child_state.calculate_age(family_state.current_date)
            # 根据年龄生成更合适的默认对话
            if age < 1:
                decision["dialogue"] = f"（轻轻抱着孩子）宝宝乖，{member_info['role']}在呢。"
            elif age < 3:
                decision["dialogue"] = f"宝贝，{member_info['role']}陪你玩好不好？"
            else:
                decision["dialogue"] = f"{member_info['child_call']}，今天{member_info['role']}陪你。"
        
        return decision
    
    def _extract_decision_from_text(self, content: str, child_state: ChildState, family_state: FamilyState) -> Dict[str, Any]:
        """
        从非 JSON 文本中提取决策信息
        
        参数:
            content: 原始文本
            child_state: 孩子状态
            family_state: 家庭状态
            
        返回:
            提取的决策字典
        """
        member_info = self._get_member_info()
        age_group = child_state.get_age_group(family_state.current_date)
        available_actions = self.AGE_APPROPRIATE_ACTIONS.get(age_group, self.AGE_APPROPRIATE_ACTIONS["primary"])
        
        # 尝试匹配行动类型
        action_type = "陪伴"
        for action in available_actions:
            if action in content:
                action_type = action
                break
        
        # 提取对话（寻找引号内的内容）
        dialogue_match = re.search(r'[""「]([^""」]+)[""」]', content)
        if dialogue_match:
            dialogue = dialogue_match.group(1)
        else:
            dialogue = f"{member_info['child_call']}，今天{member_info['role']}陪你做一些有意义的事情。"
        
        # 提取花费
        cost_match = re.search(r'(\d+)\s*元', content)
        cost = int(cost_match.group(1)) if cost_match else 0
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost,
            "reasoning": "从文本提取的决策"
        }
    
    async def _fallback_decision(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        降级决策：当 LLM 不可用时使用规则生成决策
        
        参数:
            child_state: 孩子状态
            family_state: 家庭状态
            event: 当前事件
            
        返回:
            降级决策字典
        """
        import random
        
        member_info = self._get_member_info()
        age_group = child_state.get_age_group(family_state.current_date)
        available_actions = self.AGE_APPROPRIATE_ACTIONS.get(age_group, self.AGE_APPROPRIATE_ACTIONS["primary"])
        
        # 根据状态选择行动
        if child_state.stress > 80:
            # 高压力：减压
            preferred = ["陪伴", "游戏互动", "鼓励"]
        elif child_state.stress > 60:
            preferred = ["鼓励", "陪伴", "游戏互动", "启蒙教育"]
        elif child_state.knowledge < 40:
            # 知识不足：学习
            preferred = ["辅导", "简单辅导", "启蒙教育", "花钱培训"]
        elif child_state.physical_health < 60:
            # 健康问题
            preferred = ["陪伴", "健康教育", "游戏互动"]
        else:
            # 正常情况
            preferred = available_actions
        
        # 兴趣与敏感期偏好（提高选择概率）
        interest_bias = {
            "阅读": ["早期阅读", "启蒙教育", "简单辅导", "辅导"],
            "音乐": ["简单兴趣培养", "启蒙教育"],
            "美术": ["简单兴趣培养", "创新活动"],
            "运动": ["户外活动", "健康教育", "游戏互动", "实践活动"],
            "科学": ["创新活动", "实践活动", "启蒙教育"],
            "游戏": ["游戏互动"],
            "社交": ["社交接触", "沟通", "游戏互动"],
            "自然": ["户外活动", "实践活动"]
        }
        sensitivity_bias = {
            "语言": ["早期阅读", "启蒙教育", "沟通"],
            "秩序": ["日常照料", "监督学习"],
            "感官": ["感官刺激", "游戏互动"],
            "动作": ["户外活动", "游戏互动", "实践活动"],
            "社交": ["社交接触", "沟通", "游戏互动"],
            "数学": ["启蒙教育", "简单辅导", "辅导"],
            "阅读": ["早期阅读", "启蒙教育"]
        }
        
        top_interests = []
        if hasattr(child_state, "interests"):
            top_interests = child_state.interests.get_top_interests(2)
        
        active_sensitivities = []
        if hasattr(child_state, "development_sensitivity"):
            active_sensitivities = list(child_state.development_sensitivity.get_active_sensitivities().keys())
        
        bias_actions = []
        for interest in top_interests:
            bias_actions.extend(interest_bias.get(interest, []))
        for sensitivity in active_sensitivities:
            bias_actions.extend(sensitivity_bias.get(sensitivity, []))
        
        if bias_actions:
            preferred = preferred + [a for a in bias_actions if a in available_actions]
        
        # 从偏好中选择可用的行动
        valid_actions = [a for a in preferred if a in available_actions]
        if not valid_actions:
            valid_actions = available_actions
        
        action_type = random.choice(valid_actions)
        
        # 确定花费
        cost_map = {
            "陪伴": random.randint(0, 50),
            "游戏互动": random.randint(0, 30),
            "启蒙教育": 0,
            "简单辅导": 0,
            "辅导": 0,
            "鼓励": 0,
            "沟通": 0,
            "健康教育": 0,
            "花钱培训": random.randint(150, 300),
            "简单兴趣培养": random.randint(80, 150),
            "严格要求": 0,
            "监督学习": 0,
            "创新活动": random.randint(50, 100),
            "个性化计划": 0,
            "实践活动": random.randint(50, 100)
        }
        
        cost = cost_map.get(action_type, 0)
        
        # 经济约束
        if family_state.family_savings < 5000:
            cost = 0
        elif family_state.family_savings < 10000 and cost > 100:
            cost = 0
        
        # 生成对话
        dialogue = self._generate_fallback_dialogue(action_type, member_info, event)
        
        return {
            "action_type": action_type,
            "dialogue": dialogue,
            "cost": cost,
            "reasoning": "规则降级决策",
            "llm_generated": False,
            "model": self.model_name
        }
    
    def _generate_fallback_dialogue(self, action_type: str, member_info: Dict[str, str], event: str) -> str:
        """生成降级对话"""
        import random
        
        child_call = member_info["child_call"]
        role = member_info["role"]
        
        dialogues = {
            "陪伴": [
                f"{child_call}，今天{role}陪你去公园玩，放松一下。",
                f"{child_call}，今天{role}带你出去走走，呼吸新鲜空气。",
                f"{child_call}，今天{role}专门陪你，想做什么都可以。"
            ],
            "启蒙教育": [
                f"{child_call}，今天{role}教你认识一些新的东西。",
                f"{child_call}，来，{role}给你讲一个有趣的故事。",
                f"{child_call}，今天我们一起学习新知识。"
            ],
            "游戏互动": [
                f"{child_call}，今天{role}陪你玩游戏，好不好？",
                f"{child_call}，我们一起玩积木吧。",
                f"{child_call}，今天{role}教你一个新游戏。"
            ],
            "辅导": [
                f"{child_call}，来，{role}帮你看看功课。",
                f"{child_call}，今天{role}辅导你一下作业。",
                f"{child_call}，有什么不懂的题目，{role}来帮你。"
            ],
            "鼓励": [
                f"{child_call}，你最近表现很不错，继续加油！",
                f"{child_call}，{role}相信你可以做得更好！",
                f"{child_call}，不管结果怎样，{role}都为你骄傲。"
            ],
            "花钱培训": [
                f"{child_call}，{role}给你报了一个培训班，好好学习。",
                f"{child_call}，这个课程可以帮助你进步。",
                f"{child_call}，{role}给你买了学习资料。"
            ],
            "简单辅导": [
                f"{child_call}，来，{role}帮你复习一下。",
                f"{child_call}，今天我们一起练习。"
            ],
            "健康教育": [
                f"{child_call}，要注意身体健康，早睡早起。",
                f"{child_call}，多运动，身体才会棒棒的。"
            ],
            "沟通": [
                f"{child_call}，最近有什么心事吗？可以跟{role}说说。",
                f"{child_call}，{role}想跟你聊聊天。"
            ]
        }
        
        options = dialogues.get(action_type, [f"{child_call}，今天{role}陪你。"])
        return random.choice(options)


# 为每个模型创建具体的 Agent 类
class DeepSeekLLMAgent(LLMDecisionAgent):
    """DeepSeek 模型的 LLM 决策智能体"""
    pass


class QwenLLMAgent(LLMDecisionAgent):
    """Qwen 模型的 LLM 决策智能体"""
    pass


class KimiLLMAgent(LLMDecisionAgent):
    """Kimi 模型的 LLM 决策智能体"""
    pass


class ChatGPTLLMAgent(LLMDecisionAgent):
    """ChatGPT 模型的 LLM 决策智能体"""
    pass


class GeminiLLMAgent(LLMDecisionAgent):
    """Gemini 模型的 LLM 决策智能体"""
    pass


class ClaudeLLMAgent(LLMDecisionAgent):
    """Claude 模型的 LLM 决策智能体"""
    pass


class GrokLLMAgent(LLMDecisionAgent):
    """Grok 模型的 LLM 决策智能体"""
    pass


# 模型名称到 Agent 类的映射
LLM_AGENT_CLASSES = {
    "deepseek": DeepSeekLLMAgent,
    "qwen": QwenLLMAgent,
    "kimi": KimiLLMAgent,
    "chatgpt": ChatGPTLLMAgent,
    "gemini": GeminiLLMAgent,
    "claude": ClaudeLLMAgent,
    "grok": GrokLLMAgent
}
