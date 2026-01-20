from abc import ABC, abstractmethod
from typing import Dict, Any
from src.core.state import ChildState, FamilyState


class FamilyAgent(ABC):
    """
    家庭成员角色智能体的抽象基类，所有模型适配器必须实现此类的接口
    
    角色设定：普通工薪阶层中国家庭成员，根据member参数确定具体角色
    特点：
    - 语言风格：朴实无华，带有中国家长的传统观念
    - 价值观：重视教育，希望孩子通过学习改变命运
    - 行为模式：会根据孩子的表现调整教育方式，经济有限但愿意为孩子教育投资
    - 情感表达：根据不同角色有不同的表达风格
    """
    
    def __init__(self, model_name: str, member: str = "father"):
        """
        初始化家庭成员智能体
        
        参数:
            model_name: 模型名称
            member: 家庭成员角色，可选值：father, mother, grandfather, grandmother
        """
        self.model_name = model_name
        self.member = member
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """
        获取系统提示模板，定义家庭成员角色的人设特点
        
        返回:
            系统提示字符串
        """
        # 家庭成员角色映射
        member_info = {
            "father": {
                "role": "父亲",
                "identity": "普通技术工人，大专学历",
                "personality": "严厉，不善于直接表达爱意，更倾向于通过实际行动关心女儿",
                "language_style": "朴实无华，表达直接，偶尔使用谚语或俗语"
            },
            "mother": {
                "role": "母亲",
                "identity": "办公室文员，大专学历",
                "personality": "温和，善于表达爱意，关注女儿的身心健康",
                "language_style": "亲切温暖，使用口语化表达"
            },
            "grandfather": {
                "role": "祖父",
                "identity": "退休工人，高中学历",
                "personality": "传统，重视教育，喜欢讲述过去的故事",
                "language_style": "传统保守，带有时代特色"
            },
            "grandmother": {
                "role": "祖母",
                "identity": "家庭主妇，初中学历",
                "personality": "慈祥，宠爱孙女，关注生活细节",
                "language_style": "温柔慈祥，充满爱意"
            }
        }
        
        info = member_info[self.member]
        
        return f"""
你现在需要扮演一个2010-2030年间的中国普通工薪阶层{info['role']}，你的身份设定如下：

1. 基本信息：
   - 身份：{info['identity']}
   - 家庭：有一个2010年出生的孙女/女儿，家庭属于二线城市普通工薪家庭
   - 经济状况：家庭月收入约10000元（2010年基准），生活节俭但愿意为孩子教育投资
   - 居住环境：二线城市，有一套贷款购买的小户型住房

2. 性格特点：
   - {info['personality']}
   - 传统观念：重视教育，认为"知识改变命运"，希望孙女/女儿能考上好大学
   - 经济观念：生活节俭，但愿意为孙女/女儿的教育投资
   - 教育方式：会根据孙女/女儿的表现调整，既会严格要求，也会给予鼓励
   - 压力来源：担心孙女/女儿的未来，害怕无法给她提供足够的教育资源

3. 语言风格：
   - {info['language_style']}
   - 避免使用过于书面化或复杂的词汇
   - 表达符合你的角色身份

4. 行为准则：
   - 优先考虑孙女/女儿的教育问题
   - 做出决策时会考虑家庭经济状况
   - 关注孙女/女儿的身心健康
   - 会受到当时社会环境和教育政策的影响

5. 当前情境：
   - 你需要根据提供的孙女/女儿状态和当前发生的事件，决定本周的教育方式
   - 你的决策将影响孙女/女儿的知识储备、压力值、与家庭成员的关系和身体素质
   - 每次决策需要考虑家庭经济状况，避免过度消费

现在，请根据提供的孙女/女儿状态和当前事件，以{info['role']}的身份做出决策，并按照以下JSON格式返回：
{{
    "action_type": "辅导",  # 行动类型：辅导、斥责、鼓励、花钱培训等
    "dialogue": "孙女/女儿，今天{info['role']}帮你辅导一下数学作业吧。",  # 对你孙女/女儿说的话
    "cost": 0  # 该行为产生的花费，单位：元
    "inner_thought": "你内心的真实想法，考虑到当前情境和孙女/女儿的状态，考虑到家庭经济状况"
}}

注意：
1. 请确保返回的JSON格式正确，不要包含任何额外的解释或说明
2. 行动类型要具体，符合{info['role']}的身份和当前情境
3. 对话要符合{info['role']}的语言风格，真实自然
4. 花费要合理，符合普通工薪家庭的经济状况
5. 请使用符合{info['role']}身份的称呼，如父亲使用"女儿"，祖父/祖母使用"孙女"
"""
    
    @abstractmethod
    async def decide(self, child_state: ChildState, family_state: FamilyState, event: str) -> Dict[str, Any]:
        """
        决定本周的教育行动
        
        参数:
            child_state: 孩子当前的状态
            family_state: 家庭当前的状态
            event: 当前周发生的事件描述
            
        返回:
            包含以下键的字典：
            - action_type: 行动类型，如"辅导"、"斥责"、"鼓励"、"花钱培训"等
            - dialogue: 父亲对孩子说的具体内容
            - cost: 该行为产生的花费金额
        """
        pass
    
    def _format_prompt(self, child_state: ChildState, family_state: FamilyState, event: str) -> str:
        """
        格式化提示信息，将状态和事件转换为模型可理解的格式
        
        参数:
            child_state: 孩子当前的状态
            family_state: 家庭当前的状态
            event: 当前周发生的事件描述
            
        返回:
            格式化后的提示字符串
        """
        # 获取家庭成员的称呼
        member_calls = {
            "father": "爸爸",
            "mother": "妈妈",
            "grandfather": "爷爷",
            "grandmother": "奶奶"
        }
        
        call = member_calls.get(self.member, "家长")
        
        return f"""
当前日期：{family_state.current_date.strftime('%Y-%m-%d')}

女儿状态：
- 出生日期：{child_state.birth_date.strftime('%Y-%m-%d')}
- 智商：{child_state.iq}
- 知识储备：{child_state.knowledge:.1f}分（0-100分制）
- 压力值：{child_state.stress:.1f}分（0-100分制）
- 与父亲关系：{child_state.father_relationship:.1f}分（0-100分制）
- 与母亲关系：{child_state.mother_relationship:.1f}分（0-100分制）
- 与祖父关系：{child_state.grandfather_relationship:.1f}分（0-100分制）
- 与祖母关系：{child_state.grandmother_relationship:.1f}分（0-100分制）
- 身体素质：{child_state.physical_health:.1f}分（0-100分制）
- 兴趣偏好：{', '.join(child_state.interests.get_top_interests(3)) if hasattr(child_state, 'interests') else '未知'}
- 当前敏感期：{', '.join(getattr(child_state, 'development_sensitivity', None).get_active_sensitivities().keys()) if hasattr(child_state, 'development_sensitivity') else '无'}

家庭状态：
- 父亲月薪：{family_state.father.salary:.1f}元，影响力权重：{family_state.father.influence_weight:.1f}
- 母亲月薪：{family_state.mother.salary:.1f}元，影响力权重：{family_state.mother.influence_weight:.1f}
- 祖父月薪：{family_state.grandfather.salary:.1f}元，影响力权重：{family_state.grandfather.influence_weight:.1f}
- 祖母月薪：{family_state.grandmother.salary:.1f}元，影响力权重：{family_state.grandmother.influence_weight:.1f}
- 家庭存款：{family_state.family_savings:.1f}元

当前事件：
{event}

请根据以上信息，以{call}的身份做出本周的教育决策。
"""
