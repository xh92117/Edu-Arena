"""
个性化对话系统

实现更拟人化的对话生成：
1. 口头禅和个性化表达
2. 情境记忆（记住上周说了什么）
3. 情绪影响对话风格
4. 地域特色表达（可选）
"""

import random
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class DialogueMemory:
    """对话记忆"""
    week: int
    date: str
    speaker: str
    action_type: str
    dialogue: str
    child_response: str = ""  # 孩子的回应（模拟）
    was_effective: bool = True


class PersonalizedDialogueSystem:
    """
    个性化对话系统
    
    为每个家庭成员生成符合其性格的对话
    """
    
    # 口头禅库
    CATCHPHRASES = {
        "father": {
            "严厉": [
                "听我的！",
                "少废话！",
                "你看看别人家的孩子！",
                "我说的你听不懂吗？",
                "你给我好好学！",
                "爸说了算！"
            ],
            "温和": [
                "来，咱们慢慢说。",
                "没事儿，爸帮你。",
                "别着急，慢慢来。",
                "爸相信你。",
                "有爸在呢。"
            ],
            "焦虑": [
                "你可得争口气啊！",
                "再不努力可怎么办？",
                "我这都是为你好！",
                "你看看人家xxx...",
                "爸可就指望你了！"
            ],
            "沉稳": [
                "想清楚再做。",
                "不急，一步一步来。",
                "做事要有条理。",
                "爸跟你讲个道理。"
            ]
        },
        "mother": {
            "温和": [
                "宝贝，妈在呢。",
                "乖，听话。",
                "妈妈爱你。",
                "没关系的，下次再努力。",
                "妈相信你。"
            ],
            "焦虑": [
                "妈担心你啊！",
                "你怎么不争气呢？",
                "别让妈操心了！",
                "妈都是为你好！"
            ],
            "开明": [
                "你自己决定吧。",
                "妈尊重你的选择。",
                "有什么想法跟妈说说？",
                "妈觉得你做得挺好。"
            ]
        },
        "grandfather": {
            "传统": [
                "听爷的话！",
                "爷那个年代...",
                "吃得苦中苦，方为人上人。",
                "书中自有黄金屋！",
                "我们那时候哪有这条件！",
                "年轻人要上进！"
            ],
            "慈祥": [
                "孙女儿真乖。",
                "爷给你买好吃的。",
                "别累着，歇歇。",
                "爷的好孙女。"
            ]
        },
        "grandmother": {
            "慈祥": [
                "我的乖孙女！",
                "奶给你做好吃的。",
                "瘦了，多吃点！",
                "学累了就歇歇。",
                "奶的心肝儿！",
                "别饿着肚子。"
            ],
            "溺爱": [
                "孙女想要啥奶买！",
                "谁敢欺负我孙女！",
                "学那么多干啥，身体要紧！",
                "奶心疼你。"
            ]
        }
    }
    
    # 情绪影响的表达方式
    EMOTIONAL_MODIFIERS = {
        "平静": {"prefix": "", "suffix": "", "tone": "normal"},
        "开心": {"prefix": "（笑着说）", "suffix": "！", "tone": "happy"},
        "焦虑": {"prefix": "（有些焦虑地）", "suffix": "...", "tone": "anxious"},
        "压力大": {"prefix": "（叹了口气）", "suffix": "。", "tone": "stressed"},
        "生气": {"prefix": "（提高声音）", "suffix": "！", "tone": "angry"},
        "疲惫": {"prefix": "（疲惫地）", "suffix": "...", "tone": "tired"},
        "骄傲": {"prefix": "（欣慰地）", "suffix": "！", "tone": "proud"},
        "担忧": {"prefix": "（担心地）", "suffix": "...", "tone": "worried"}
    }
    
    # 孩子回应模板（用于模拟互动）
    CHILD_RESPONSES = {
        "happy": [
            "好的，{role}！",
            "嗯！",
            "知道了~",
            "我会努力的！",
            "谢谢{role}！"
        ],
        "reluctant": [
            "好吧...",
            "知道了...",
            "嗯...",
            "我尽量...",
            "哦。"
        ],
        "resistant": [
            "为什么？",
            "我不想...",
            "能不能不要...",
            "别管我！",
            "烦死了！"
        ],
        "excited": [
            "太好了！",
            "真的吗？！",
            "耶！",
            "我最喜欢{role}了！",
            "好开心！"
        ],
        "sad": [
            "好吧...",
            "我知道了...",
            "对不起...",
            "我会努力的...",
            "别生气了..."
        ]
    }
    
    def __init__(self, member: str, personality: str):
        """
        初始化对话系统
        
        参数:
            member: 家庭成员类型
            personality: 性格类型
        """
        self.member = member
        self.personality = personality
        self.dialogue_history: deque = deque(maxlen=12)  # 记住最近12周的对话
        self.effective_dialogues: List[str] = []  # 有效的对话模式
        self.ineffective_dialogues: List[str] = []  # 无效的对话模式
        
    def generate_dialogue(self, 
                          action_type: str,
                          child_age: float,
                          child_stress: float,
                          relationship: float,
                          emotional_state: str,
                          context: Dict[str, Any] = None) -> str:
        """
        生成个性化对话
        
        参数:
            action_type: 行为类型
            child_age: 孩子年龄
            child_stress: 孩子压力
            relationship: 亲子关系
            emotional_state: 家长当前情绪
            context: 上下文信息
            
        返回:
            生成的对话
        """
        # 获取基础对话模板
        base_dialogue = self._get_base_dialogue(action_type, child_age)
        
        # 添加口头禅（30%概率）
        if random.random() < 0.3:
            catchphrase = self._get_catchphrase()
            if catchphrase:
                base_dialogue = f"{catchphrase} {base_dialogue}"
        
        # 根据情绪修饰对话
        dialogue = self._apply_emotional_modifier(base_dialogue, emotional_state)
        
        # 根据上下文调整
        if context:
            dialogue = self._adjust_for_context(dialogue, context)
        
        # 如果关系不好，语气可能更生硬
        if relationship < 50:
            dialogue = self._make_dialogue_colder(dialogue)
        
        # 如果孩子压力大且是严格类行为，可能会软化语气
        if child_stress > 70 and action_type in ["严格要求", "监督学习"]:
            dialogue = self._soften_dialogue(dialogue)
        
        return dialogue
    
    def _get_base_dialogue(self, action_type: str, child_age: float) -> str:
        """获取基础对话模板"""
        role_names = {
            "father": "爸爸",
            "mother": "妈妈",
            "grandfather": "爷爷",
            "grandmother": "奶奶"
        }
        role = role_names.get(self.member, "家长")
        child_call = "孙女" if self.member in ["grandfather", "grandmother"] else "女儿"
        
        # 婴儿期特殊处理
        if child_age < 1:
            return random.choice([
                f"（轻声地）宝宝乖，{role}在呢。",
                f"（温柔地）小宝贝，{role}抱抱。",
                f"（哼着歌）睡吧睡吧，{role}守着你。"
            ])
        elif child_age < 3:
            return random.choice([
                f"宝贝，{role}陪你玩。",
                f"乖，{role}在这儿呢。",
                f"宝贝真棒！"
            ])
        
        # 根据行为类型生成对话
        dialogue_templates = {
            "辅导": [
                f"{child_call}，来，{role}帮你看看这道题。",
                f"{child_call}，有不会的吗？{role}来帮你。",
                f"来，咱们一起学。"
            ],
            "鼓励": [
                f"{child_call}，{role}相信你一定行！",
                f"你最近进步很大，{role}看在眼里。",
                f"不管结果怎样，{role}都为你骄傲。"
            ],
            "陪伴": [
                f"{child_call}，今天{role}哪儿也不去，就陪着你。",
                f"走，咱们出去逛逛。",
                f"今天{role}专门陪你。"
            ],
            "严格要求": [
                f"{child_call}，学习不能松懈！",
                f"你得给我认真点！",
                f"现在不努力，以后怎么办？"
            ],
            "游戏互动": [
                f"今天咱们玩点什么？",
                f"{role}陪你玩一会儿。",
                f"想玩什么，{role}都陪你。"
            ],
            "启蒙教育": [
                f"来，{role}教你认识新东西。",
                f"今天咱们学点有趣的。",
                f"{role}给你讲个故事吧。"
            ],
            "沟通": [
                f"{child_call}，最近怎么样？有什么想说的吗？",
                f"来，跟{role}聊聊。",
                f"有心事可以告诉{role}。"
            ]
        }
        
        templates = dialogue_templates.get(action_type, [f"{child_call}，{role}陪你。"])
        return random.choice(templates)
    
    def _get_catchphrase(self) -> Optional[str]:
        """获取口头禅"""
        member_phrases = self.CATCHPHRASES.get(self.member, {})
        personality_phrases = member_phrases.get(self.personality, [])
        
        if personality_phrases:
            return random.choice(personality_phrases)
        
        # 降级：使用任意口头禅
        all_phrases = []
        for phrases in member_phrases.values():
            all_phrases.extend(phrases)
        
        return random.choice(all_phrases) if all_phrases else None
    
    def _apply_emotional_modifier(self, dialogue: str, emotional_state: str) -> str:
        """应用情绪修饰"""
        modifier = self.EMOTIONAL_MODIFIERS.get(emotional_state, self.EMOTIONAL_MODIFIERS["平静"])
        
        prefix = modifier.get("prefix", "")
        suffix = modifier.get("suffix", "")
        
        # 如果对话已经有标点，不再添加
        if dialogue.endswith(("！", "。", "？", "...", "~")):
            suffix = ""
        
        return f"{prefix}{dialogue}{suffix}"
    
    def _adjust_for_context(self, dialogue: str, context: Dict[str, Any]) -> str:
        """根据上下文调整对话"""
        # 如果上周也做了同样的事，可以有连续性
        last_action = context.get("last_action")
        if last_action and last_action == context.get("current_action"):
            prefixes = [
                "和上周一样，",
                "继续",
                "还是"
            ]
            dialogue = random.choice(prefixes) + dialogue.lstrip("，。")
        
        # 如果上周的行为失败了
        if context.get("last_action_failed"):
            dialogue = "这次换个方法，" + dialogue
        
        return dialogue
    
    def _make_dialogue_colder(self, dialogue: str) -> str:
        """使对话更生硬（关系不好时）"""
        # 移除温柔的表达
        cold_replacements = [
            ("宝贝", "你"),
            ("乖", ""),
            ("我们", "你"),
            ("咱们", "你"),
        ]
        for old, new in cold_replacements:
            dialogue = dialogue.replace(old, new)
        return dialogue
    
    def _soften_dialogue(self, dialogue: str) -> str:
        """软化对话（孩子压力大时）"""
        softening_prefixes = [
            "我知道你累了，但",
            "别太有压力，",
            "慢慢来，"
        ]
        return random.choice(softening_prefixes) + dialogue.lstrip("，。")
    
    def record_dialogue(self, week: int, date: str, action_type: str, 
                         dialogue: str, was_effective: bool = True):
        """记录对话"""
        memory = DialogueMemory(
            week=week,
            date=date,
            speaker=self.member,
            action_type=action_type,
            dialogue=dialogue,
            was_effective=was_effective
        )
        self.dialogue_history.append(memory)
        
        # 学习有效/无效的对话模式
        if was_effective:
            if dialogue not in self.effective_dialogues:
                self.effective_dialogues.append(dialogue)
        else:
            if dialogue not in self.ineffective_dialogues:
                self.ineffective_dialogues.append(dialogue)
    
    def get_dialogue_context(self) -> Dict[str, Any]:
        """获取对话上下文"""
        context = {}
        
        if self.dialogue_history:
            last = self.dialogue_history[-1]
            context["last_action"] = last.action_type
            context["last_dialogue"] = last.dialogue
            context["last_action_failed"] = not last.was_effective
        
        return context
    
    def simulate_child_response(self, 
                                 action_type: str,
                                 child_stress: float,
                                 child_emotional_state: str,
                                 relationship: float) -> str:
        """
        模拟孩子的回应
        
        参数:
            action_type: 行为类型
            child_stress: 孩子压力
            child_emotional_state: 孩子情绪
            relationship: 亲子关系
            
        返回:
            孩子的回应
        """
        role_names = {
            "father": "爸",
            "mother": "妈",
            "grandfather": "爷爷",
            "grandmother": "奶奶"
        }
        role = role_names.get(self.member, "")
        
        # 根据情绪和关系决定回应类型
        if child_stress > 80:
            response_type = "resistant" if relationship < 60 else "sad"
        elif relationship < 40:
            response_type = "reluctant"
        elif child_emotional_state in ["开心", "兴奋"]:
            response_type = "excited"
        elif action_type in ["严格要求", "监督学习"]:
            response_type = "reluctant" if random.random() < 0.5 else "happy"
        else:
            response_type = "happy"
        
        responses = self.CHILD_RESPONSES.get(response_type, self.CHILD_RESPONSES["happy"])
        response = random.choice(responses)
        
        return response.format(role=role)


# 对话系统工厂
class DialogueSystemFactory:
    """对话系统工厂"""
    
    _instances: Dict[str, PersonalizedDialogueSystem] = {}
    
    @classmethod
    def get_dialogue_system(cls, member: str, personality: str) -> PersonalizedDialogueSystem:
        """获取或创建对话系统实例"""
        key = f"{member}_{personality}"
        if key not in cls._instances:
            cls._instances[key] = PersonalizedDialogueSystem(member, personality)
        return cls._instances[key]
    
    @classmethod
    def reset(cls):
        """重置所有实例"""
        cls._instances = {}
